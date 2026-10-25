"""Two-phase model: a classical network writes four angles, then a VQC reads them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from data.creditcard import FEATURE_COLUMNS, RAW_COLUMNS
from models.baseline import BaselineVQC
from models.circuit import qubit_pairs
from models.config import VQCConfig
from models.runtime import TORCH_DEVICE, as_quantum, set_seed, to_numpy


@dataclass(frozen=True)
class TwoPhaseConfig:
    vqc: VQCConfig = field(default_factory=VQCConfig)
    hidden_size: int = 8
    mlp_epochs: int = 20
    mlp_learning_rate: float = 1e-2
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 64
    separation_weight: float = 0.5
    spread_weight: float = 0.001
    separation_margin: float = 0.5
    spread_target_std: float = 0.25

    @property
    def batch_size(self) -> int:
        return self.vqc.batch_size


class FeatureScaler:
    """Freeze a standardiser for ``Time`` and ``Amount``. ``V1``–``V28`` stay as they are."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> FeatureScaler:
        self.scaler.fit(frame[list(RAW_COLUMNS)])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = frame.loc[:, FEATURE_COLUMNS].copy()
        out[list(RAW_COLUMNS)] = self.scaler.transform(frame[list(RAW_COLUMNS)])
        return out.to_numpy(dtype=np.float64)


class AngleMLP(torch.nn.Module):
    """30 features → 4 angles in ``[0, π]``, plus a classical fraud head on those angles."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden = torch.nn.Linear(len(FEATURE_COLUMNS), hidden_size)
        self.angles_layer = torch.nn.Linear(hidden_size, 4)
        self.classifier = torch.nn.Linear(4, 1)

    def angles(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(features))
        return torch.sigmoid(self.angles_layer(hidden)) * torch.pi

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles(features)
        logits = self.classifier(angles).squeeze(-1)
        return angles, logits


def feature_map_coordinates(
    angles: torch.Tensor, connectivity: str = "all"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates the ZZ feature map actually uses.

    ``RZ(2x)`` depends on ``(cos 2x, sin 2x)``. Each pairwise block depends on
    ``(π - x_i)(π - x_j)``. ``connectivity`` selects the same pairs as the circuit.
    """
    rotation = torch.cat([torch.cos(2 * angles), torch.sin(2 * angles)], dim=-1)
    pairs = [
        (torch.pi - angles[:, i]) * (torch.pi - angles[:, j])
        for i, j in qubit_pairs(angles.shape[1], connectivity)
    ]
    return rotation, torch.stack(pairs, dim=-1)


def _block_distance(values: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    return (values - prototype).pow(2).mean(dim=-1)


def separation_loss(
    angles: torch.Tensor,
    labels: torch.Tensor,
    prototypes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    margin: float,
    connectivity: str = "all",
) -> torch.Tensor:
    """Distance to the matching class prototype, plus a push away from the other up to ``margin``."""
    rotation, pairwise = feature_map_coordinates(angles, connectivity)
    rot_legit, rot_fraud, pair_legit, pair_fraud = prototypes
    distance_legit = 0.5 * (
        _block_distance(rotation, rot_legit) + _block_distance(pairwise, pair_legit)
    )
    distance_fraud = 0.5 * (
        _block_distance(rotation, rot_fraud) + _block_distance(pairwise, pair_fraud)
    )
    fraud = labels > 0.5
    distance_same = torch.where(fraud, distance_fraud, distance_legit)
    distance_other = torch.where(fraud, distance_legit, distance_fraud)
    return (distance_same + F.relu(margin - distance_other)).mean()


def spread_loss(angles: torch.Tensor, target_std: float) -> torch.Tensor:
    """Keep each angle from collapsing, and keep the four angles from copying each other."""
    standard_deviation = angles.std(dim=0, unbiased=False).clamp_min(1e-8)
    variance_hinge = F.relu(target_std - standard_deviation).sum()
    centered = angles - angles.mean(dim=0, keepdim=True)
    normalised = centered / standard_deviation
    correlation = normalised.T @ normalised / angles.shape[0]
    off_diagonal = correlation - torch.eye(angles.shape[1], device=angles.device, dtype=angles.dtype)
    return variance_hinge + off_diagonal.pow(2).sum()


def class_prototypes(
    angles: torch.Tensor, labels: torch.Tensor, connectivity: str = "all"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation, pairwise = feature_map_coordinates(angles, connectivity)
    legit = labels < 0.5
    fraud = ~legit
    if not bool(legit.any()) or not bool(fraud.any()):
        raise ValueError("Both classes must be present to build prototypes.")
    return (
        rotation[legit].mean(dim=0).detach(),
        rotation[fraud].mean(dim=0).detach(),
        pairwise[legit].mean(dim=0).detach(),
        pairwise[fraud].mean(dim=0).detach(),
    )


class TwoPhaseVQC:
    def __init__(self, config: TwoPhaseConfig | None = None, vqc_optimizer_fn: Callable | None = None):
        self.config = config or TwoPhaseConfig()
        self.vqc = BaselineVQC(self.config.vqc)
        self.vqc_optimizer_fn = vqc_optimizer_fn
        self.scaler = FeatureScaler()
        self.frontend: AngleMLP | None = None

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        if self.frontend is None:
            raise RuntimeError("Call fit_frontend or fit before encode.")
        scaled = as_quantum(self.scaler.transform(frame))
        self.frontend.eval()
        with torch.no_grad():
            return to_numpy(self.frontend.angles(scaled))

    def fit_frontend(
        self,
        features_train: pd.DataFrame,
        labels_train,
        features_valid: pd.DataFrame,
        labels_valid,
        seed: int,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Train only the classical network that produces the four angles."""
        set_seed(seed)
        self.frontend = AngleMLP(self.config.hidden_size).to(device=TORCH_DEVICE, dtype=torch.float64)
        self.scaler.fit(features_train)
        train_x = as_quantum(self.scaler.transform(features_train))
        train_y = as_quantum(np.asarray(labels_train, dtype=np.float64))
        return self._fit_frontend(train_x, train_y, features_valid, np.asarray(labels_valid), verbose)

    def fit(
        self,
        features_train: pd.DataFrame,
        labels_train,
        features_valid: pd.DataFrame,
        labels_valid,
        seed: int,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        phase1 = self.fit_frontend(features_train, labels_train, features_valid, labels_valid, seed, verbose)
        if self.frontend is None:
            raise RuntimeError("Frontend training did not produce a network.")
        for parameter in self.frontend.parameters():
            parameter.requires_grad_(False)
        self.frontend.eval()
        if verbose:
            print("phase 2: training the VQC on frozen angles")
        phase2 = self.vqc.fit(
            self.encode(features_train),
            np.asarray(labels_train),
            self.encode(features_valid),
            np.asarray(labels_valid),
            seed,
            verbose=verbose,
            optimizer_fn=self.vqc_optimizer_fn,
        )
        return phase1, phase2

    def predict(self, frame: pd.DataFrame, chunk: int | None = None) -> np.ndarray:
        return self.vqc.predict(self.encode(frame), chunk=chunk)

    def predict_head(self, frame: pd.DataFrame) -> np.ndarray:
        """Fraud probability from the frontend's single classification head."""
        return self._classifier_probabilities(frame)

    def _fit_frontend(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        features_valid: pd.DataFrame,
        labels_valid: np.ndarray,
        verbose: bool,
    ) -> pd.DataFrame:
        self.frontend.train()
        for parameter in self.frontend.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            self.frontend.parameters(),
            lr=self.config.mlp_learning_rate,
            weight_decay=self.config.mlp_weight_decay,
        )
        history: list[dict] = []
        best_ap = -1.0
        best_state = None
        if verbose:
            print("phase 1: training the classical frontend")
        for epoch in range(1, self.config.mlp_epochs + 1):
            order = torch.randperm(len(features))
            totals = {"loss": [], "bce": [], "separation": [], "spread": []}
            for start in range(0, len(features), self.config.mlp_batch_size):
                idx = order[start : start + self.config.mlp_batch_size]
                batch_x = features[idx]
                batch_y = labels[idx]
                optimizer.zero_grad()
                with torch.no_grad():
                    prototypes = class_prototypes(
                        self.frontend.angles(features), labels, self.config.vqc.connectivity
                    )
                angles, logits = self.frontend(batch_x)
                bce = F.binary_cross_entropy_with_logits(logits, batch_y)
                separated = separation_loss(
                    angles,
                    batch_y,
                    prototypes,
                    self.config.separation_margin,
                    self.config.vqc.connectivity,
                )
                spread = spread_loss(angles, self.config.spread_target_std)
                loss = bce + self.config.separation_weight * separated + self.config.spread_weight * spread
                loss.backward()
                optimizer.step()
                totals["loss"].append(loss.item())
                totals["bce"].append(bce.item())
                totals["separation"].append(separated.item())
                totals["spread"].append(spread.item())
            probabilities = self._classifier_probabilities(features_valid)
            val_ap = float(average_precision_score(labels_valid, probabilities))
            row = {
                "epoch": epoch,
                "train loss": float(np.mean(totals["loss"])),
                "bce": float(np.mean(totals["bce"])),
                "separation": float(np.mean(totals["separation"])),
                "spread": float(np.mean(totals["spread"])),
                "validation AUPRC": val_ap,
            }
            history.append(row)
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = {name: value.detach().cpu().clone() for name, value in self.frontend.state_dict().items()}
            if verbose:
                print(
                    f"epoch {epoch:2d}   train loss {row['train loss']:.4f}"
                    f"   bce {row['bce']:.4f}   separation {row['separation']:.4f}"
                    f"   spread {row['spread']:.4f}   validation AUPRC {val_ap:.3f}"
                )
        self.frontend.load_state_dict(best_state)
        self.frontend.to(device=TORCH_DEVICE, dtype=torch.float64)
        if verbose:
            print(f"phase 1 kept the frontend from validation AUPRC {best_ap:.3f}")
        return pd.DataFrame(history)

    def _classifier_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        scaled = as_quantum(self.scaler.transform(frame))
        was_training = self.frontend.training
        self.frontend.eval()
        with torch.no_grad():
            _, logits = self.frontend(scaled)
            probabilities = to_numpy(torch.sigmoid(logits))
        self.frontend.train(was_training)
        return probabilities


class ClassificationHead:
    """Same learned angles as the two-phase model, scored by its one classical head.

    Phase 2 is skipped, so no circuit is trained or evaluated.
    """

    def __init__(self, config: TwoPhaseConfig | None = None):
        self.config = config or TwoPhaseConfig()
        self.encoder = TwoPhaseVQC(self.config)

    def fit(
        self,
        features_train: pd.DataFrame,
        labels_train,
        features_valid: pd.DataFrame,
        labels_valid,
        seed: int,
        verbose: bool = True,
    ) -> pd.DataFrame:
        return self.encoder.fit_frontend(
            features_train, labels_train, features_valid, labels_valid, seed, verbose
        )

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        return self.encoder.encode(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.encoder.predict_head(frame)


if __name__ == "__main__":
    from data.creditcard import load_transactions, prepare_splits

    transactions = load_transactions()
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_splits(transactions)
    model = TwoPhaseVQC()
    phase1, phase2 = model.fit(x_train, y_train, x_valid, y_valid, seed=42)
    test_scores = model.predict(x_test)
    print(f"test AUPRC {average_precision_score(y_test, test_scores):.3f}")
    print(phase1.tail(1).to_string(index=False))
    print(phase2.tail(1).to_string(index=False))
