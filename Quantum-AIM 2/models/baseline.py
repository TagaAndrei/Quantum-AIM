"""Baseline variational classifier: PCA angles, ZZ feature map, single ⟨Z⟩ readout."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.creditcard import RAW_COLUMNS
from models.circuit import EfficientSU2Ansatz, ZZFeatureMap
from models.config import VQCConfig
from models.runtime import (
    TORCH_DEVICE,
    as_quantum,
    quantum_runtime,
    score_circuit,
    set_seed,
    to_numpy,
)


class AngleEncoder:
    """PCA angles in ``[0, π]``. ``Time`` and ``Amount`` are standardised first."""

    def __init__(self, n_components: int, seed: int):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame: pd.DataFrame) -> np.ndarray:
        out = frame.copy()
        out[list(RAW_COLUMNS)] = self.scaler.transform(frame[list(RAW_COLUMNS)])
        return out.to_numpy()

    def fit(self, frame: pd.DataFrame) -> AngleEncoder:
        self.scaler.fit(frame[list(RAW_COLUMNS)])
        pca_output = self.pca.fit_transform(self._scaled(frame))
        self.angles.fit(pca_output)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        projected = self.pca.transform(self._scaled(frame))
        return np.clip(self.angles.transform(projected), 0, np.pi)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class BaselineVQC:
    def __init__(self, config: VQCConfig | None = None):
        self.config = config or VQCConfig()
        self.feature_map = ZZFeatureMap(
            self.config.n_qubits, self.config.feature_map_reps, self.config.connectivity
        )
        self.ansatz = EfficientSU2Ansatz(self.config.n_qubits, self.config.ansatz_reps)
        self.device = qml.device("default.qubit", wires=self.config.n_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch", diff_method="backprop")
        self.weights: torch.nn.Parameter | None = None

    def circuit(self, x, weights):
        self.feature_map(x)
        self.ansatz(weights)
        return qml.expval(qml.PauliZ(0))

    @staticmethod
    def expectation_to_probability(expectation):
        return (1 - expectation) / 2

    def fraud_probability(self, x, weights):
        return self.expectation_to_probability(self.qnode(x, weights))

    def initialize(self, seed: int) -> torch.nn.Parameter:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = 2 * np.pi * torch.rand(
            *self.config.weight_shape, generator=generator, dtype=torch.float64, device="cpu"
        )
        return torch.nn.Parameter(values.to(torch.get_default_device()))

    def predict(self, features, weights=None, chunk: int | None = None) -> np.ndarray:
        weights = self._require_weights(weights)
        if chunk is None:
            chunk = self.config.predict_chunk
        features_t = as_quantum(features)
        weights_t = weights.to(TORCH_DEVICE)
        with quantum_runtime(), torch.no_grad():
            scores = torch.cat(
                [
                    self.fraud_probability(features_t[start : start + chunk], weights_t).reshape(-1)
                    for start in range(0, len(features_t), chunk)
                ]
            )
        return to_numpy(scores)

    def fit(
        self,
        features_train,
        labels_train,
        features_valid,
        labels_valid,
        seed: int,
        epochs: int | None = None,
        verbose: bool = True,
        optimizer_fn: Callable | None = None,
        initial_weights: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        if epochs is None:
            epochs = self.config.epochs
        set_seed(seed)
        with quantum_runtime():
            if initial_weights is None:
                weights = self.initialize(seed)
            else:
                weights = torch.nn.Parameter(initial_weights.detach().clone().to(dtype=torch.float64))
            if weights.device != TORCH_DEVICE:
                weights = torch.nn.Parameter(weights.detach().to(TORCH_DEVICE))
            if optimizer_fn is None:
                optimizer = torch.optim.Adam([weights], lr=self.config.learning_rate)
            else:
                optimizer = optimizer_fn(weights)
            loss_fn = torch.nn.BCELoss()
            features_t = as_quantum(features_train)
            labels_t = as_quantum(labels_train)
            history: list[dict] = []
            best_ap = -1.0
            best_weights = None
            for epoch in range(1, epochs + 1):
                order = torch.randperm(len(features_t))
                losses = []
                for start in range(0, len(features_t), self.config.batch_size):
                    idx = order[start : start + self.config.batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(self.fraud_probability(features_t[idx], weights), labels_t[idx])
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                val_ap = average_precision_score(labels_valid, self.predict(features_valid, weights))
                history.append(
                    {"epoch": epoch, "train loss": float(np.mean(losses)), "validation AUPRC": val_ap}
                )
                if val_ap > best_ap:
                    best_ap = val_ap
                    best_weights = weights.detach().clone()
                if verbose:
                    print(
                        f"epoch {epoch:2d}   train loss {np.mean(losses):.4f}"
                        f"   validation AUPRC {val_ap:.3f}"
                    )
        self.weights = torch.nn.Parameter(best_weights.detach().to("cpu"))
        return pd.DataFrame(history)

    def noisy_qnode(self, probability: float):
        noisy_device = qml.noise.insert(
            qml.device("default.mixed", wires=self.config.n_qubits),
            qml.DepolarizingChannel,
            probability,
            position="all",
        )
        qnode = qml.QNode(self.circuit, noisy_device, interface="torch", diff_method="backprop")
        return qml.transforms.decompose(qnode, gate_set=set(self.config.native_gates))

    def predict_noisy(self, features, probability: float, weights=None, chunk: int | None = None) -> np.ndarray:
        weights = self._require_weights(weights)
        if chunk is None:
            chunk = self.config.noisy_chunk
        return score_circuit(
            self.noisy_qnode(probability),
            features,
            weights,
            self.expectation_to_probability,
            chunk,
        )

    def noise_locations(self, features=None, weights=None) -> int:
        if features is None:
            features = torch.zeros(self.config.n_qubits, dtype=torch.float64)
        if weights is None:
            weights = torch.zeros(self.config.weight_shape, dtype=torch.float64)
        qnode = qml.transforms.decompose(
            qml.QNode(self.circuit, qml.device("default.qubit", wires=self.config.n_qubits)),
            gate_set=set(self.config.native_gates),
        )
        resources = qml.specs(qnode)(features, weights)["resources"]
        return sum(size * count for size, count in resources.gate_sizes.items())

    def specs(self, features=None, weights=None) -> dict:
        if features is None:
            features = torch.zeros(self.config.n_qubits, dtype=torch.float64)
        if weights is None:
            weights = torch.zeros(self.config.weight_shape, dtype=torch.float64)
        return qml.specs(self.qnode)(features, weights)

    def _require_weights(self, weights) -> torch.Tensor:
        if weights is None:
            weights = self.weights
        if weights is None:
            raise RuntimeError("No weights yet. Call initialize or fit, or pass weights explicitly.")
        return weights


if __name__ == "__main__":
    model = BaselineVQC()
    print(f"parameters: {model.config.n_parameters}   weight shape: {model.config.weight_shape}")
    print(f"noise locations: {model.noise_locations()}")
    resources = model.specs()["resources"]
    print(f"gates: {resources.num_gates}   depth: {resources.depth}   gate types: {dict(resources.gate_types)}")
