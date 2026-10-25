"""Write and reload one ``model.pth`` for each trained run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from models.baseline import AngleEncoder, BaselineVQC
from models.lightcone import LightConeVQC
from models.runtime import TORCH_DEVICE
from models.two_phase import AngleMLP, ClassificationHead, FeatureScaler, TwoPhaseVQC
from paths import TRAINED_MODELS


def model_path(name: str, seed: int) -> Path:
    """``models/trained/<experiment>/seed<seed>/model.pth``."""
    return TRAINED_MODELS / name / f"seed{seed}" / "model.pth"


def save_model(name: str, seed: int, state: dict) -> Path:
    path = model_path(name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"saved {path}", flush=True)
    return path


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_baseline(name: str, seed: int, encoder: AngleEncoder, model: BaselineVQC) -> Path:
    if model.weights is None:
        raise RuntimeError(f"{name} seed {seed} has no circuit weights to save.")
    return save_model(
        name,
        seed,
        {
            "kind": "baseline",
            "name": name,
            "seed": seed,
            "config": model.config,
            "weights": model.weights.detach().cpu(),
            "encoder": {"scaler": encoder.scaler, "pca": encoder.pca, "angles": encoder.angles},
        },
    )


def save_two_phase(
    name: str,
    seed: int,
    model: TwoPhaseVQC,
    genetic=None,
) -> Path:
    if model.frontend is None or model.vqc.weights is None:
        raise RuntimeError(f"{name} seed {seed} is missing a frontend or circuit weights.")
    state = {
        "kind": "two_phase",
        "name": name,
        "seed": seed,
        "config": model.config,
        "weights": model.vqc.weights.detach().cpu(),
        "frontend": _cpu_state(model.frontend),
        "scaler": model.scaler.scaler,
    }
    if genetic is not None:
        state["genetic"] = genetic
    return save_model(name, seed, state)


def save_head(name: str, seed: int, model: ClassificationHead) -> Path:
    frontend = model.encoder
    if frontend.frontend is None:
        raise RuntimeError(f"{name} seed {seed} has no classification head to save.")
    return save_model(
        name,
        seed,
        {
            "kind": "head",
            "name": name,
            "seed": seed,
            "config": frontend.config,
            "frontend": _cpu_state(frontend.frontend),
            "scaler": frontend.scaler.scaler,
        },
    )


def save_lightcone(name: str, seed: int, frontend: TwoPhaseVQC, circuit: LightConeVQC) -> Path:
    if frontend.frontend is None or circuit.weights is None:
        raise RuntimeError(f"{name} seed {seed} is missing a frontend or circuit weights.")
    return save_model(
        name,
        seed,
        {
            "kind": "lightcone",
            "name": name,
            "seed": seed,
            "config": frontend.config,
            "weights": circuit.weights.detach().cpu(),
            "lam": circuit.lam,
            "train_bloch": circuit.train_bloch,
            "svm": circuit.svm,
            "frontend": _cpu_state(frontend.frontend),
            "scaler": frontend.scaler.scaler,
        },
    )


@dataclass
class LoadedCheckpoint:
    """A trained model restored from ``model.pth`` for noise and cost sweeps."""

    path: Path
    kind: str
    name: str
    seed: int
    circuit: object | None
    weights: torch.Tensor | None
    encode: object
    config: object
    genetic: object | None = None
    head: ClassificationHead | None = None


def discover_checkpoints(root: Path | None = None) -> list[Path]:
    """Every ``model.pth`` under ``models/trained``, sorted by path."""
    base = root or TRAINED_MODELS
    if not base.exists():
        return []
    return sorted(base.rglob("model.pth"))


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    """Rebuild the encoder and circuit from a saved ``model.pth``."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    kind = str(state["kind"])
    name = str(state.get("name", path.parent.parent.name))
    seed = int(state.get("seed", 0))
    genetic = state.get("genetic")

    if kind == "baseline":
        config = state["config"]
        encoder = AngleEncoder(n_components=config.n_qubits, seed=seed)
        encoder.scaler = state["encoder"]["scaler"]
        encoder.pca = state["encoder"]["pca"]
        encoder.angles = state["encoder"]["angles"]
        circuit = BaselineVQC(config)
        weights = torch.nn.Parameter(state["weights"].detach().cpu().to(dtype=torch.float64))
        circuit.weights = weights
        return LoadedCheckpoint(
            path=path,
            kind=kind,
            name=name,
            seed=seed,
            circuit=circuit,
            weights=weights,
            encode=encoder.transform,
            config=config,
            genetic=genetic,
        )

    if kind == "two_phase":
        config = state["config"]
        model = TwoPhaseVQC(config)
        model.scaler = FeatureScaler()
        model.scaler.scaler = state["scaler"]
        model.frontend = AngleMLP(config.hidden_size).to(device=TORCH_DEVICE, dtype=torch.float64)
        model.frontend.load_state_dict(state["frontend"])
        model.frontend.eval()
        for parameter in model.frontend.parameters():
            parameter.requires_grad_(False)
        weights = torch.nn.Parameter(state["weights"].detach().cpu().to(dtype=torch.float64))
        model.vqc.weights = weights
        return LoadedCheckpoint(
            path=path,
            kind=kind,
            name=name,
            seed=seed,
            circuit=model.vqc,
            weights=weights,
            encode=model.encode,
            config=config,
            genetic=genetic,
        )

    if kind == "lightcone":
        config = state["config"]
        frontend = TwoPhaseVQC(config)
        frontend.scaler = FeatureScaler()
        frontend.scaler.scaler = state["scaler"]
        frontend.frontend = AngleMLP(config.hidden_size).to(device=TORCH_DEVICE, dtype=torch.float64)
        frontend.frontend.load_state_dict(state["frontend"])
        frontend.frontend.eval()
        for parameter in frontend.frontend.parameters():
            parameter.requires_grad_(False)
        circuit = LightConeVQC(config.vqc)
        weights = torch.nn.Parameter(state["weights"].detach().cpu().to(dtype=torch.float64))
        circuit.weights = weights
        circuit.lam = state["lam"]
        circuit.train_bloch = state["train_bloch"]
        circuit.svm = state["svm"]
        return LoadedCheckpoint(
            path=path,
            kind=kind,
            name=name,
            seed=seed,
            circuit=circuit,
            weights=weights,
            encode=frontend.encode,
            config=config,
            genetic=genetic,
        )

    if kind == "head":
        config = state["config"]
        model = ClassificationHead(config)
        model.encoder.scaler = FeatureScaler()
        model.encoder.scaler.scaler = state["scaler"]
        model.encoder.frontend = AngleMLP(config.hidden_size).to(
            device=TORCH_DEVICE, dtype=torch.float64
        )
        model.encoder.frontend.load_state_dict(state["frontend"])
        model.encoder.frontend.eval()
        for parameter in model.encoder.frontend.parameters():
            parameter.requires_grad_(False)
        return LoadedCheckpoint(
            path=path,
            kind=kind,
            name=name,
            seed=seed,
            circuit=None,
            weights=None,
            encode=model.encode,
            config=config,
            genetic=None,
            head=model,
        )

    raise ValueError(f"Unknown checkpoint kind {kind} in {path}")
