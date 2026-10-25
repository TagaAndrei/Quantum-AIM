"""Circuit width, depth, and training settings shared by the VQC models."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.runtime import TORCH_DEVICE

NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}


@dataclass(frozen=True)
class VQCConfig:
    n_qubits: int = 4
    feature_map_reps: int = 2
    ansatz_reps: int = 3
    connectivity: str = "all"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.05
    predict_chunk: int = 20000
    noisy_chunk: int = 8192 if TORCH_DEVICE.type == "cuda" else 2000
    noise_levels: tuple[float, ...] = (0.001, 0.01, 0.05)
    native_gates: frozenset[str] = field(default_factory=lambda: frozenset(NATIVE_GATES))

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.ansatz_reps + 1, self.n_qubits, 2)

    @property
    def n_parameters(self) -> int:
        layers, qubits, angles = self.weight_shape
        return layers * qubits * angles
