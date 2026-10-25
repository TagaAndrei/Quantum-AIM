"""Gates shared by the baseline readout and the light-cone kernel."""

from __future__ import annotations

import numpy as np
import pennylane as qml

CONNECTIVITIES = ("all", "chain")


def qubit_pairs(n_qubits: int, connectivity: str = "all") -> list[tuple[int, int]]:
    """Qubit pairs that share a two-qubit block.

    ``all`` connects every pair. ``chain`` connects each qubit only to the next one.
    """
    if connectivity == "all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if connectivity == "chain":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    known = ", ".join(CONNECTIVITIES)
    raise ValueError(f"Unknown connectivity {connectivity}. Known connectivities: {known}")


class ZZFeatureMap:
    def __init__(self, n_qubits: int, reps: int, connectivity: str = "all"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.connectivity = connectivity
        self.pairs = qubit_pairs(n_qubits, connectivity)

    def __call__(self, x) -> None:
        """Apply the map. ``x`` has shape ``(..., n_qubits)`` and entries in ``[0, π]``."""
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[..., i], wires=i)
            for i, j in self.pairs:
                phi = 2 * (np.pi - x[..., i]) * (np.pi - x[..., j])
                qml.CNOT(wires=[i, j])
                qml.RZ(phi, wires=j)
                qml.CNOT(wires=[i, j])


class EfficientSU2Ansatz:
    def __init__(self, n_qubits: int, reps: int):
        self.n_qubits = n_qubits
        self.reps = reps

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.reps + 1, self.n_qubits, 2)

    def __call__(self, weights) -> None:
        for layer in range(self.reps + 1):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            if layer < self.reps:
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
