"""Light-cone VQC, following Suzuki, Sakuma, and Kawaguchi, arXiv:2403.18733.

Each qubit is one light-cone. Its feature is the projected quantum kernel on that
qubit's reduced state, Tr[ρ_i^{(l)} ρ_j^{(l)}] (Eq. 7). Ansatz weights are
trained to maximize kernel-target alignment. Cone weights λ are then solved
from the quadratic program in Eq. (11)–(13) and normalised to sum to one.
Scores are the decision values of a precomputed-kernel SVM on that kernel, so
the Task 2 noise evaluation can call ``predict`` and ``predict_noisy``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from models.circuit import EfficientSU2Ansatz, ZZFeatureMap
from models.config import VQCConfig
from models.runtime import TORCH_DEVICE, as_quantum, quantum_runtime, set_seed, to_numpy


def stack_pauli(values, n_qubits: int) -> torch.Tensor:
    """Pack per-qubit X, Y, Z expectations into ``(batch, qubits, 3)``."""
    if isinstance(values, torch.Tensor):
        values = (values,)
    stacked = torch.stack([value.reshape(-1) for value in values], dim=-1)
    return stacked.reshape(-1, n_qubits, 3)


def local_kernels(bloch: torch.Tensor) -> torch.Tensor:
    """Fidelity of one-qubit reduced states, shape ``(qubits, batch, batch)``."""
    vectors = bloch.permute(1, 0, 2)
    dots = torch.einsum("qia,qja->qij", vectors, vectors)
    return (1 + dots) / 2


def kernel_alignment(kernel: torch.Tensor, labels_pm: torch.Tensor) -> torch.Tensor:
    """Standard kernel-target alignment ⟨K, yyᵀ⟩ / (‖K‖_F ‖yyᵀ‖_F)."""
    numerator = labels_pm @ kernel @ labels_pm
    frobenius = torch.sqrt(kernel.square().sum().clamp_min(1e-12))
    label_norm = labels_pm.square().sum().clamp_min(1e-12)
    return numerator / (frobenius * label_norm)


def lightcone_weights(local: np.ndarray, labels_pm: np.ndarray) -> np.ndarray:
    """Nonnegative cone weights from Eq. (11), normalised by the L1 norm."""
    n_cones = local.shape[0]
    gram = local.reshape(n_cones, -1)
    pairwise = gram @ gram.T
    alignment = gram @ np.outer(labels_pm, labels_pm).reshape(-1)
    start = np.full(n_cones, 1 / n_cones)

    def objective(weights):
        return float(weights @ pairwise @ weights - 2 * alignment @ weights)

    def jacobian(weights):
        return 2 * pairwise @ weights - 2 * alignment

    result = minimize(
        objective,
        start,
        jac=jacobian,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * n_cones,
    )
    found = np.clip(result.x, 0, None)
    total = float(found.sum())
    if total <= 1e-12:
        return start
    return found / total


def combine_kernels(local: np.ndarray, weights: np.ndarray) -> np.ndarray:
    kernel = np.tensordot(weights, local, axes=(0, 0))
    kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(kernel, kernel.diagonal() + 1e-8)
    return kernel


def cross_kernel(query: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Kernel between query and reference Bloch vectors, shape ``(query, reference)``."""
    dots = np.einsum("mqa,nqa->qmn", query, reference)
    local = (1 + dots) / 2
    return np.tensordot(weights, local, axes=(0, 0))


class LightConeVQC:
    def __init__(self, config: VQCConfig | None = None):
        self.config = config or VQCConfig()
        self.feature_map = ZZFeatureMap(
            self.config.n_qubits, self.config.feature_map_reps, self.config.connectivity
        )
        self.ansatz = EfficientSU2Ansatz(self.config.n_qubits, self.config.ansatz_reps)
        self.device = qml.device("default.qubit", wires=self.config.n_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch", diff_method="backprop")
        self.weights: torch.nn.Parameter | None = None
        self.lam: np.ndarray | None = None
        self.train_bloch: np.ndarray | None = None
        self.svm: SVC | None = None

    def circuit(self, x, weights):
        self.feature_map(x)
        self.ansatz(weights)
        expectations = []
        for qubit in range(self.config.n_qubits):
            expectations.append(qml.expval(qml.PauliX(qubit)))
            expectations.append(qml.expval(qml.PauliY(qubit)))
            expectations.append(qml.expval(qml.PauliZ(qubit)))
        return expectations

    def initialize(self, seed: int) -> torch.nn.Parameter:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = 2 * np.pi * torch.rand(
            *self.config.weight_shape, generator=generator, dtype=torch.float64, device="cpu"
        )
        return torch.nn.Parameter(values.to(torch.get_default_device()))

    def _bloch(self, qnode, features, weights, chunk: int, grad: bool = False) -> torch.Tensor:
        features_t = as_quantum(features)
        weights_t = weights.to(TORCH_DEVICE)
        pieces = []
        context = torch.enable_grad() if grad else torch.no_grad()
        with quantum_runtime(), context:
            for start in range(0, len(features_t), chunk):
                values = qnode(features_t[start : start + chunk], weights_t)
                pieces.append(stack_pauli(values, self.config.n_qubits))
        return torch.cat(pieces, dim=0)

    def _fit_svm(self, train_bloch: np.ndarray, labels, weights: np.ndarray) -> SVC:
        vectors = np.transpose(train_bloch, (1, 0, 2))
        local = (1 + np.einsum("qia,qja->qij", vectors, vectors)) / 2
        kernel = combine_kernels(local, weights)
        svm = SVC(kernel="precomputed")
        svm.fit(kernel, np.asarray(labels).astype(int))
        return svm

    def _scores(self, query: np.ndarray, train_bloch: np.ndarray, weights: np.ndarray, svm: SVC) -> np.ndarray:
        return np.asarray(svm.decision_function(cross_kernel(query, train_bloch, weights)), dtype=np.float64)

    def predict(self, features, weights=None, chunk: int | None = None) -> np.ndarray:
        weights = self._require_weights(weights)
        if chunk is None:
            chunk = self.config.predict_chunk
        if self.svm is None or self.train_bloch is None or self.lam is None:
            raise RuntimeError("Call fit before predict.")
        bloch = self._bloch(self.qnode, features, weights, chunk)
        return self._scores(to_numpy(bloch), self.train_bloch, self.lam, self.svm)

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
    ) -> pd.DataFrame:
        if epochs is None:
            epochs = self.config.epochs
        set_seed(seed)
        labels_np = np.asarray(labels_train).astype(int)
        labels_pm_np = np.where(labels_np > 0, 1.0, -1.0)
        with quantum_runtime():
            weights = self.initialize(seed)
            if weights.device != TORCH_DEVICE:
                weights = torch.nn.Parameter(weights.detach().to(TORCH_DEVICE))
            if optimizer_fn is None:
                optimizer = torch.optim.Adam([weights], lr=self.config.learning_rate)
            else:
                optimizer = optimizer_fn(weights)
            features_t = as_quantum(features_train)
            labels_pm = as_quantum(labels_pm_np)
            lam = np.full(self.config.n_qubits, 1 / self.config.n_qubits)
            history: list[dict] = []
            best_ap = -1.0
            best_weights = None
            best_lam = None
            best_bloch = None
            best_svm = None
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                bloch = self._bloch(self.qnode, features_t, weights, len(features_t), grad=True)
                local = local_kernels(bloch)
                lam_t = torch.as_tensor(lam, dtype=local.dtype, device=local.device)
                alignment = kernel_alignment(torch.einsum("q,qij->ij", lam_t, local), labels_pm)
                (-alignment).backward()
                optimizer.step()

                measured = self._bloch(self.qnode, features_t, weights, len(features_t))
                local_np = to_numpy(local_kernels(measured))
                lam = lightcone_weights(local_np, labels_pm_np)
                train_bloch = to_numpy(measured)
                svm = self._fit_svm(train_bloch, labels_np, lam)
                valid_bloch = to_numpy(self._bloch(self.qnode, features_valid, weights, self.config.predict_chunk))
                val_ap = float(average_precision_score(labels_valid, self._scores(valid_bloch, train_bloch, lam, svm)))
                history.append(
                    {
                        "epoch": epoch,
                        "train loss": float((-alignment).detach().cpu()),
                        "validation AUPRC": val_ap,
                    }
                )
                if val_ap > best_ap:
                    best_ap = val_ap
                    best_weights = weights.detach().clone()
                    best_lam = lam.copy()
                    best_bloch = train_bloch.copy()
                    best_svm = svm
                if verbose:
                    print(
                        f"epoch {epoch:2d}   KTA {float(alignment.detach().cpu()):.4f}"
                        f"   validation AUPRC {val_ap:.3f}"
                    )
        self.weights = torch.nn.Parameter(best_weights.detach().to("cpu"))
        self.lam = best_lam
        self.train_bloch = best_bloch
        self.svm = best_svm
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
        if self.svm is None or self.train_bloch is None or self.lam is None:
            raise RuntimeError("Call fit before predict_noisy.")
        bloch = self._bloch(self.noisy_qnode(probability), features, weights, chunk)
        return self._scores(to_numpy(bloch), self.train_bloch, self.lam, self.svm)

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

    def _require_weights(self, weights) -> torch.Tensor:
        if weights is None:
            weights = self.weights
        if weights is None:
            raise RuntimeError("No weights yet. Call initialize or fit, or pass weights explicitly.")
        return weights
