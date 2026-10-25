"""Torch device and tensor helpers shared by every circuit."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def resolve_torch_device() -> torch.device:
    """Device that holds the circuit state."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


TORCH_DEVICE = resolve_torch_device()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_quantum(array, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=TORCH_DEVICE)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@contextmanager
def quantum_runtime():
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_device(TORCH_DEVICE)
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def score_circuit(qnode, features, weights, readout, chunk: int) -> np.ndarray:
    features_t = as_quantum(features)
    weights_t = weights.to(TORCH_DEVICE)
    with quantum_runtime(), torch.no_grad():
        pieces = []
        for start in range(0, len(features_t), chunk):
            values = qnode(features_t[start : start + chunk], weights_t)
            if isinstance(values, (tuple, list)):
                values = torch.stack(list(values))
            pieces.append(readout(values).reshape(-1))
        return to_numpy(torch.cat(pieces))
