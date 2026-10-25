"""Build the optimizer that steps variational circuit weights.

Adam and AdamW are the PyTorch implementations. Muon is the single-device
Newton–Schulz update. Frontend parameters and light-cone weights λ are trained
elsewhere and are not passed to these factories.
"""

from __future__ import annotations

import torch

from optimizers.muon import SingleDeviceMuon

OPTIMIZERS = ("adam", "adamw", "muon")

ADAMW_WEIGHT_DECAY = 1e-2
MUON_LEARNING_RATE = 0.02
MUON_WEIGHT_DECAY = 0.0
MUON_MOMENTUM = 0.95


def muon(weights: torch.nn.Parameter) -> SingleDeviceMuon:
    """Upstream single-device defaults. The VQC learning rate is not used."""
    return SingleDeviceMuon(
        [weights],
        lr=MUON_LEARNING_RATE,
        weight_decay=MUON_WEIGHT_DECAY,
        momentum=MUON_MOMENTUM,
    )


def make_vqc_optimizer(kind: str, learning_rate: float):
    """Return a callable ``weights -> optimizer`` for ansatz parameters."""

    def adam(weights: torch.nn.Parameter):
        return torch.optim.Adam([weights], lr=learning_rate)

    def adamw(weights: torch.nn.Parameter):
        return torch.optim.AdamW([weights], lr=learning_rate, weight_decay=ADAMW_WEIGHT_DECAY)

    if kind == "adam":
        return adam
    if kind == "adamw":
        return adamw
    if kind == "muon":
        return muon
    known = ", ".join(OPTIMIZERS)
    raise ValueError(f"Unknown optimizer {kind}. Known optimizers: {known}")
