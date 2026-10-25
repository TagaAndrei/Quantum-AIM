"""Single-device Muon from https://github.com/KellerJordan/Muon/blob/master/muon.py.

The repository's ``Muon`` class is the distributed implementation and calls
``torch.distributed`` on every step. This file keeps the non-distributed
``SingleDeviceMuon`` path, including the batched Newton–Schulz update that
accepts parameters with two or more dimensions.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Coefficients maximize the slope at zero. The iteration therefore does not
    produce UV^T but rather something like US'V^T where S' is diagonal with
    S_ii' ~ Uniform(0.5, 1.5).
    """
    assert G.ndim >= 2  # batched Muon: leading dimensions are a batch of matrices
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters: collapse the last 3 dimensions
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon variant for usage in non-distributed settings.

    Defaults match the GitHub class: lr 0.02 (spectral-norm units), momentum
    0.95, and no weight decay. Nesterov and five Newton–Schulz steps are fixed
    inside ``muon_update``, as in the upstream step.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
