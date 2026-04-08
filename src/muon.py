"""Muon optimizer — SGD with Nesterov momentum + Newton-Schulz orthogonalization.

Newton-Schulz orthogonalization maps a gradient matrix G to an approximately
orthogonal update with spectral norm ~1, eliminating the bimodal gradient
distribution that arises from the magnitude difference between weight-matrix
gradients (~10⁻¹) and LayerNorm gradients (~10⁻²).

Reference: Keller Jordan's Muon implementation.

Public API
----------
    zeropower_via_newtonschulz5(G, steps=5) -> Tensor
    Muon(params, lr, momentum=0.95, nesterov=True)
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Return an approximately orthogonal matrix with spectral norm ~1.

    Applies a 5th-order Newton-Schulz iteration to orthogonalize *G*.
    Tall matrices (rows > cols) are transposed before the iteration and
    transposed back afterward so the inner iteration always operates on a
    wide matrix.  Computation is performed in float32 regardless of the
    input dtype; the result is cast back to the original dtype before return.

    Parameters
    ----------
    G : Tensor
        Input matrix.  Must have ``ndim >= 2``.
    steps : int
        Number of Newton-Schulz iterations (default 5).

    Returns
    -------
    Tensor
        Same shape and dtype as *G*, with singular values approximately 1.
    """
    assert G.ndim >= 2, f"zeropower_via_newtonschulz5 requires ndim >= 2, got {G.ndim}"
    orig_dtype = G.dtype

    X = G.float()
    # Normalize to unit Frobenius norm so the iteration starts near convergence.
    X = X / (X.norm() + 1e-7)

    # Newton-Schulz coefficients for the degree-5 polynomial approximation.
    a, b, c = 3.4445, -4.7750, 2.0315

    # Transpose tall matrices so the inner product A = X @ X.T is small.
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


class Muon(Optimizer):
    """SGD with Nesterov momentum and Newton-Schulz orthogonalization.

    For parameters with ``ndim >= 2`` the gradient update is orthogonalized via
    ``zeropower_via_newtonschulz5`` before being applied, normalizing the
    effective step size across all matrix parameters regardless of their raw
    gradient magnitude.

    1-D parameters (biases) bypass orthogonalization and receive a standard
    SGD-with-momentum update.

    Parameters
    ----------
    params : iterable
        Iterable of ``torch.Tensor`` parameters or param-group dicts.
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (default 0.95).
    nesterov : bool
        Use Nesterov momentum (default ``True``).
    ns_steps : int
        Number of Newton-Schulz iterations (default 5).
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf: torch.Tensor = state["momentum_buffer"]

                # Standard SGD momentum update: buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    # Nesterov look-ahead: use grad + momentum * buf
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # Orthogonalize 2-D (and higher) params only.
                if update.ndim >= 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                p.data.add_(update, alpha=-lr)

        return loss
