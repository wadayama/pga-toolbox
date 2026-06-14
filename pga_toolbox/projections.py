"""Closed-form Euclidean projections for constrained PGA.

These projectors are designed to be plugged into `pga_ascent`,
`pga_ascent_armijo`, and the descent variants. They are applied inside
`torch.no_grad()` blocks during projected gradient methods and return
new tensors (do not modify input in place); the caller is responsible
for using `tensor.copy_()` or similar when in-place semantics are
desired, although the PGA drivers in this package handle that copying
automatically.

Currently provided:
  - `project_frobenius_ball(A, P)`: project a single matrix onto
    {X : ||X||_F^2 <= P}.
  - `project_total_power(params, P)`: project a list of matrices onto
    {{A_m} : sum_m ||A_m||_F^2 <= P}, by uniform rescaling.
"""

from __future__ import annotations

import torch


def project_frobenius_ball(A: torch.Tensor, P: float) -> torch.Tensor:
    """Project ``A`` onto the Frobenius ball {X : ||X||_F^2 <= P}.

    Formula:
        A_proj = A * min(1, sqrt(P) / ||A||_F).

    The projection is exact (in the Euclidean sense) for the
    Frobenius-ball constraint and preserves the direction (uniform
    rescaling).

    Args:
        A: Complex or real tensor (any shape; norm is taken as Frobenius).
        P: Positive power budget.

    Returns:
        New tensor; same shape and dtype as A.
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    norm = torch.linalg.norm(A)
    sqrt_P = torch.sqrt(torch.tensor(P, dtype=norm.dtype, device=norm.device))
    scale = torch.where(norm <= sqrt_P, torch.ones_like(norm), sqrt_P / norm)
    return A * scale


def project_total_power(
    params: list[torch.Tensor],
    P: float,
) -> list[torch.Tensor]:
    """Project a list of matrices onto sum_m ||A_m||_F^2 <= P.

    Formula:
        A_m_proj = A_m * min(1, sqrt(P) / sqrt(sum_m ||A_m||_F^2)).

    All matrices are rescaled by the same factor, preserving the
    relative magnitudes (which is the Euclidean projection of the
    stacked vector onto a single ball).

    Args:
        params: List of complex or real tensors (any shapes).
        P: Positive total power budget.

    Returns:
        List of new tensors (same length and shapes as ``params``).
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    if len(params) == 0:
        raise ValueError("params must be a non-empty list.")
    total_sq = sum((torch.linalg.norm(p) ** 2) for p in params)
    sqrt_total = torch.sqrt(total_sq)
    sqrt_P = torch.sqrt(
        torch.tensor(P, dtype=sqrt_total.dtype, device=sqrt_total.device)
    )
    scale = torch.where(
        sqrt_total <= sqrt_P, torch.ones_like(sqrt_total), sqrt_P / sqrt_total
    )
    return [p * scale for p in params]


def _bcast_to(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(B,)`` vector to broadcast over the trailing dims of ``like``."""
    return vec.reshape(-1, *([1] * (like.ndim - 1)))


def project_frobenius_ball_batched(A: torch.Tensor, P: float) -> torch.Tensor:
    """Per-element Frobenius-ball projection for a batched tensor.

    ``A`` has a leading batch dimension ``(B, *shape)``; each slice ``A[b]`` is
    projected independently onto {X : ||X||_F^2 <= P}. This is the batch-aware
    counterpart of :func:`project_frobenius_ball` used by the batched
    (multi-start) SPG driver.

    Args:
        A: Complex or real tensor of shape ``(B, *shape)``.
        P: Positive power budget (shared value, applied per element).

    Returns:
        New tensor of the same shape; ``out[b]`` satisfies the constraint.
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    norm_sq = (A.abs() ** 2).flatten(1).sum(1)  # (B,)
    factor = torch.sqrt(torch.clamp(P / norm_sq, max=1.0))  # (B,)
    return A * _bcast_to(factor, A)


def project_total_power_batched(
    params: list[torch.Tensor],
    P: float,
) -> list[torch.Tensor]:
    """Per-element total-power projection for a batched parameter list.

    Each tensor in ``params`` has a leading batch dimension ``(B, *shape_m)``.
    For every batch element ``b`` independently, the stacked vector
    ``{A_m[b]}_m`` is projected onto {sum_m ||A_m[b]||_F^2 <= P} by a single
    per-element rescaling. This is the constraint actually wanted by batched
    multi-start: the ``B`` restarts must NOT be coupled (which is what the
    non-batched :func:`project_total_power` would do, reducing over the whole
    batch to one scalar).

    Args:
        params: Non-empty list of tensors, each shaped ``(B, *shape_m)`` with a
            common leading batch size ``B``.
        P: Positive total-power budget (applied per element).

    Returns:
        List of new tensors (same shapes); each element satisfies its own
        total-power constraint.
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    if len(params) == 0:
        raise ValueError("params must be a non-empty list.")
    total_sq = sum((p.abs() ** 2).flatten(1).sum(1) for p in params)  # (B,)
    factor = torch.sqrt(torch.clamp(P / total_sq, max=1.0))  # (B,)
    return [p * _bcast_to(factor, p) for p in params]
