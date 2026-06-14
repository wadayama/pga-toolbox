"""Fixed-step projected gradient ascent / descent.

This module provides the primitive PGA drivers that match the
historical `gaussian-dag.pga_ascent` / `cmi-dag.pga_descent` APIs.
Both are topology-agnostic outer-loop wrappers around a user-supplied
objective closure and parameter list, with optional Euclidean projection
applied after each update.

API conventions:
  - The closure returns a scalar `torch.Tensor` (the objective value).
  - `params` is a list of leaf tensors with ``requires_grad=True``;
    complex tensors are supported and PyTorch's `.grad` is the natural
    Wirtinger (real-Euclidean steepest) direction for ascent.
  - The projector callable takes the parameter list and either:
      (a) mutates the parameters in place (returning ``None``), or
      (b) returns a sequence of new tensors which the driver copies
          back via ``.copy_()``.
    Convention (b) lets users wire `project_frobenius_ball` /
    `project_total_power` directly as the projector without an explicit
    `.copy_` wrapper.
  - History is recorded *before* each update (history[t] is the objective
    at the pre-update parameter values of iteration t).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

ObjectiveClosure = Callable[[], torch.Tensor]
Projector = Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]]


def _apply_projector(params: list[torch.Tensor], projector: Projector | None) -> None:
    """Run the projector under ``no_grad`` and copy results back if functional."""
    if projector is None:
        return
    out = projector(params)
    if out is None:
        return
    if len(out) != len(params):
        raise ValueError(
            f"projector returned {len(out)} tensors, expected {len(params)}."
        )
    for p, q in zip(params, out):
        p.copy_(q)


def _zero_grads(params: list[torch.Tensor]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def _require_leaf_grad(params: list[torch.Tensor]) -> None:
    for idx, p in enumerate(params):
        if not p.requires_grad:
            raise ValueError(
                f"params[{idx}] does not have requires_grad=True."
            )


def _check_grad_present(params: list[torch.Tensor]) -> None:
    for idx, p in enumerate(params):
        if p.grad is None:
            raise RuntimeError(
                f"params[{idx}] received no gradient after backward(): the "
                "parameter has requires_grad=True but does not participate "
                "in the autograd graph produced by the closure. Common "
                "causes: (a) the parameter is declared but never used in "
                "the closure; (b) the closure rebinds the parameter to a "
                "new tensor (e.g. via `F = F.detach()` or in-place "
                "arithmetic outside torch.no_grad); (c) a typo in a "
                "closure-captured variable name."
            )


def pga_ascent(
    compute_obj: ObjectiveClosure,
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Projector | None = None,
) -> list[float]:
    """Run constant-step projected gradient ASCENT on ``compute_obj``.

    Each iteration: zero grads, evaluate closure (record value), call
    ``.backward()`` on the objective tensor, then under ``no_grad``
    update ``p <- p + step_size * p.grad`` and project.

    Args:
        compute_obj: Closure returning a scalar torch tensor (the
            objective to MAXIMIZE).
        params: List of leaf tensors with ``requires_grad=True``.
        step_size: Positive constant step size.
        num_iters: Number of iterations (must be > 0).
        projector: Optional projector callable (in-place or functional).

    Returns:
        history: list[float] of length ``num_iters``, where
        ``history[t]`` is the objective evaluated at the pre-update
        parameter values of iteration ``t``.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    _require_leaf_grad(params)

    history: list[float] = []
    for _ in range(num_iters):
        _zero_grads(params)
        obj = compute_obj()
        obj.backward()
        history.append(obj.item())
        _check_grad_present(params)
        with torch.no_grad():
            for p in params:
                p.add_(step_size * p.grad)
            _apply_projector(params, projector)
    return history


def pga_descent(
    compute_cost: ObjectiveClosure,
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Projector | None = None,
) -> list[float]:
    """Run constant-step projected gradient DESCENT on ``compute_cost``.

    Internally equivalent to negating the closure and running
    :func:`pga_ascent`; the returned history records the cost values
    (positive, descending) at the pre-update parameter values.

    Args:
        compute_cost: Closure returning a scalar torch tensor (the cost
            to MINIMIZE).
        params: List of leaf tensors with ``requires_grad=True``.
        step_size: Positive constant step size.
        num_iters: Number of iterations (must be > 0).
        projector: Optional projector callable (in-place or functional).

    Returns:
        history: list[float] of length ``num_iters`` recording the COST
        at each pre-update iteration.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    _require_leaf_grad(params)

    history: list[float] = []
    for _ in range(num_iters):
        _zero_grads(params)
        cost = compute_cost()
        # Descent on cost == ascent on (-cost); the gradient stored in
        # p.grad is the descent direction (negative of cost-ascent grad).
        (-cost).backward()
        history.append(cost.item())
        _check_grad_present(params)
        with torch.no_grad():
            for p in params:
                p.add_(step_size * p.grad)
            _apply_projector(params, projector)
    return history
