"""Batched (vectorised) Spectral Projected Gradient — parallel multi-start.

Run ``B`` independent SPG optimisations, one per random initial point, as a
single vectorised computation over a leading **batch dimension** ``B``. On SIMD
/ GPU hardware, evaluating ``B`` initialisations costs ~the same wall-clock as
one, so ``B`` random restarts come almost for free. The batch index is the
**multi-start index** (each element starts from a different init), NOT a data
minibatch.

This is the mechanism for global search (best-of-``B``) on problems whose
landscape has multiple distinct-valued local optima (see the originating
project's ``pga-smoke`` playground search). The per-element local solver is the
same SPG as :mod:`pga_toolbox.spg`.

Design: see ``notes/BATCHED_SPG_DESIGN.md``. Key points:
  - ``params[m]`` has shape ``(B, *shape_m)``; element ``b`` owns
    ``params[m][b]``. Batch elements must be INDEPENDENT in the closure (no
    cross-batch ops); that independence makes the single-backward gradient
    trick correct: ``compute_obj().sum().backward()`` gives each element its own
    gradient.
  - ``compute_obj()`` returns a real tensor of shape ``(B,)``.
  - The projector must be BATCH-AWARE (per-element), e.g.
    :func:`pga_toolbox.project_total_power_batched`. The non-batched projectors
    would couple the restarts and are wrong here.
  - Every scalar of :mod:`pga_toolbox.spg` becomes a ``(B,)`` tensor: step size
    ``alpha``, objective ``phi``, accept mask, and ``lambda`` in a masked
    per-element backtracking line search. Elements that reach a stationary point
    are RETIRED (frozen at their best) and carried forward.
  - Per-element best-point copy-back: on return ``params[m][b]`` holds element
    ``b``'s best-seen point (the spectral step is nonmonotone, so last != best).

Robustness requirement: a batched ``cholesky`` / ``logdet`` raises if ANY
element is non-PD, so the driver cannot per-element ``try/except``. The batched
closure must be **NaN-safe** — return ``NaN`` for a bad element rather than
raising (e.g. via ``torch.linalg.cholesky_ex`` + ``where`` or sufficient
``jitter``). The line search treats non-finite objectives as rejects.

Like the scalar SPG / Armijo drivers, this relies on EXACT, deterministic
objective values and gradients; it is not for the stochastic (minibatch-noise)
setting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .pga import (
    Projector,
    _apply_projector,
    _check_grad_present,
    _require_leaf_grad,
    _zero_grads,
)

# A batched objective closure returns a real tensor of shape (B,).
BatchedObjectiveClosure = Callable[[], torch.Tensor]


@dataclass
class BatchedHistory:
    """Result of a batched (multi-start) SPG run.

    Attributes:
        history: List of ``(B,)`` tensors; ``history[t]`` is the best-so-far
            objective (ascent) / cost (descent) of every element at outer
            iteration ``t`` (carried forward after an element retires, so the
            stack is a clean ``(T, B)`` grid).
        best_obj: ``(B,)`` tensor of the final best objective / cost per element.
        winner: Index of the global incumbent element (``argmax`` for ascent,
            equivalently the element with the best ``best_obj``).
    """

    history: list[torch.Tensor]
    best_obj: torch.Tensor
    winner: int


def _bcast(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return vec.reshape(-1, *([1] * (like.ndim - 1)))


def _binner(
    grads: Sequence[torch.Tensor],
    disps: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Per-element real-Euclidean (Wirtinger) inner product -> ``(B,)``."""
    total: torch.Tensor | None = None
    for g, d in zip(grads, disps):
        t = torch.real(g.conj() * d) if g.is_complex() else g * d
        t = t.flatten(1).sum(1)
        total = t if total is None else total + t
    return total


def _project_batched(
    params: list[torch.Tensor],
    vals: list[torch.Tensor],
    projector: Projector | None,
) -> list[torch.Tensor]:
    """Load ``vals`` into ``params`` and project in place; return a snapshot."""
    with torch.no_grad():
        for p, v in zip(params, vals):
            p.copy_(v)
        _apply_projector(params, projector)
    return [p.detach().clone() for p in params]


def _run_spg_batched_loop(
    compute_obj: BatchedObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None,
    max_iter: int,
    forward_budget: int | None,
    nm_window: int,
    alpha0: float | torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    c: float,
    shrink: float,
    max_bt: int,
    sign: float,
) -> BatchedHistory:
    """Shared batched-SPG driver; ``sign = +1`` ascent, ``sign = -1`` descent."""
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if forward_budget is not None and forward_budget <= 0:
        raise ValueError(
            f"forward_budget must be positive when set, got {forward_budget}"
        )
    if not (0.0 < c < 1.0):
        raise ValueError(f"c must lie in (0, 1), got {c}")
    if not (0.0 < shrink < 1.0):
        raise ValueError(f"shrink must lie in (0, 1), got {shrink}")
    if nm_window < 1:
        raise ValueError(f"nm_window must be >= 1, got {nm_window}")
    if max_bt < 1:
        raise ValueError(f"max_bt must be >= 1, got {max_bt}")
    if alpha_min <= 0 or alpha_max <= 0 or alpha_min >= alpha_max:
        raise ValueError(
            f"step bounds invalid: alpha_min={alpha_min}, alpha_max={alpha_max}"
        )
    if len(params) == 0:
        raise ValueError("params must be a non-empty list.")
    _require_leaf_grad(params)

    p0 = params[0]
    device = p0.device
    rdtype = p0.real.dtype
    B = p0.shape[0]
    for idx, p in enumerate(params):
        if p.shape[0] != B:
            raise ValueError(
                f"params[{idx}] has batch size {p.shape[0]}, expected {B}."
            )

    if isinstance(alpha0, torch.Tensor):
        alpha = alpha0.to(dtype=rdtype, device=device).reshape(B).clone()
    else:
        if alpha0 <= 0:
            raise ValueError(f"alpha0 must be positive, got {alpha0}")
        alpha = torch.full((B,), float(alpha0), dtype=rdtype, device=device)

    # --- Initial value, gradient, snapshot. ---
    _zero_grads(params)
    obj = compute_obj()
    forward_count = 1
    raw = obj.detach().reshape(B)
    (obj if sign > 0 else -obj).sum().backward()
    _check_grad_present(params)
    phi = sign * raw
    g_asc = [p.grad.detach().clone() for p in params]
    x = [p.detach().clone() for p in params]

    phi_window: list[torch.Tensor] = [phi.clone()]
    best_phi = phi.clone()
    best_params = [xi.clone() for xi in x]
    active = torch.ones(B, dtype=torch.bool, device=device)
    history: list[torch.Tensor] = []

    for _ in range(max_iter):
        if not bool(active.any()):
            break

        a = torch.clamp(alpha, alpha_min, alpha_max)
        trial = [xi + _bcast(a, xi) * gi for xi, gi in zip(x, g_asc)]
        proj = _project_batched(params, trial, projector)
        mask_r = active.to(rdtype)
        d = [(pt - xi) * _bcast(mask_r, xi) for pt, xi in zip(proj, x)]
        inner = _binner(g_asc, d)  # (B,)
        stat = inner <= 1e-16  # includes inactive (d == 0 => inner == 0)

        win = torch.stack(phi_window[-nm_window:], 0)  # (w, B)
        phi_ref = win.min(0).values  # (B,)

        lam = torch.ones(B, dtype=rdtype, device=device)
        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        acc_phi = phi.clone()
        acc_x = [xi.clone() for xi in x]

        for _bt in range(max_bt):
            cand = [xi + _bcast(lam, xi) * di for xi, di in zip(x, d)]
            with torch.no_grad():
                for p, cv in zip(params, cand):
                    p.copy_(cv)
                raw_c = compute_obj().detach().reshape(B)
            forward_count += 1
            phic = sign * raw_c
            ok = (
                (~accepted)
                & (~stat)
                & torch.isfinite(phic)
                & (phic >= phi_ref + c * lam * inner)
            )
            if bool(ok.any()):
                for m in range(len(acc_x)):
                    acc_x[m] = torch.where(_bcast(ok, acc_x[m]), cand[m], acc_x[m])
                acc_phi = torch.where(ok, phic, acc_phi)
                accepted = accepted | ok
            lam = torch.where(accepted | stat, lam, lam * shrink)
            if bool((accepted | stat).all()) or bool((lam < 1e-20).all()):
                break

        # --- Gradient at the accepted points (acc_x == x where not accepted). ---
        with torch.no_grad():
            for p, av in zip(params, acc_x):
                p.copy_(av)
        _zero_grads(params)
        obj = compute_obj()
        forward_count += 1
        (obj if sign > 0 else -obj).sum().backward()
        _check_grad_present(params)
        g_asc_new = [p.grad.detach().clone() for p in params]

        # --- Spectral (BB) step update, per element. ---
        s = [av - xi for av, xi in zip(acc_x, x)]
        y_min = [-(gn - go) for gn, go in zip(g_asc_new, g_asc)]
        sy = _binner(s, y_min)
        ss = _binner(s, s)
        good = sy > 1e-14
        bb = torch.where(
            good,
            torch.clamp(ss / torch.where(good, sy, torch.ones_like(sy)),
                        alpha_min, alpha_max),
            torch.clamp(alpha * 2.0, max=alpha_max),
        )
        alpha = torch.where(active, bb, alpha)

        # --- Commit. ---
        x = acc_x
        g_asc = g_asc_new
        phi = acc_phi.clone()
        phi_window.append(phi.clone())

        improved = phi > best_phi
        if bool(improved.any()):
            best_phi = torch.where(improved, phi, best_phi)
            for m in range(len(best_params)):
                best_params[m] = torch.where(
                    _bcast(improved, best_params[m]), x[m], best_params[m]
                )

        # Active elements that made no progress this round are stationary: retire.
        active = active & accepted
        history.append((sign * best_phi).clone())

        if forward_budget is not None and forward_count >= forward_budget:
            break

    # Nonmonotone: leave params at the per-element best-seen iterate.
    with torch.no_grad():
        for p, b in zip(params, best_params):
            p.copy_(b)
    best_obj = sign * best_phi
    winner = int(torch.argmax(best_phi).item())
    return BatchedHistory(history=history, best_obj=best_obj, winner=winner)


def pga_ascent_spg_batched(
    compute_obj: BatchedObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None = None,
    max_iter: int = 500,
    forward_budget: int | None = None,
    nm_window: int = 10,
    alpha0: float | torch.Tensor = 1.0,
    alpha_min: float = 1e-10,
    alpha_max: float = 1e10,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_bt: int = 50,
) -> BatchedHistory:
    """Batched (parallel multi-start) Spectral Projected Gradient ASCENT.

    Runs ``B`` independent SPG ascents over a leading batch dimension. With
    ``B == 1`` the trajectory matches :func:`pga_toolbox.pga_ascent_spg`.

    Args:
        compute_obj: Closure returning a real tensor of shape ``(B,)`` (one
            objective per element). Must be NaN-safe and have INDEPENDENT batch
            elements (see module docstring).
        params: Non-empty list of leaf tensors, each shaped ``(B, *shape_m)``
            with ``requires_grad=True`` (real or complex).
        projector: BATCH-AWARE Euclidean projector onto a CONVEX set, e.g.
            :func:`pga_toolbox.project_total_power_batched`. ``None`` runs
            unconstrained.
        max_iter: Maximum outer iterations.
        forward_budget: Optional cap on batched objective evaluations.
        nm_window: Nonmonotone window M.
        alpha0: Initial spectral step (scalar or ``(B,)`` tensor).
        alpha_min: Lower clamp on the step.
        alpha_max: Upper clamp on the step.
        c: Sufficient-increase coefficient, in (0, 1).
        shrink: Backtracking shrink factor, in (0, 1).
        max_bt: Maximum backtracks per outer iteration.

    Returns:
        :class:`BatchedHistory`. On return ``params[m][b]`` holds element
        ``b``'s best-seen point; ``params[m][result.winner]`` is the global
        incumbent and ``compute_obj()`` equals ``result.best_obj``.
    """
    return _run_spg_batched_loop(
        compute_obj=compute_obj,
        params=params,
        projector=projector,
        max_iter=max_iter,
        forward_budget=forward_budget,
        nm_window=nm_window,
        alpha0=alpha0,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        c=c,
        shrink=shrink,
        max_bt=max_bt,
        sign=+1.0,
    )


def pga_descent_spg_batched(
    compute_cost: BatchedObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None = None,
    max_iter: int = 500,
    forward_budget: int | None = None,
    nm_window: int = 10,
    alpha0: float | torch.Tensor = 1.0,
    alpha_min: float = 1e-10,
    alpha_max: float = 1e10,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_bt: int = 50,
) -> BatchedHistory:
    """Batched (parallel multi-start) Spectral Projected Gradient DESCENT.

    Symmetric to :func:`pga_ascent_spg_batched`, minimising a cost. See that
    function for argument semantics. ``result.best_obj`` holds the per-element
    minimum cost and ``result.winner`` is the lowest-cost element.
    """
    return _run_spg_batched_loop(
        compute_obj=compute_cost,
        params=params,
        projector=projector,
        max_iter=max_iter,
        forward_budget=forward_budget,
        nm_window=nm_window,
        alpha0=alpha0,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        c=c,
        shrink=shrink,
        max_bt=max_bt,
        sign=-1.0,
    )
