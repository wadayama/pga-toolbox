"""Spectral Projected Gradient (SPG) ascent / descent.

SPG (Birgin, Martinez & Raydan, SIAM J. Optim. 10(4), 2000) combines three
ingredients into a method that is as cheap per iteration as steepest descent
yet exploits curvature like a quasi-Newton method:

  1. A Barzilai-Borwein (BB) spectral step length
         alpha = <s, s> / <s, y>,   s = x_k - x_{k-1},  y = g_k - g_{k-1},
     which encodes the Hessian as a single scalar (1/alpha) * I -- an O(n)
     "poor man's" quasi-Newton estimate of the local curvature.
  2. A projected gradient direction  d = P_Omega(x - alpha * grad) - x  taken
     along the FEASIBLE segment x + lambda * d, lambda in (0, 1]. This assumes
     the feasible set Omega is CONVEX (true for every projection shipped with
     this toolbox: Frobenius ball, total-power ball), so the whole segment is
     feasible and no re-projection is needed inside the line search.
  3. A nonmonotone (Grippo-Lampariello-Lucidi) line search that accepts a step
     when the objective improves relative to the best of the last ``nm_window``
     accepted values -- NOT relative to the immediately previous value. The BB
     step is deliberately nonmonotone; forcing strict monotonicity would
     destroy the very oscillation that lets it traverse ill-conditioned valleys
     quickly.

On the originating project's MI-maximisation smoke benchmark (linear Gaussian
DAG, active total-power constraint), SPG reaches the same optimum as the Armijo
line search with ~6x fewer objective evaluations and ~20x fewer than a tuned
fixed step, without precision loss.

Conventions (shared with :mod:`pga_toolbox.line_search`):
  - The closure returns a scalar ``torch.Tensor``; complex leaves carry the
    real-Euclidean (Wirtinger) gradient in ``.grad``, and all inner products
    use ``Re<conj(g), d>`` via :func:`_wirtinger_real_inner`.
  - ``history[t]`` is the objective (ascent) / cost (descent) at the accepted
    point of outer iteration ``t``. **Unlike the Armijo drivers, this history
    is NOT monotone** -- that is intrinsic to the spectral step.
  - Because the iterates are nonmonotone, the LAST accepted point need not be
    the best one. The driver therefore tracks the best-seen point and copies it
    back into ``params`` on return, so on exit ``params`` always holds the
    incumbent optimum and ``compute_obj()`` there equals ``max``/``min`` of the
    recorded history (for ascent / descent respectively).

Caveat: like the Armijo variants, SPG relies on EXACT, deterministic objective
values and gradients (the BB ratio and the value comparison both break under
minibatch noise). It is not intended for the stochastic setting.
"""

from __future__ import annotations

import torch

from .line_search import _wirtinger_real_inner
from .pga import (
    ObjectiveClosure,
    Projector,
    _apply_projector,
    _check_grad_present,
    _require_leaf_grad,
    _zero_grads,
)


def _project_point(
    params: list[torch.Tensor],
    vals: list[torch.Tensor],
    projector: Projector | None,
) -> list[torch.Tensor]:
    """Return P_Omega(vals): load ``vals`` into ``params`` and project in place.

    Mutates ``params`` (the caller owns its own snapshot of the current
    iterate, so this is safe). Handles both in-place projectors (return
    ``None``) and functional projectors (return a new sequence) via the shared
    :func:`_apply_projector`.
    """
    with torch.no_grad():
        for p, v in zip(params, vals):
            p.copy_(v)
        _apply_projector(params, projector)
    return [p.detach().clone() for p in params]


def _run_spg_loop(
    compute_obj: ObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None,
    max_iter: int,
    forward_budget: int | None,
    nm_window: int,
    alpha0: float,
    alpha_min: float,
    alpha_max: float,
    c: float,
    shrink: float,
    max_bt: int,
    sign: float,
) -> list[float]:
    """Shared SPG driver; ``sign = +1`` ascent, ``sign = -1`` descent.

    Internally maximises ``phi = sign * raw`` where ``raw`` is the closure
    value, so ``phi`` is the objective for ascent and ``-cost`` for descent.
    ``.grad`` is made the ascent direction of ``phi`` by back-propagating
    ``obj`` (ascent) or ``-cost`` (descent), matching the Armijo driver.
    """
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
    if alpha0 <= 0 or alpha_min <= 0 or alpha_max <= 0 or alpha_min >= alpha_max:
        raise ValueError(
            f"step bounds invalid: alpha0={alpha0}, alpha_min={alpha_min}, "
            f"alpha_max={alpha_max}"
        )
    _require_leaf_grad(params)

    def _finite(v: float) -> bool:
        return v == v and v not in (float("inf"), float("-inf"))

    history: list[float] = []
    forward_count = 0
    alpha = alpha0

    # --- Initial point: value, gradient, snapshot. ---
    _zero_grads(params)
    obj_t = compute_obj()
    forward_count += 1
    raw = obj_t.item()
    (obj_t if sign > 0 else -obj_t).backward()
    _check_grad_present(params)
    phi = sign * raw
    # Ascent direction of phi is exactly p.grad (see docstring).
    g_asc = [p.grad.detach().clone() for p in params]
    x = [p.detach().clone() for p in params]

    phi_window: list[float] = [phi]
    best_phi = phi
    best_params = [xi.clone() for xi in x]

    for _ in range(max_iter):
        a = min(alpha_max, max(alpha_min, alpha))
        # Projected gradient direction: P(x + a * g_asc) - x  (ascent on phi).
        trial = [xi + a * gi for xi, gi in zip(x, g_asc)]
        proj_trial = _project_point(params, trial, projector)
        d = [t - xi for t, xi in zip(proj_trial, x)]
        inner_asc = _wirtinger_real_inner(g_asc, d)
        if inner_asc <= 1e-16:
            # Projected gradient ~ 0: first-order stationary, terminate.
            break

        # --- Nonmonotone backtracking along the feasible segment x + lam*d. ---
        phi_ref = min(phi_window[-nm_window:])
        lam = 1.0
        accepted = False
        new_phi = phi
        new_x = x
        for _bt in range(max_bt):
            cand = [xi + lam * di for xi, di in zip(x, d)]
            with torch.no_grad():
                for p, v in zip(params, cand):
                    p.copy_(v)
            try:
                raw_cand = compute_obj().item()
            except Exception:
                forward_count += 1
                lam *= shrink
                continue
            forward_count += 1
            phi_cand = sign * raw_cand
            if _finite(phi_cand) and phi_cand >= phi_ref + c * lam * inner_asc:
                accepted = True
                new_phi = phi_cand
                new_x = cand
                break
            lam *= shrink
        if not accepted:
            # Restore current iterate; no acceptable step => stationary.
            with torch.no_grad():
                for p, xi in zip(params, x):
                    p.copy_(xi)
            break

        # --- Recompute gradient at the accepted point. ---
        _zero_grads(params)
        obj_t = compute_obj()
        forward_count += 1
        raw_new = obj_t.item()
        (obj_t if sign > 0 else -obj_t).backward()
        _check_grad_present(params)
        g_asc_new = [p.grad.detach().clone() for p in params]

        # --- Spectral (BB) step update from this step's (s, y). ---
        s = [nx - xi for nx, xi in zip(new_x, x)]
        # y in MINIMISATION form: grad_min = -g_asc, so y_min = -(g_new - g_old).
        y_min = [-(gn - go) for gn, go in zip(g_asc_new, g_asc)]
        sy = _wirtinger_real_inner(s, y_min)
        ss = _wirtinger_real_inner(s, s)
        if sy > 1e-14:
            alpha = min(alpha_max, max(alpha_min, ss / sy))
        else:
            # Non-convex / flat curvature: grow to keep exploring.
            alpha = min(alpha_max, alpha * 2.0)

        # --- Commit. ---
        x = new_x
        g_asc = g_asc_new
        phi = new_phi
        history.append(raw_new)
        phi_window.append(new_phi)
        if new_phi > best_phi:
            best_phi = new_phi
            best_params = [xi.clone() for xi in x]

        if forward_budget is not None and forward_count >= forward_budget:
            break

    # Nonmonotone: leave params at the best-seen iterate, not the last one.
    with torch.no_grad():
        for p, b in zip(params, best_params):
            p.copy_(b)
    return history


def pga_ascent_spg(
    compute_obj: ObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None = None,
    max_iter: int = 500,
    forward_budget: int | None = None,
    nm_window: int = 10,
    alpha0: float = 1.0,
    alpha_min: float = 1e-10,
    alpha_max: float = 1e10,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_bt: int = 50,
) -> list[float]:
    """Spectral Projected Gradient ASCENT (Barzilai-Borwein + nonmonotone LS).

    A cheap-curvature alternative to :func:`pga_ascent_armijo`: same O(n) cost
    per step, but the spectral step length adapts to the local curvature so it
    typically reaches the optimum in far fewer objective evaluations on
    ill-conditioned problems. The feasible set defined by ``projector`` must be
    CONVEX (every projection in this toolbox is).

    Args:
        compute_obj: Closure returning a scalar torch tensor (objective to
            MAXIMIZE).
        params: List of leaf tensors with ``requires_grad=True`` (real or
            complex).
        projector: Optional Euclidean projector onto a CONVEX set (in-place or
            functional). ``None`` runs unconstrained BB + nonmonotone search.
        max_iter: Maximum number of accepted outer iterations.
        forward_budget: Optional cap on total objective evaluations (including
            backtracks); the loop exits after the acceptance that reaches it.
        nm_window: Nonmonotone window M; the sufficient-increase test compares
            against the best of the last ``nm_window`` accepted values.
        alpha0: Initial spectral step length.
        alpha_min: Lower clamp on the spectral step length.
        alpha_max: Upper clamp on the spectral step length.
        c: Sufficient-increase coefficient, in (0, 1).
        shrink: Backtracking shrink factor, in (0, 1).
        max_bt: Maximum backtracks per outer iteration.

    Returns:
        history: list[float] of accepted OBJECTIVE values, one per successful
        outer iteration. **Not monotone** (intrinsic to the spectral step). On
        return ``params`` holds the best-seen iterate, so ``compute_obj()``
        there equals ``max(history)``. May be empty if the start point is
        already stationary.
    """
    return _run_spg_loop(
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


def pga_descent_spg(
    compute_cost: ObjectiveClosure,
    params: list[torch.Tensor],
    *,
    projector: Projector | None = None,
    max_iter: int = 500,
    forward_budget: int | None = None,
    nm_window: int = 10,
    alpha0: float = 1.0,
    alpha_min: float = 1e-10,
    alpha_max: float = 1e10,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_bt: int = 50,
) -> list[float]:
    """Spectral Projected Gradient DESCENT (Barzilai-Borwein + nonmonotone LS).

    Symmetric to :func:`pga_ascent_spg`, minimising a cost. See that function
    for the argument semantics.

    Returns:
        history: list[float] of accepted COST values, one per successful outer
        iteration (not monotone). On return ``params`` holds the best-seen
        iterate, so ``compute_cost()`` there equals ``min(history)``.
    """
    return _run_spg_loop(
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
