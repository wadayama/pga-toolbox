# Batched (vectorised) SPG — multi-start design memo

Status: DESIGN (pre-implementation). Target: pga-toolbox v0.4.0.

## 1. Goal

Run `B` independent SPG optimisations — one per random initial point — as a
single vectorised computation over a leading **batch dimension** `B`. On SIMD /
GPU hardware, evaluating `B` initialisations costs ~the same wall-clock as one,
so we get `B` random restarts almost for free. The batch index is the
**multi-start index** (each batch element starts from a different init), NOT a
data minibatch.

This is the mechanism for global search (best-of-`B`) on problems whose MI
landscape has multiple distinct-valued local optima. Such regimes arise in the
originating project's MI-maximisation experiments (e.g. low-power, high-noise
linear-Gaussian network settings, where best-of-12 beats the median single-start
by tens of percent, and a smaller gain persists at high MI). These serve as the
validation testbeds for this feature.

Non-goal (v0.4): non-convex feasible sets (unit-modulus / RIS). SPG assumes a
convex projection; that is a separate work item (v0.5) where multi-start ALSO
becomes essential.

## 2. Public API

```python
def pga_ascent_spg_batched(
    compute_obj,            # closure: () -> Tensor of shape (B,)
    params,                 # list of leaf tensors, each shape (B, *param_shape)
    *,
    projector=None,         # BATCH-AWARE projector (see §5)
    max_iter=500,
    forward_budget=None,    # cap on batched objective evaluations
    nm_window=10,
    alpha0=1.0,             # scalar or (B,) tensor
    alpha_min=1e-10, alpha_max=1e10,
    c=1e-4, shrink=0.5, max_bt=50,
) -> BatchedHistory: ...

def pga_descent_spg_batched(...) -> BatchedHistory: ...   # symmetric, sign=-1
```

Symmetry with scalar SPG: the only structural change is a leading dim `B` on
every param and a `(B,)` return from the closure. With `B == 1` the result must
match `pga_ascent_spg` to tolerance (equivalence test, §10).

### Data-layout convention
- `params[m]` has shape `(B, *shape_m)`; batch element `b` owns
  `params[m][b]`. Elements must be **independent** in the closure (no cross-batch
  ops such as batchnorm). This independence is what makes the gradient trick
  (§4) correct.
- `compute_obj()` returns a real tensor of shape `(B,)` — one objective per
  element.

### Return value `BatchedHistory`
A small dataclass:
- `history: list[Tensor]` — `history[t]` is a `(B,)` tensor of the **best-so-far
  objective** for every element at outer iteration `t` (carried forward after an
  element retires, so the stack is a clean `(T, B)` grid for plotting).
- `best_obj: Tensor (B,)` — final best objective per element.
- `winner: int` — `argmax` (ascent) / `argmin` (descent) over `best_obj`.
On return, `params[m][b]` holds element `b`'s **best-seen** point (per-element
best-point copy-back, generalising scalar SPG). So `compute_obj()` after the
call equals `best_obj`, and `params[m][winner]` is the global incumbent.

## 3. Per-element state

Everything that was a scalar in `spg.py` becomes a `(B,)` tensor or a per-element
mask:

| scalar SPG            | batched SPG                         |
|-----------------------|-------------------------------------|
| `alpha: float`        | `alpha: Tensor (B,)`                |
| `phi: float`          | `phi: Tensor (B,)`                  |
| nonmonotone window    | ring buffer `phi_hist: (B, M)`      |
| `accepted: bool`      | `accepted: BoolTensor (B,)`         |
| `lam: float`          | `lam: Tensor (B,)`                  |
| best point            | `best_phi: (B,)`, `best_params` list|
| (none)                | `active: BoolTensor (B,)` (retired) |

## 4. Gradient via sum-then-backward

The `B` objectives are independent, so

```python
obj = compute_obj()        # (B,)
obj.sum().backward()       # params[m].grad[b] == d(obj_b)/d(params[m][b])
```

gives each element its own gradient with a single backward (cross terms vanish).
`g_asc = [p.grad.clone() for p in params]` is the ascent direction of `phi`
exactly as in scalar SPG (`(-obj).sum().backward()` for descent).

### Batched Wirtinger inner product
Reduce over the parameter list and all **non-batch** dims, keep the batch dim:

```python
def _binner(grads, disps):                 # -> (B,)
    tot = None
    for g, d in zip(grads, disps):
        t = torch.real(g.conj() * d) if g.is_complex() else g * d
        t = t.flatten(1).sum(1)            # sum over non-batch dims
        tot = t if tot is None else tot + t
    return tot
```

## 5. Batch-aware projection (critical subtlety)

The total-power constraint is **per element**: `Σ_m ‖F_m[b]‖² ≤ P` for each `b`.
The existing `project_total_power` reduces over the *whole* list to ONE scalar —
wrong under batching (it would couple the `B` restarts). We therefore ship
batch-aware variants that reduce over non-batch dims and rescale per element:

```python
def project_total_power_batched(params, P):     # each p: (B, *shape)
    total_sq = sum(p.abs().pow(2).flatten(1).sum(1) for p in params)   # (B,)
    factor = torch.sqrt(torch.clamp(P / total_sq, max=1.0))            # (B,)
    return [p * factor.reshape(-1, *([1] * (p.ndim - 1))) for p in params]

def project_frobenius_ball_batched(A, P):       # A: (B, *shape)
    norm_sq = A.abs().pow(2).flatten(1).sum(1)
    factor = torch.sqrt(torch.clamp(P / norm_sq, max=1.0))
    return A * factor.reshape(-1, *([1] * (A.ndim - 1)))
```

Decision: ship these as new public functions (do NOT overload the scalar ones —
silent shape-dependent behaviour is a footgun). The batched driver requires the
projector to be batch-aware; document this loudly.

## 6. Masked per-element line search (the core new logic)

Per outer iteration, over the **active** elements:

```
a        = clamp(alpha, alpha_min, alpha_max)          # (B,)
trial    = [x_m + a * g_m for ...]                     # broadcast a
proj     = batched_project(trial)
d        = [proj_m - x_m]                              # feasible step
inner    = _binner(g_asc, d)                           # (B,)  >0 if ascent dir
stat     = inner <= 1e-16                              # stationary this iter
phi_ref  = phi_hist[:, -nm_window:].min(dim=1)         # (B,)

lam       = ones(B)
accepted  = zeros(B, bool)
acc_phi   = phi.clone()
acc_x     = [x_m.clone() for ...]
for _ in range(max_bt):
    cand   = [x_m + lam * d_m for ...]                 # only matters for ~accepted
    set params = cand
    raw    = compute_obj()                             # (B,)  ONE batched forward
    phicand= sign * raw
    ok     = (~accepted) & ~stat & isfinite(phicand) \
             & (phicand >= phi_ref + c * lam * inner)  # GLL nonmonotone, actual disp
    # record newly-accepted elements
    for m: acc_x[m] = where(ok, cand[m], acc_x[m])
    acc_phi = where(ok, phicand, acc_phi)
    accepted = accepted | ok
    lam = where(accepted, lam, lam * shrink)           # shrink only the unaccepted
    if (accepted | stat).all() or (lam < 1e-20).all(): break

# elements never accepted and not stationary => no progress this round => retire.
```

Notes:
- One **batched** forward per backtrack round; cost is governed by the element
  needing the most backtracks. Already-accepted elements are recomputed but
  masked out — acceptable for moderate `B`; see §8 for compaction.
- Feasibility of `x + lam·d` along the whole segment holds per element because
  each element's feasible set is convex and both endpoints are feasible — no
  re-projection inside the search (same as scalar SPG).
- The sufficient-increase test uses the **actual** displacement `lam·d`
  (`inner` scaled by `lam`), keeping it sound under projection.

## 7. BB step update + retirement

After the search, set `params = where(accepted, acc_x, x)` and run ONE batched
forward+backward to get `g_asc_new` for all elements. Then per element:

```
s     = acc_x - x                        # = lam·d (0 for non-accepted)
y_min = -(g_asc_new - g_asc)             # minimisation-form curvature
sy    = _binner(s, y_min)                # (B,)
ss    = _binner(s, s)                    # (B,)
alpha = where(sy > 1e-14,
              clamp(ss / sy, alpha_min, alpha_max),
              clamp(alpha * 2, max=alpha_max))   # non-convex/flat guard
```

Retirement: an element is marked `active = False` when its line search fails
(stationary / no acceptable step). Retired elements are frozen at their best
point and carried forward in `history`. Loop terminates when `~active.all()` or
`max_iter` / `forward_budget` is hit.

Per-element best-point copy-back: whenever `acc_phi[b] > best_phi[b]`, update
`best_phi[b]` and `best_params[*][b]`. On return, write `best_params` into
`params` (nonmonotone => last ≠ best).

## 8. Performance & compaction (v1 simple, v2 optimisation)

- **v1**: keep the full `B` batch for the whole run. Retired/accepted elements
  are recomputed and masked — simple, correct, some wasted FLOPs.
- **v2**: when `active.sum() < B/2`, `index_select` the live elements into a
  compact batch and continue, remapping at the end. Cuts the tail cost when
  restarts converge at very different rates.
- **v2+ (optional, steady-state population)**: refill retired slots with fresh
  random inits until a global forward budget is spent — turns the driver into a
  continuous multi-start / population search. Defer; note as future work.

Validation target: wall-clock of `B = 12` within a small constant factor of
`B = 1` on CPU (amortised kernel/Python overhead), and best-of-`B` strictly
beats the median single-start in the multimodal regime described in §1.

## 9. Robustness: NaN-safe batched objective (REQUIREMENT)

A batched `torch.linalg.cholesky` / `logdet` throws if **any** element is
non-PD, so the driver cannot use per-element `try/except` (it would lose the
whole batch). Requirement on the batched closure: it must be **NaN-safe** —
return `NaN` for a bad element rather than raising (e.g. via
`torch.linalg.cholesky_ex` + `where`, or sufficient `jitter` in `logdet_hpd`).
The line search already treats non-finite `phi_cand` as a reject, so NaN-safe
objectives integrate cleanly. Document this prominently; provide a helper
wrapper in the example if needed.

## 10. Testing plan (`tests/test_spg_batched.py`)

1. **Equivalence**: `B=1` batched run matches scalar `pga_ascent_spg` to ~1e-8
   on a quadratic (same trajectory).
2. **Independence / no cross-talk**: `B=4` with four different targets; each
   element converges to its own target (validates sum-then-backward).
3. **Batched projection**: per-element power constraint `Σ_m‖F_m[b]‖²≤P` holds
   for every `b`; compare against looping scalar `project_total_power`.
4. **Best-of-B on a bimodal toy**: a hand-built 2-optimum objective where some
   seeds fall into the bad basin; assert `best_obj.max()` reaches the global and
   `winner` points at a good-basin element.
5. **Per-element best-point copy-back** under nonmonotone steps.
6. **Retirement**: elements initialised at the optimum retire immediately while
   others keep going; `history` grid stays `(T, B)`.
7. **Real playground** (in the originating project's experiments, not unit
   tests): best-of-B beats the median single-start in the multimodal regime,
   and the gain persists at high MI.
8. **Invalid args** mirror scalar SPG.

## 11. Open decisions (confirm before coding)

1. **Naming**: `*_spg_batched` functions (proposed) vs a `batch=True` flag on the
   existing SPG (rejected — changes return type/shape semantics).
2. **Return type**: `BatchedHistory` dataclass (proposed) vs a bare
   `list[Tensor(B,)]`. The dataclass carries `best_obj` / `winner` conveniences.
3. **Batched projectors**: ship `project_*_batched` as new public functions
   (proposed) vs overload (rejected).
4. **v1 scope**: full-batch, no compaction, no refill. Compaction (v2) and
   steady-state refill (v2+) deferred.
5. **Version**: release as **v0.4.0**. (Note: the README roadmap also lists
   "more projections" under v0.4; batched multi-start can share that release or
   take v0.4 alone with projections sliding to v0.5 — decide at release time.)

## 12. Summary of new/changed files (at implementation time)

- `pga_toolbox/spg_batched.py` — `pga_ascent_spg_batched`,
  `pga_descent_spg_batched`, `BatchedHistory`, shared `_run_spg_batched_loop`.
- `pga_toolbox/projections.py` — add `project_total_power_batched`,
  `project_frobenius_ball_batched`.
- `pga_toolbox/__init__.py` — exports; `__version__ = "0.4.0"`.
- `tests/test_spg_batched.py` — §10.
- originating project's experiments — playground demo (best-of-B vs
  single-start) in the multimodal regime.
- `README.md` — API table row + a batched-multistart quickstart; roadmap update.
```
