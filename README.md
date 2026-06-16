# pga-toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

Projected gradient ascent / descent for complex-valued (Wirtinger) and
real parameters, with fixed-step and Armijo backtracking line search
variants. Built on PyTorch; depends only on `torch`.

This library extracts the small but recurrent optimisation core that
several companion research libraries of the author (e.g. `gaussian-dag`,
`cmi-dag`, `fading-dag`, `bussgang-dag` — internal projects, not
necessarily public) have been copy-vendoring. The goal is a single
source of truth so improvements (Armijo line search now, BB / SPG
later) reach every dependent library at once.

## Why

PyTorch's `.grad` on a complex leaf with a real-valued objective is the
natural Wirtinger gradient — the real-Euclidean steepest-ascent
direction on the (real, imaginary) lift. Generic optimisation libraries
either assume real parameters or assume non-projected updates. This
toolbox handles both:

- Complex Wirtinger parameters out of the box.
- Optional Euclidean projection applied after each accepted step.
- A persistent-step Armijo line search that adapts to the problem
  scale (no manual `step_size` tuning) and consistently beats a fixed
  step on hard problems — see `examples/` and the smoke test that
  motivated this library.

## Install

```bash
git clone https://github.com/wadayama/pga-toolbox.git
cd pga-toolbox
uv sync                   # creates .venv with torch as the sole runtime dependency
uv run pytest             # unit tests for pga, line_search, projections, SPG;
                          # each runs on CPU and, when a GPU is present, on CUDA
```

Or as a path dependency from a sister project's `pyproject.toml`:

```toml
[tool.uv.sources]
pga-toolbox = { path = "../pga-toolbox" }
```

## Quickstart

### Fixed-step projected gradient ascent

```python
import torch
from pga_toolbox import pga_ascent, project_total_power

F_list = [
    torch.randn(4, 4, dtype=torch.complex128).requires_grad_(True)
    for _ in range(9)
]

def closure():
    return my_mi_objective(F_list)        # any scalar torch.Tensor

def projector(params):
    return project_total_power(params, P=36.0)

history = pga_ascent(
    closure, F_list,
    step_size=0.05, num_iters=200, projector=projector,
)
print(f"final objective = {history[-1]:.4f}")
```

### Adaptive Armijo line search (recommended)

The persistent-step Armijo variant typically reaches the same (or
better) objective in far fewer iterations and removes the need to tune
`step_size`:

```python
from pga_toolbox import pga_ascent_armijo

history = pga_ascent_armijo(
    closure, F_list,
    projector=projector,
    max_iter=200,
)
```

That is the entire API change. All other arguments are optional and
default to the values verified on a single-link MIMO benchmark from the
originating methodology (1792-iter fixed-step → 5-iter Armijo; see the
citation below).

### Spectral Projected Gradient (SPG) — fastest on hard problems

SPG (Barzilai–Borwein spectral step + nonmonotone projected line search)
is as cheap per iteration as steepest descent but adapts to the local
curvature like a quasi-Newton method. On the project's MI-maximisation
smoke benchmark it reaches the same optimum as Armijo with ~6× fewer
objective evaluations (and ~20× fewer than a tuned fixed step), with no
precision loss. The feasible set defined by `projector` must be **convex**
(every projection shipped here is).

```python
from pga_toolbox import pga_ascent_spg

history = pga_ascent_spg(
    closure, F_list,
    projector=projector,
    max_iter=200,
)
```

Two semantic differences from the Armijo variants, both intrinsic to the
spectral step: the returned `history` is **not monotone**, and on return
`params` holds the **best-seen iterate** (not the last), so the objective
there equals `max(history)` for ascent (`min` for descent).

### Batched parallel multi-start SPG

`pga_ascent_spg_batched` runs `B` independent SPG optimisations — one per
random initial point — as a single vectorised computation over a leading
**batch dimension**. On SIMD / GPU hardware `B` restarts cost ~the same
wall-clock as one, so multi-start is nearly free. This is the tool for
problems whose landscape has multiple distinct-valued local optima.

```python
import torch
from pga_toolbox import pga_ascent_spg_batched, project_total_power_batched

B = 16
F_list = [
    torch.randn(B, 4, 4, dtype=torch.complex128).requires_grad_(True)
    for _ in range(9)
]

def closure():
    return my_mi_objective(F_list)        # returns a real tensor of shape (B,)

res = pga_ascent_spg_batched(
    closure, F_list,
    projector=lambda ps: project_total_power_batched(ps, P=36.0),
    max_iter=200,
)
print(f"best of {B}: {res.best_obj.max():.4f}  (winner = #{res.winner})")
```

Requirements: `params[m]` has shape `(B, *shape_m)`; the closure returns a
`(B,)` tensor; batch elements must be **independent** (no cross-batch ops),
which makes the single-backward gradient correct; and the projector must be
**batch-aware** (per element) — use the `*_batched` projections. The closure
should be **NaN-safe** (return `NaN` for a bad element rather than raising), so
one ill-conditioned restart cannot abort the whole batch. On return,
`params[m][b]` holds element `b`'s best-seen point and `res.winner` indexes the
global incumbent. See `notes/BATCHED_SPG_DESIGN.md`.

### Descent variants

Symmetric ascent / descent wrappers:

```python
from pga_toolbox import (
    pga_descent, pga_descent_armijo, pga_descent_spg, pga_descent_spg_batched,
)

history = pga_descent_armijo(cost_closure, params, projector=projector)
history = pga_descent_spg(cost_closure, params, projector=projector)
res = pga_descent_spg_batched(cost_closure, params, projector=projector_batched)
```

## Public API

| function | role | typical use |
| --- | --- | --- |
| `pga_ascent` | fixed-step projected gradient ASCENT | baseline / known good step size |
| `pga_descent` | fixed-step projected gradient DESCENT | minimise a cost |
| `pga_ascent_armijo` | Armijo line search ASCENT (persistent step) | no `step_size` tuning |
| `pga_descent_armijo` | Armijo line search DESCENT | symmetric descent |
| `pga_ascent_spg` | Spectral Projected Gradient ASCENT (BB + nonmonotone) | recommended; fewest evals on hard / ill-conditioned problems (convex constraint) |
| `pga_descent_spg` | Spectral Projected Gradient DESCENT | symmetric descent |
| `pga_ascent_spg_batched` | batched parallel multi-start SPG ASCENT | `B` random restarts at ~the cost of one; multimodal landscapes |
| `pga_descent_spg_batched` | batched parallel multi-start SPG DESCENT | symmetric descent |
| `project_frobenius_ball` | project one matrix onto `{X : ‖X‖_F^2 ≤ P}` | per-matrix power constraint |
| `project_total_power` | project a list onto `{Σ_m ‖A_m‖_F^2 ≤ P}` | shared total power budget |
| `project_frobenius_ball_batched` / `project_total_power_batched` | per-element projections over a leading batch dim | batched multi-start |

The `pga_*` drivers accept three closure / parameter conventions:

1. **Closure**: `() -> torch.Tensor` returning a scalar.
2. **Parameters**: `list[torch.Tensor]` with `requires_grad=True`,
   real or complex.
3. **Projector**: `(params) -> None | Sequence[Tensor]`. In-place
   projectors return `None`; functional projectors return a sequence
   that the driver copies back via `.copy_()`. Both conventions are
   accepted within the same call.

## Conventions

- **Wirtinger gradient**: `tensor.grad` on a complex leaf with a real
  scalar loss is the real-Euclidean steepest-ascent direction; the
  drivers treat it as such (no factor-of-two correction is applied at
  the driver level; absorb it into the step size if you prefer the
  literature convention).
- **History semantics**: `history[t]` is the objective evaluated at the
  *pre-update* parameters of iteration `t`. The Armijo variants log
  *accepted* objective values, one per successful outer iteration.
- **Backward compatibility**: the fixed-step API matches the legacy
  `pga_ascent` / `pga_descent` signature used by `gaussian-dag`,
  `cmi-dag`, `fading-dag`, and `bussgang-dag` so each sister library
  can drop in a dependency on `pga-toolbox` without altering its
  callers.

## Roadmap

- v0.1: fixed-step + Armijo (deterministic).
- v0.3: Spectral Projected Gradient (SPG) — Barzilai–Borwein spectral step +
  nonmonotone projected line search. Validated on the MI-maximisation smoke
  benchmark (~6× fewer evals than Armijo, ~20× fewer than a tuned fixed step,
  same optimum).
- v0.4 (this release): batched parallel multi-start SPG — `B` random restarts
  over a leading batch dimension at ~the cost of one. Validated on a multimodal
  MI regime (best-of-16 beats the median single-start by tens of percent, with
  B=16 wall-clock ≈ B=1). See `notes/BATCHED_SPG_DESIGN.md`.
- v0.2 (planned): stochastic SGD ascent / descent + projection
  (closure-resamples convention from `fading-dag`). Note: the deterministic
  line-search / BB machinery (Armijo, SPG) is **not** directly usable under
  minibatch noise.
- v0.5+ (planned): non-convex projections (unit modulus for RIS phases,
  simplex, rank constraints) — where multi-start becomes essential.

## License

MIT — see `LICENSE`.

## Citation

If this toolbox underpins a publication, please cite the originating
methodology paper:

> T. Wadayama and Na Siqi, *Mutual Information Optimization via
> K-Recursion and Automatic Differentiation for Linear Gaussian
> Wireless Networks*, arXiv:2606.06982 \[cs.IT\], 2026.

The Armijo persistent-step convention is described in the originating
methodology (see the citation above), verified to give a 1792-iter to
5-iter speedup on a single-link MIMO smoke benchmark.
