# pga-toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

Projected gradient ascent / descent for complex-valued (Wirtinger) and
real parameters, with fixed-step and Armijo backtracking line search
variants. Built on PyTorch; depends only on `torch`.

This library extracts the small but recurrent optimisation core that
several sister libraries (`gaussian-dag`, `cmi-dag`, `fading-dag`,
`bussgang-dag`, ...) have been copy-vendoring. The goal is a single
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
uv run pytest             # 24 unit tests across pga, line_search, projections
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
default to the values verified on the cmi-dag single-link MIMO
benchmark (1792-iter fixed-step → 5-iter Armijo, see
`PGD_IMPROVEMENT.md` in this repository's `notes/`).

### Descent variants

Symmetric ascent / descent wrappers:

```python
from pga_toolbox import pga_descent, pga_descent_armijo

history = pga_descent_armijo(cost_closure, params, projector=projector)
```

## Public API

| function | role | typical use |
| --- | --- | --- |
| `pga_ascent` | fixed-step projected gradient ASCENT | baseline / known good step size |
| `pga_descent` | fixed-step projected gradient DESCENT | minimise a cost |
| `pga_ascent_armijo` | Armijo line search ASCENT (persistent step) | recommended; no `step_size` tuning |
| `pga_descent_armijo` | Armijo line search DESCENT | symmetric descent |
| `project_frobenius_ball` | project one matrix onto `{X : ‖X‖_F^2 ≤ P}` | per-matrix power constraint |
| `project_total_power` | project a list onto `{Σ_m ‖A_m‖_F^2 ≤ P}` | shared total power budget |

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

- v0.1 (this release): fixed-step + Armijo (deterministic).
- v0.2 (planned): stochastic SGD ascent / descent + projection
  (closure-resamples convention from `fading-dag`).
- v0.3 (planned): Barzilai–Borwein step (PGD_IMPROVEMENT.md §3.2)
  and Spectral Projected Gradient (SPG, §3.3).
- v0.4+ (planned): more projections (unit modulus for RIS phases,
  simplex, rank constraints, etc.).

## License

MIT — see `LICENSE`.

## Citation

If this toolbox underpins a publication, please cite the originating
methodology paper:

> T. Wadayama and Na Siqi, *Mutual Information Optimization via
> K-Recursion and Automatic Differentiation for Linear Gaussian
> Wireless Networks*, arXiv:2606.06982 \[cs.IT\], 2026.

The Armijo persistent-step convention is documented in
`PGD_IMPROVEMENT.md` (in this repository's `notes/` of the originating
project), verified to give 1792-iter to 5-iter speedup on the cmi-dag
smoke benchmark.
