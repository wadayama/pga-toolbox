"""Generate the repository's visual abstract (docs/figures/visual_abstract.{png,pdf}).

Two panels, both produced from *actual* runs of the toolbox (no hand-drawn or
synthetic curves):

  (A) Convergence under an active constraint. An ill-conditioned complex
      quadratic (kappa = 1000) is maximised on a Frobenius/2-norm ball with the
      constraint active. We plot the suboptimality gap  f* - f_best-so-far
      versus the number of objective evaluations for the fixed-step, Armijo, and
      SPG drivers. The adaptive methods reach the optimum in far fewer
      evaluations than a (stably) tuned fixed step.

  (B) Multi-start escapes local optima. A deliberately multimodal objective
      (a sum of Gaussian bumps of distinct heights) is optimised on a ball by
      batched parallel multi-start SPG with B random restarts. The histogram of
      final values shows most single starts land on lower bumps while best-of-B
      finds the tallest.

The objective evaluation count in (A) is measured by instrumenting the closure
(every forward call is logged); the curve is the running best, i.e. the best
objective found within k evaluations — the honest "evaluations to optimum"
metric, uniform across methods including line-search backtracks.

Run:
    uv sync --extra examples
    uv run python examples/visual_abstract.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_armijo,
    pga_ascent_spg,
    pga_ascent_spg_batched,
    project_total_power,
    project_total_power_batched,
)

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DTYPE = torch.complex128
RDTYPE = torch.float64

# Monochrome-leaning palette: greys for the baselines, one accent for the
# recommended method / the best-of-B marker.
GREY_LIGHT = "0.70"
GREY_MID = "0.45"
INK = "0.10"
ACCENT = "#1f5fa8"


# ---------------------------------------------------------------------------
# Panel A: convergence on an ill-conditioned constrained complex quadratic
# ---------------------------------------------------------------------------
def panel_a(ax) -> None:
    torch.manual_seed(0)
    n, kappa, P = 40, 1.0e3, 1.0

    # Hermitian PD A = U diag(lambda) U^H, eigenvalues log-spaced in [1, kappa].
    lam = torch.logspace(0, np.log10(kappa), n, dtype=RDTYPE)
    M = torch.randn(n, n, dtype=DTYPE)
    U, _ = torch.linalg.qr(M)                      # random unitary
    A = (U * lam.to(DTYPE)) @ U.mH                 # Hermitian PD
    A = 0.5 * (A + A.mH)
    b = torch.randn(n, dtype=DTYPE)

    def objective(x: torch.Tensor) -> torch.Tensor:
        # f(x) = -Re(x^H A x) + 2 Re(b^H x); concave, constraint ||x||^2 <= P.
        quad = torch.real(torch.vdot(x, A @ x))
        lin = torch.real(torch.vdot(b, x))
        return -quad + 2.0 * lin

    def make_logged_closure():
        log: list[float] = []
        x = torch.zeros(n, dtype=DTYPE, requires_grad=True)

        def closure() -> torch.Tensor:
            val = objective(x)
            log.append(float(val.detach()))
            return val

        return closure, [x], log

    def projector(params):
        return project_total_power(params, P)

    runs = {}
    # Fixed step: stable bound is alpha < 1/lambda_max for this scaling.
    c, p, log = make_logged_closure()
    pga_ascent(c, p, step_size=1.0 / kappa, num_iters=4000, projector=projector)
    runs["fixed-step"] = log
    # Armijo.
    c, p, log = make_logged_closure()
    pga_ascent_armijo(c, p, projector=projector, max_iter=400, forward_budget=600)
    runs["Armijo"] = log
    # SPG.
    c, p, log = make_logged_closure()
    pga_ascent_spg(c, p, projector=projector, max_iter=400, forward_budget=600)
    runs["SPG"] = log

    # Optimum: best objective seen across all runs (high-precision reference).
    f_star = max(max(v) for v in runs.values())

    def gap_curve(values):
        best = np.maximum.accumulate(np.array(values, dtype=np.float64))
        gap = f_star - best
        gap = np.clip(gap, 1e-6, None)             # floor for the log axis
        return np.arange(1, len(gap) + 1), gap

    styles = {
        "fixed-step": dict(color=GREY_LIGHT, ls=":", lw=2.0),
        "Armijo": dict(color=GREY_MID, ls="--", lw=1.8),
        "SPG": dict(color=ACCENT, ls="-", lw=2.2),
    }
    for name in ("fixed-step", "Armijo", "SPG"):
        k, gap = gap_curve(runs[name])
        ev = len(runs[name])
        ax.semilogy(k, gap, label=f"{name}  ({ev} evals)", **styles[name])

    ax.set_xlim(0, 320)
    ax.set_ylim(1e-6, 5.0)
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel(r"suboptimality  $f^\star - f_{\mathrm{best}}$")
    ax.set_title("(A)  convergence under an active constraint  ($\\kappa=10^3$)",
                 fontsize=10, loc="left")
    ax.grid(True, which="both", lw=0.3, color="0.85")
    ax.legend(frameon=False, fontsize=8, loc="lower left")


# ---------------------------------------------------------------------------
# Panel B: multi-start on a multimodal objective (batched SPG)
# ---------------------------------------------------------------------------
def panel_b(ax) -> None:
    torch.manual_seed(1)
    n, B, P = 2, 256, 1.0

    # Multimodal objective: a smooth upper envelope (log-sum-exp) of K
    # paraboloids of distinct heights, one local maximum per anchor with a
    # Voronoi-like basin that tiles the ball, so every random start climbs to
    # *some* peak. f(x) = (1/beta) log sum_k exp( beta (h_k - c ||x - a_k||^2) ).
    K, beta, c = 6, 10.0, 3.0
    ang = torch.linspace(0, 2 * np.pi, K + 1)[:K]
    anchors = 0.65 * torch.stack([torch.cos(ang), torch.sin(ang)], 1).to(RDTYPE)
    heights = torch.linspace(0.4, 1.0, K, dtype=RDTYPE)[torch.randperm(K)]

    def objective_batched(x: torch.Tensor) -> torch.Tensor:  # x: (B, n) -> (B,)
        d2 = ((x[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)   # (B, K)
        return torch.logsumexp(beta * (heights[None, :] - c * d2), dim=1) / beta

    # Random starts uniformly in the ball of radius sqrt(P).
    r = torch.sqrt(torch.rand(B, dtype=RDTYPE)) * (P ** 0.5)
    th = torch.rand(B, dtype=RDTYPE) * 2 * np.pi
    x0 = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    x = x0.clone().requires_grad_(True)

    def closure() -> torch.Tensor:
        return objective_batched(x)

    def projector(params):
        return project_total_power_batched(params, P)

    result = pga_ascent_spg_batched(
        closure, [x], projector=projector, max_iter=300, forward_budget=600,
    )
    finals = result.best_obj.detach().cpu().numpy()                 # (B,)

    median = float(np.median(finals))
    best = float(finals.max())

    ax.set_xlim(0.35, 1.05)
    ax.hist(finals, bins=24, color=GREY_LIGHT, edgecolor=GREY_MID, lw=0.5)
    ax.axvline(median, color=INK, ls="--", lw=1.5,
               label=f"median start = {median:.2f}")
    ax.axvline(best, color=ACCENT, ls="-", lw=2.2,
               label=f"best-of-{B} = {best:.2f}")
    ax.set_xlabel("final objective")
    ax.set_ylabel(f"count over {B} random starts")
    ax.set_title("(B)  multi-start escapes local optima", fontsize=10, loc="left")
    ax.grid(True, axis="y", lw=0.3, color="0.85")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    panel_a(axA)
    panel_b(axB)
    fig.suptitle(
        "pga-toolbox — projected gradient ascent/descent for complex (Wirtinger) "
        "& real parameters",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"visual_abstract.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
