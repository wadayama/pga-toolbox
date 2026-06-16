# Mathematical Foundations

> Implementation-oriented summary of the optimisation methods in
> `pga-toolbox`: the constrained first-order problem they solve, the
> Wirtinger-gradient convention for complex parameters, the closed-form
> Euclidean projections, and the four ascent/descent drivers —
> **fixed-step**, **Armijo backtracking**, **Spectral Projected Gradient
> (SPG)**, and **batched parallel multi-start SPG**.
>
> This is a self-contained, code-side exposition; it points to the
> implementation in `pga_toolbox/`. Each method is a standard algorithm
> from the optimisation literature (cited inline); the value of the toolbox
> is a single, complex-aware, projection-first implementation shared across
> the author's companion libraries (`gaussian-dag`, `cmi-dag`,
> `fading-dag`, `bussgang-dag`).

## Contents

1. [Problem setting](#1-problem-setting)
2. [The Wirtinger gradient convention](#2-the-wirtinger-gradient-convention)
3. [Euclidean projections](#3-euclidean-projections)
4. [Fixed-step projected gradient ascent](#4-fixed-step-projected-gradient-ascent)
5. [Armijo backtracking line search](#5-armijo-backtracking-line-search)
6. [Spectral Projected Gradient (SPG)](#6-spectral-projected-gradient-spg)
7. [Batched parallel multi-start SPG](#7-batched-parallel-multi-start-spg)
8. [Descent variants](#8-descent-variants)

---

## 1. Problem setting

All drivers address the **constrained smooth maximisation**

$$
\max_{x \in \Omega} \ f(x),
\qquad x \in \mathbb{R}^n \ \text{or}\ \mathbb{C}^n,
$$

where $f$ is a real-valued, differentiable objective supplied as a Python
closure `compute_obj()` returning a scalar tensor, and $\Omega$ is a
**closed convex** feasible set supplied as a projector
$\mathcal{P}_\Omega$. The descent drivers minimise a cost $c(x)$ over the
same kind of $\Omega$.

Two assumptions are shared by every method here:

- **Convex $\Omega$.** The shipped projections (Frobenius ball, total-power
  ball; §3) are projections onto convex sets, so a point on the segment
  between two feasible points is feasible — used by the line searches.
- **Exact, deterministic $f$ and $\nabla f$.** The Armijo and SPG step
  rules compare objective values and gradient differences; minibatch noise
  breaks them. (A stochastic SGD driver is on the roadmap.)

The objective and parameters are decoupled from any model: `pga-toolbox`
never sees the DAG, the MI, or the channel — only the closure and the
parameter list. Code: `pga_toolbox/pga.py`.

---

## 2. The Wirtinger gradient convention

For a **real-valued** $f$ of a complex variable $\Theta \in \mathbb{C}^{p
\times q}$, the steepest-ascent direction in the real-Euclidean metric on
$(\mathrm{Re}\Theta, \mathrm{Im}\Theta)$ is the conjugate
(Wirtinger) cogradient

$$
\nabla_{\Theta^\ast} f
= \left(\frac{\partial f}{\partial \Theta^\ast}\right)^{\!\top}.
$$

PyTorch populates a complex leaf's `.grad` with exactly this quantity for a
real scalar loss (up to the well-known factor of two that the literature
sometimes carries; the drivers **absorb it into the step size** and apply
no correction). Consequently the toolbox treats `tensor.grad` as the
ascent direction directly: real and complex parameters share one code
path. Wherever an inner product of a gradient $g$ and a displacement $d$ is
needed (the line searches), it is the **real-Euclidean** product

$$
\langle g, d\rangle_{\mathbb{R}}
= \mathrm{Re}\,\langle \overline{g}, d\rangle
= \mathrm{Re}\sum_i \overline{g_i}\, d_i,
$$

which reduces to the ordinary dot product for real tensors. Code:
`_wirtinger_real_inner` in `pga_toolbox/line_search.py`.

---

## 3. Euclidean projections

A projection returns the nearest feasible point,

$$
\mathcal{P}_\Omega(\xi) = \arg\min_{\zeta \in \Omega}\ \|\zeta - \xi\|_F^2,
$$

and the toolbox ships the two that recur in power-constrained design, both
in **closed form** (`pga_toolbox/projections.py`):

- **Frobenius ball** $\Omega = \{X : \|X\|_F^2 \le P\}$ —
  uniform rescaling
  $$
  \mathcal{P}_\Omega(A) = A \cdot \min\!\left\{1,\ \sqrt{P}/\|A\|_F\right\}.
  $$
- **Total-power ball** $\Omega = \{(A_m) : \sum_m \|A_m\|_F^2 \le P\}$ —
  a **single common** scale on all blocks,
  $$
  A_m \leftarrow s\, A_m, \quad
  s = \min\!\left\{1,\ \sqrt{P / \textstyle\sum_m \|A_m\|_F^2}\right\}.
  $$

Both sets are convex, so the line-search feasibility argument of §6 holds.
Batch-aware variants (`project_*_batched`) apply the same formula
per leading-batch element, keeping multi-start restarts independent (§7).
A projector may either mutate the parameters in place or return new
tensors; the drivers copy back automatically.

---

## 4. Fixed-step projected gradient ascent

The primitive driver iterates

$$
x_{t+1} = \mathcal{P}_\Omega\!\bigl(x_t + \alpha\,\nabla f(x_t)\bigr),
\qquad t = 0,1,\dots,T-1,
$$

with a constant step $\alpha > 0$. One gradient (one `backward()`) per
iteration; the history records $f(x_t)$ at the **pre-update** iterate.
This matches the legacy `gaussian-dag.pga_ascent` signature so a sister
library can swap in the toolbox without changing callers.

Simple and predictable, but $\alpha$ must be hand-tuned to the problem
scale: too small wastes iterations, too large stalls against the
constraint. The next three methods remove that tuning. Code:
`pga_ascent` in `pga_toolbox/pga.py`.

---

## 5. Armijo backtracking line search

Instead of a fixed $\alpha$, choose the step each iteration by backtracking
until a **sufficient-increase** test passes (Armijo, 1966). At iterate
$x$ with gradient $g = \nabla f(x)$, form the trial point and its
**actual (projected) displacement**

$$
x^+(e) = \mathcal{P}_\Omega(x + e\,g), \qquad \Delta = x^+(e) - x,
$$

and accept $e$ when

$$
f\bigl(x^+(e)\bigr) \ \ge\ f(x) + c\,\langle g, \Delta\rangle_{\mathbb{R}},
\qquad c \in (0,1).
$$

Testing against the **post-projection** displacement $\Delta$ (rather than
the raw step $e\,g$) keeps the condition sound when the trial leaves
$\Omega$ and is projected back. If the test fails, shrink $e \leftarrow
\beta_{\downarrow} e$ and retry; if it passes, take the step.

A **persistent step size** is carried across iterations and grown,
$e \leftarrow \min\{e\cdot\beta_{\uparrow},\ e_{\max}\}$, at the start of
each iteration, so the method adapts to the local scale instead of
re-discovering it from a fixed guess every time. Termination: the outer
cap `max_iter`, a total-`forward_budget` cap, or backtracking collapsing
to $e < e_{\min}$ (read as a stationary point). History is recorded at
**accept events only** — one monotone value per outer iteration. Code:
`pga_ascent_armijo` in `pga_toolbox/line_search.py`.

---

## 6. Spectral Projected Gradient (SPG)

SPG (Birgin, Martínez & Raydan, 2000) keeps the per-iteration cost of
steepest ascent but exploits curvature like a quasi-Newton method, via
three ingredients.

**(a) Barzilai–Borwein spectral step.** From successive iterates and
gradients,

$$
s = x_k - x_{k-1}, \qquad y = g_k - g_{k-1},
\qquad
\alpha_{\mathrm{BB}} = \frac{\langle s, s\rangle}{\langle s, y\rangle},
$$

clamped to $[\alpha_{\min}, \alpha_{\max}]$. This encodes the local Hessian
as a single scalar $\tfrac{1}{\alpha}I$ — an $O(n)$ curvature estimate —
written here in the minimisation convention ($g$ the gradient of the cost;
the ascent driver applies it to $-f$, §8).

**(b) Projected-gradient direction along a feasible segment.** With the
spectral step,

$$
d = \mathcal{P}_\Omega(x - \alpha_{\mathrm{BB}}\,g) - x,
\qquad x \leftarrow x + \lambda\, d,\ \ \lambda \in (0,1].
$$

Because $\Omega$ is **convex**, the whole segment $x + \lambda d$ is
feasible, so no re-projection is needed inside the line search.

**(c) Nonmonotone (Grippo–Lampariello–Lucidi) line search.** A step is
accepted when the objective improves relative to the **best of the last
`nm_window` accepted values**, not the immediately previous one:

$$
\phi(x + \lambda d) \ \le\ \max_{0 \le j < W} \phi_{k-j} \ +\ \gamma\,\lambda\,\langle g, d\rangle_{\mathbb{R}}.
$$

Strict monotonicity would suppress the very oscillation that lets the BB
step traverse ill-conditioned valleys; the nonmonotone window allows it.

Because iterates are nonmonotone, **the last point need not be the best**.
The driver tracks the best-seen iterate and copies it back into `params`
on return, so on exit `params` holds the incumbent optimum and the history
extremum equals the objective there. On the originating MI-maximisation
benchmark SPG reached the same optimum as Armijo with $\sim 6\times$ fewer
objective evaluations (and $\sim 20\times$ fewer than a tuned fixed step).
Code: `pga_ascent_spg` in `pga_toolbox/spg.py`.

---

## 7. Batched parallel multi-start SPG

Non-concave $f$ has multiple local optima; the remedy is **multi-start**.
The batched driver runs $B$ independent SPG instances — one per random
initial point — as a single vectorised computation over a leading
**batch dimension** $B$:

- `params[m]` has shape $(B, *\text{shape}_m)$; element $b$ owns
  `params[m][b]`, and elements are **independent** in the closure. That
  independence makes one backward pass suffice: since
  $\sum_b f_b(x_b)$ has $\partial/\partial x_b = \nabla f_b(x_b)$,
  `compute_obj().sum().backward()` hands each element its own gradient.
- `compute_obj()` returns a real vector of shape $(B,)$.
- Every per-iteration scalar of §6 — the BB step $\alpha$, objective
  $\phi$, accept mask, and backtracking $\lambda$ — becomes a $(B,)$
  tensor; the line search backtracks **per element** under a mask.
- An element that reaches a stationary point is **retired** (frozen at its
  best) and carried forward, so the recorded history is a clean $(T,B)$
  best-so-far grid.
- Per-element best-point copy-back: on return `params[m][b]` is element
  $b$'s best iterate. The global solution is $\arg\max_b$ of the final
  best objectives — **best-of-$B$**.

On SIMD/GPU hardware $B$ restarts cost $\approx$ the wall-clock of one, so
global search is nearly free. Two requirements: the projector must be
**batch-aware** (`project_*_batched`, so restarts stay decoupled), and the
closure must be **NaN-safe** — a batched `cholesky`/`logdet` raises if
*any* element is infeasible, so a bad element must yield `NaN` (e.g. via
`cholesky_ex` + `where`, or jitter) rather than throw; the line search
treats non-finite values as rejects. Code: `pga_ascent_spg_batched` in
`pga_toolbox/spg_batched.py`.

---

## 8. Descent variants

Every ascent driver has a `*_descent` twin that **minimises** a cost
$c(x)$. The relationship is exact negation, $\min c = -\max(-c)$: the
driver differentiates $-c$ so that `.grad` holds the descent direction,
steps along it, and records the (positive, descending) cost in the
history. SPG/Armijo curvature and acceptance logic carry over unchanged
under this sign flip. Drivers: `pga_descent`, `pga_descent_armijo`,
`pga_descent_spg`, `pga_descent_spg_batched`.

---

## Notation summary

| Symbol | Meaning |
| --- | --- |
| $f$, $c$ | objective to maximise / cost to minimise (closure) |
| $x \in \mathbb{R}^n$ or $\mathbb{C}^n$ | optimisation parameters (`params`) |
| $\Omega$, $\mathcal{P}_\Omega$ | convex feasible set and its Euclidean projection |
| $g = \nabla f(x)$ | gradient; for complex $x$, the Wirtinger ascent direction (`.grad`) |
| $\langle g, d\rangle_{\mathbb{R}}$ | real-Euclidean inner product $\mathrm{Re}\langle\overline g, d\rangle$ |
| $\alpha$ | step size (fixed) / spectral BB step $\alpha_{\mathrm{BB}}$ (SPG) |
| $e$, $c$, $\beta_\uparrow,\beta_\downarrow$ | Armijo step, sufficient-increase constant, grow/shrink factors |
| $s, y$ | iterate / gradient differences for the BB step |
| $W$ = `nm_window` | nonmonotone window length (SPG) |
| $B$ | number of parallel multi-start restarts (batched SPG) |
| $P$ | power budget for the Frobenius / total-power projections |

For usage and runnable examples see [`README.md`](README.md) and
[`examples/`](examples/); for the companion model libraries that supply the
objective closures, see `gaussian-dag`, `cmi-dag`, `fading-dag`, and
`bussgang-dag`.
