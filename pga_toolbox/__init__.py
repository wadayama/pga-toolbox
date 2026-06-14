"""pga-toolbox: projected gradient ascent / descent for complex
(Wirtinger) and real parameters, with fixed-step, Armijo backtracking
line search, and Spectral Projected Gradient (SPG) variants.

Public API:
    Fixed-step:
        pga_ascent, pga_descent
    Armijo line search (persistent step):
        pga_ascent_armijo, pga_descent_armijo
    Spectral Projected Gradient (Barzilai-Borwein + nonmonotone search):
        pga_ascent_spg, pga_descent_spg
    Closed-form projections:
        project_frobenius_ball, project_total_power

Typical usage:
    >>> from pga_toolbox import pga_ascent_armijo, project_total_power
    >>>
    >>> def closure():
    ...     return my_mi_objective(F_list)
    >>>
    >>> def projector(params):
    ...     return project_total_power(params, P=36.0)
    >>>
    >>> history = pga_ascent_armijo(closure, F_list, projector=projector)

For complex parameters, PyTorch's ``.grad`` on a complex leaf with a
real-valued objective is the natural Wirtinger gradient (the
real-Euclidean steepest direction on the 2n-real lift); both the
fixed-step and Armijo drivers treat it as such.
"""

from .line_search import pga_ascent_armijo, pga_descent_armijo
from .pga import pga_ascent, pga_descent
from .projections import project_frobenius_ball, project_total_power
from .spg import pga_ascent_spg, pga_descent_spg

__version__ = "0.3.0"

__all__ = [
    "pga_ascent",
    "pga_descent",
    "pga_ascent_armijo",
    "pga_descent_armijo",
    "pga_ascent_spg",
    "pga_descent_spg",
    "project_frobenius_ball",
    "project_total_power",
    "__version__",
]
