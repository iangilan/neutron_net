"""VQ linear kinetic transport solver, version 9."""

from .problems import Problem1D, Problem2D, make_problem_1d, make_problem_2d
from .solver1d import Result1D, solve_1d
from .solver2d import Result2D, solve_2d
from .transfer import transfer_angular_field

__version__ = "9.0.0"

__all__ = [
    "Problem1D", "Problem2D", "make_problem_1d", "make_problem_2d",
    "Result1D", "Result2D", "solve_1d", "solve_2d",
    "transfer_angular_field",
]
