import numpy as np

from tqjcp.problems import make_problem_2d
from tqjcp.solver2d import solve_2d


def test_mixed_order_line_source_smoke():
    problem = make_problem_2d("line_source", nx=5, final_time=0.02)
    baseline = solve_2d(
        problem, method="uc", uncollided_angle_order=4, collided_angle_order=2,
        compression="none", max_source_iters=8, tol=1e-5,
        final_time=0.02, cfl=1.0, save_angular=False,
    )
    compressed = solve_2d(
        problem, method="uc", uncollided_angle_order=4, collided_angle_order=2,
        compression="moment", bits=5, preserve="phi_current",
        positivity_limiter="repair", max_source_iters=8, tol=1e-5,
        final_time=0.02, cfl=1.0, save_angular=False,
    )
    assert baseline.phi.shape == (5, 5)
    assert np.isfinite(baseline.phi).all()
    assert np.isfinite(compressed.phi).all()
    assert compressed.n_uncollided_angles == 16
    assert compressed.n_collided_angles == 4
    assert compressed.transfer_max_moment_error < 1e-10
    assert compressed.quantization.max_moment_error < 1e-10


def test_equal_orders_match_standard_path():
    problem = make_problem_2d("line_source", nx=4, final_time=0.01)
    a = solve_2d(problem, method="uc", angle_order=4, compression="none",
                 max_source_iters=5, tol=1e-5, final_time=0.01, cfl=1.0)
    b = solve_2d(problem, method="uc", angle_order=4,
                 uncollided_angle_order=4, collided_angle_order=4,
                 compression="none", max_source_iters=5, tol=1e-5,
                 final_time=0.01, cfl=1.0)
    assert np.allclose(a.phi, b.phi, rtol=0.0, atol=0.0)
