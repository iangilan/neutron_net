import numpy as np

from tqjcp.quadrature import upper_hemisphere_product_quadrature
from tqjcp.transfer import angular_moments, transfer_angular_field, transfer_moment_error


def test_transfer_preserves_phi_and_current():
    rng = np.random.default_rng(12)
    q4 = upper_hemisphere_product_quadrature(4)
    q8 = upper_hemisphere_product_quadrature(8)
    source = rng.normal(size=(7, q4.n_angles))
    target = transfer_angular_field(source, q4, q8, preserve="phi_current")
    a = angular_moments(source, q4, "phi_current")
    b = angular_moments(target, q8, "phi_current")
    assert np.max(np.abs(a - b)) < 5.0e-12
    assert transfer_moment_error(source, q4, q8) < 5.0e-12


def test_equal_order_transfer_is_identity():
    rng = np.random.default_rng(3)
    q = upper_hemisphere_product_quadrature(4)
    source = rng.normal(size=(2, 3, q.n_angles))
    target = transfer_angular_field(source, q, q)
    assert np.array_equal(source, target)
