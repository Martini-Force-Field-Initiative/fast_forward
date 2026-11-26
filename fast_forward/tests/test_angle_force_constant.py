"""Angles are written as type 1 or type 10, both of which reduce to
1/2 k (theta - theta0)^2 near theta0, so the Boltzmann-inverted force
constant must be k = RT / sigma^2 independent of theta0."""
import numpy as np
import pytest
from fast_forward.interaction_fit import InteractionFitter


@pytest.mark.parametrize("theta0", [100, 120, 150, 165])
@pytest.mark.parametrize("sigma_deg", [5, 10])
def test_angle_force_constant_has_no_sin_term(theta0, sigma_deg):
    fitter = InteractionFitter(precision=3, temperature=300, constraint_converter=20000,
                               max_dihedrals=1, dihedral_scaling=1)
    x = np.arange(0, 181, 0.5)
    y = np.exp(-0.5 * ((x - theta0) / sigma_deg) ** 2)
    y /= y.sum()
    fitter._angles_fitter(np.column_stack([x, y]), "A_B_C")
    center, k = fitter.fit_parameters['angles']["A_B_C"]
    expected = fitter.kb * fitter.temperature / np.deg2rad(sigma_deg) ** 2
    assert center == pytest.approx(theta0, abs=0.1)
    assert k == pytest.approx(expected, rel=0.02)
