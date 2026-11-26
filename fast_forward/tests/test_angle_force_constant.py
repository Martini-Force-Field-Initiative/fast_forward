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


def _write_angle(fitter, k, capsys):
    fitter.fit_parameters['angles']["A_B_C"] = [120.0, k]
    fitter.fit_to_gmx('angles', "A_B_C", [[0, 1, 2]], vs_constructors=[])
    return fitter.interactions_dict['angles'][-1].parameters[2], capsys.readouterr().out


@pytest.mark.parametrize("max_fc, k, expected_k, warns", [
    (150, 100, 100, False),    # below cap and warning threshold
    (150, 200, 150, True),     # cap below 250: warn above the cap
    (150, 150, 150, False),    # exactly at a cap below 250
    (150, 400, 150, True),     # capped and above 250
    (1000, 400, 400, True),    # not capped, still above 250
    (1000, 250, 250, False),   # exactly at the threshold
    (600, 700, 600, True),     # default cap
])
def test_angle_cap_and_warning(max_fc, k, expected_k, warns, capsys):
    fitter = InteractionFitter(precision=3, temperature=300, constraint_converter=20000,
                               max_dihedrals=1, dihedral_scaling=1, max_angle_fc=max_fc)
    written_k, out = _write_angle(fitter, k, capsys)
    assert written_k == expected_k
    assert ("WARNING: Angle A_B_C" in out) is warns
    if warns:
        assert f"above {min(250, max_fc):g} kJ/mol/rad^2" in out


def test_default_angle_cap():
    fitter = InteractionFitter(precision=3, temperature=300, constraint_converter=20000,
                               max_dihedrals=1, dihedral_scaling=1)
    assert fitter.max_angle_fc == 600
