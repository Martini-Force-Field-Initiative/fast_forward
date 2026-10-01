"""Improper dihedrals: the Gaussian fit has to find the peak wherever it
sits in [-180, 180), including at the +-180 wrap."""
import numpy as np
import pytest

from fast_forward.interaction_fit import InteractionFitter

TEMPERATURE = 298.15
KB = 0.0083144626  # kJ/(mol K)


def _distribution(center_deg, sigma_deg):
    """Like interaction_distribution's: 360 one-degree bins over [-180, 180)."""
    x = np.arange(-180, 180) + 0.5
    delta = (x - center_deg + 180) % 360 - 180  # periodic distance to the center
    y = np.exp(-0.5 * (delta / sigma_deg) ** 2)
    return np.column_stack([x, y / y.sum()])


@pytest.mark.parametrize("center", [0.0, 180.0, -179.0, 90.0, -120.0, 165.0])
def test_improper_center_and_force_constant(center):
    sigma = 7.7
    fitter = InteractionFitter(precision=3, temperature=TEMPERATURE, constraint_converter=10000,
                               max_dihedrals=5, dihedral_scaling=1000)
    fitter._dihedrals_fitter(_distribution(center, sigma), "hinge")
    fitted = fitter.fit_parameters["dihedrals"]["hinge"]
    assert isinstance(fitted, list), "a narrow distribution should be fitted as an improper"
    fitted_center, k = fitted
    off = (np.degrees(fitted_center) - center + 180) % 360 - 180
    assert abs(off) < 1.0, f"center {np.degrees(fitted_center):.1f} for a peak at {center}"
    assert k == pytest.approx(KB * TEMPERATURE / np.radians(sigma) ** 2, rel=0.05)
