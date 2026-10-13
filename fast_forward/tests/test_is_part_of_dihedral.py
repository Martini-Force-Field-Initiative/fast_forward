"""An angle is part of a dihedral if it is atoms 0-2 or 1-3 of any dihedral,
in either direction -- not only of the last one in the list."""
import numpy as np
import pytest

from fast_forward.interaction_fit import _is_part_of_dihedral

DIHEDRALS = [np.array([1, 2, 3, 4]),
             np.array([5, 6, 7, 8]),
             np.array([9, 10, 11, 12])]


@pytest.mark.parametrize("angle, expected", [
    ([1, 2, 3], True),     # first dihedral, atoms 0-2
    ([2, 3, 4], True),     # first dihedral, atoms 1-3
    ([6, 7, 8], True),     # middle dihedral
    ([10, 11, 12], True),  # last dihedral
    ([4, 3, 2], True),     # reversed, first dihedral
    ([7, 6, 5], True),     # reversed, middle dihedral
    ([1, 2, 4], False),    # atoms of a dihedral, but not consecutive
    ([3, 4, 5], False),    # spans two dihedrals
    ([20, 21, 22], False), # in no dihedral
])
def test_is_part_of_dihedral(angle, expected):
    assert _is_part_of_dihedral(np.array(angle), DIHEDRALS) is expected


def test_no_dihedrals():
    assert _is_part_of_dihedral(np.array([1, 2, 3]), []) is False
