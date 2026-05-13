# Source Code adopted from buildH by Patrick Fuchs under BSD3 LICENSCE
# see below for original copyright notice
#
# BSD 3-Clause License
#
# Copyright (c) 2019, Patrick FUCHS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Module to reconstruct hydrogens from a group of atoms.
Most of this comes from  buildh.readthedocs.io and
is proted to be compatible with the vermouth way of
handling molecules.
"""
import fnmatch
import numpy as np
from fast_forward.linalg_functions import u_vect, apply_rotation, cross_product, vector_angle

# Constants.
# From https://en.wikipedia.org/wiki/Carbon%E2%80%93hydrogen_bond
LENGTH_CH_BOND = 1.09  # in Angst
# From https://en.wikipedia.org/wiki/Tetrahedron, tetrahedral angle equals
# arccos(-1/3) ~ 1.9106 rad ~ 109.47 deg.
TETRAHEDRAL_ANGLE = np.arccos(-1/3)


def get_CH(atom, helper1, helper2, helper3):
    """
    Reconstruct the unique hydrogen of a sp3 carbon.

    Parameters
    ----------
    atom : :class:`~numpy.ndarray`
        Central atom on which we want to reconstruct the hydrogen.
    helper1 : :class:`~numpy.ndarray`
        First neighbor of central atom.
    helper2 : :class:`~numpy.ndarray`
        Second neighbor of central atom.
    helper3 : :class:`~numpy.ndarray`
        Third neighbor of central atom.

    Returns
    -------
    :class:`~numpy.ndarray`
        Coordinates of the rebuilt hydrogen: `([x_H, y_H, z_H])`.
    """
    # Calculate vector along tetrahedron median.
    # !!! Important !!! Use unit vectors (some bonds may have different length).
    v2 = u_vect(helper1 - atom) + u_vect(helper2 - atom) \
         + u_vect(helper3 - atom)
    # CH bond is on the opposite direction.
    unit_vect_H = u_vect(-v2)
    coor_H = LENGTH_CH_BOND * unit_vect_H + atom
    return coor_H


def get_CH2(atom, helper1, helper2):
    """
    Reconstruct the 2 hydrogens of a sp3 carbon (methylene group).

    Parameters
    ----------
    atom : :class:`~numpy.ndarray`
        Central atom on which we want to reconstruct hydrogens.
    helper1 : :class:`~numpy.ndarray`
        Heavy atom before central atom.
    helper2 : :class:`~numpy.ndarray`
        Heavy atom after central atom.

    Returns
    -------
    tuple
        tupple of :class:`~numpy.ndarray`
        Coordinates of the two hydrogens:
        `([x_H1, y_H1, z_H1], [x_H2, y_H2, z_H2])`.
    """
    # atom->helper1 vector.
    v2 = u_vect(helper1 - atom)
    # atom->helper2 vector.
    v3 = u_vect(helper2 - atom)
    # Vector orthogonal to the helpers/atom plane.
    #v4 = normalize(np.cross(v3, v2))
    v4 = u_vect(cross_product(v3, v2))
    # Rotation axis is atom->helper1 vec minus atom->helper2 vec.
    rotation_axis = u_vect(v2 - v3)
    # Vector to be rotated by theta/2, perpendicular to rotation axis and v4.
    vec_to_rotate = u_vect(cross_product(v4, rotation_axis))
    # Reconstruct the two hydrogens.
    unit_vect_H1 = apply_rotation(vec_to_rotate, rotation_axis,
                                      -TETRAHEDRAL_ANGLE/2)

    hcoor_H1 = LENGTH_CH_BOND * unit_vect_H1 + atom

    unit_vect_H2 = apply_rotation(vec_to_rotate, rotation_axis,
                                      TETRAHEDRAL_ANGLE/2)
    hcoor_H2 = LENGTH_CH_BOND * unit_vect_H2 + atom
    return [hcoor_H1, hcoor_H2]


def get_CH3(atom, helper1, helper2):
    """
    Reconstruct the 3 hydrogens of a sp3 carbon (methyl group).

    Parameters
    ----------
    atom : :class:`~numpy.ndarray`
        Central atom on which we want to reconstruct hydrogens.
    helper1 : :class:`~numpy.ndarray`
        Heavy atom before central atom.
    helper2 : :class:`~numpy.ndarray`
        Heavy atom before helper1 (two atoms away from central atom).

    Returns
    -------
    tuple
        tuple of :class:`~numpy.ndarray`
        Coordinates of the 3 hydrogens:
        `([x_H1, y_H1, z_H1], [x_H2, y_H2, z_H2], [x_H3, y_H3, z_H3])`.
    """
    ### Build CH3e.
    theta = TETRAHEDRAL_ANGLE
    # atom->helper1 vector.
    v2 = helper1 - atom
    # atom->helper2 vector.
    v3 = helper2 - atom
    # Rotation axis is perpendicular to the atom/helpers plane.
    #rotation_axis = normalize(np.cross(v3, v2))
    rotation_axis = u_vect(cross_product(v3, v2))
    # Rotate v2 by tetrahedral angle. New He will be in the same plane
    # as atom and helpers.
    unit_vect_He = apply_rotation(v2, rotation_axis, theta)
    coor_He = LENGTH_CH_BOND * unit_vect_He + atom
    ### Build CH3r.
    theta = (2/3) * np.pi
    rotation_axis = u_vect(helper1 - atom)
    v4 = u_vect(coor_He - atom)
    # Now we rotate atom->He bond around atom->helper1 bond by 2pi/3.
    unit_vect_Hr = apply_rotation(v4, rotation_axis, theta)
    coor_Hr = LENGTH_CH_BOND * unit_vect_Hr + atom
    ### Build CH3s.
    theta = -(2/3) * np.pi
    rotation_axis = u_vect(helper1 - atom)
    v5 = u_vect(coor_He - atom)
    # Last we rotate atom->He bond around atom->helper1 bond by -2pi/3.
    unit_vect_Hs = apply_rotation(v5, rotation_axis, theta)
    coor_Hs = LENGTH_CH_BOND * unit_vect_Hs + atom
    return [coor_He, coor_Hr, coor_Hs]


def get_CH_double_bond(atom, helper1, helper2):
    """Reconstruct the hydrogen of a sp2 carbon.
    Parameters
    ----------
    atom : :class:`~numpy.ndarray`
        Central atom on which we want to reconstruct the hydrogen.
    helper1 : :class:`~numpy.ndarray`
        Heavy atom before central atom.
    helper2 : :class:`~numpy.ndarray`
        Heavy atom after central atom.
    Returns
    -------
    tuple
        tuple of :class:`~numpy.ndarray`
        Coordinates of the rebuilt hydrogen: `([x_H, y_H, z_H])`.
    """
    # calc CCC_angle helper1-atom-helper2 (in rad).
    v1 = helper1 - atom
    v2 = helper2 - atom
    CCC_angle = vector_angle(v1, v2)
    # We want to bisect the C-C-C angle ==> we take half of (2pi-CCC_angle).
    # Factorizing yields: pi - CCC_angle/2.
    theta = np.pi - (CCC_angle / 2)
    # atom->helper1 vector.
    v2 = helper1 - atom
    # atom->helper2 vector.
    v3 = helper2 - atom
    # The rotation axis is orthogonal to the atom/helpers plane.
    #rotation_axis = normalize(np.cross(v2, v3))
    rotation_axis = u_vect(cross_product(v2, v3))
    # Reconstruct H by rotating v3 by theta.
    unit_vect_H = apply_rotation(v3, rotation_axis, theta)
    coor_H = LENGTH_CH_BOND * unit_vect_H + atom
    return coor_H

def match_attributes(atom, attributes):
    for attr, value in attributes.items():
        try:
            assert fnmatch.fnmatch(getattr(atom, attr), value)
        except AttributeError:
            return False
    return True

def find_any_bonded_neighbor(atom, bonded_graph, black_list):
    """
    Find any bonded neighbor of atomgroup.

    Parameters
    ----------
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        atom-group of single atom
    attributes: dict
        a dict of attributes the atom has to full-fill
        only string attributes are allowed

    Returns
    ------
    :class:`~MDAnalysis.core.groups.AtomGroup`
        atomgroup of bonded neighbor
    """
    bond_list = bonded_graph.neighbors(atom)
    for atom in bond_list:
        if atom not in black_list:
            return atom
    return None

REF_ATOMS =  {"CH3": [0, 1],
              "CH2": [0, 0],
              "CH1": [0, 0, 0],
              "CHd": [0, 0],
             }

BUILD_HYDRO = {"CH3": get_CH3,
               "CH2": get_CH2,
               "CH1": get_CH,
               "CHd": get_CH_double_bond,
              }

def find_helper_atoms(universe, atom, carbon_type, bonded_graph):
    """
    Given a carbon-type find all helper atoms
    needed in the reconstruction of the coordinates.
    """
    construction_atoms = [atom.index]
    helper_positions = {"atom": atom.position}
    help_keys = ["helper1", "helper2", "helper3"]
    for help_key, help_idx in zip(help_keys, REF_ATOMS[carbon_type]):
        anchor = construction_atoms[help_idx]
        neighbor = find_any_bonded_neighbor(anchor, bonded_graph, construction_atoms)
        construction_atoms.append(neighbor)
        helper_positions[help_key] = universe.atoms[neighbor].position
    return helper_positions
