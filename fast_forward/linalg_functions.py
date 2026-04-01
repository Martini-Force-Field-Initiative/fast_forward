# Part of the Source Code adopted from buildH by Patrick Fuchs under BSD3 LICENSCE
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
Collection of linear algebra functions mostly related to
geometry operations.
"""
import numpy as np
from numba import jit, njit

def _u_vect(vect):
    """
    Compute unit vector of vector.

    Parameters
    ----------
    vect: :class:`~numpy.ndarray`

    Returns
    -----------
    :class:`~numpy.ndarray`
    """
    unit_vect = vect/np.linalg.norm(vect)
    return unit_vect

# this is the numba implementation
u_vect = njit(_u_vect)

def _vector_angle(v1, v2):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    v1: :class:`~numpy.ndarray`
    v2: :class:`~numpy.ndarray`

    Returns
    ---------
    float
    """
    dot = np.dot(u_vect(v1), u_vect(v2))
    if dot > 1.0 and dot < 1.001:
        dot = 1.0
    if dot < -1.0 and dot > -1.001:
        dot = -1.0
    angle = np.arccos(dot)
    return angle

# this is the numba implementation
vector_angle = njit(_vector_angle)

def _vector_angle_degrees(v1, v2):
    """
    Compute the angle between two vectors
    in degrees and between 0 180 degrees.

    Parameters
    ----------
    v1: :class:`~numpy.ndarray`
    v2: :class:`~numpy.ndarray`

    Returns
    ---------
    float
    """
    angle = 180.0/np.pi * vector_angle(v1, v2)
    return angle

# this is the numba implementation
vector_angle_degrees = njit(_vector_angle_degrees)

@njit
def vec2quaternion(vec, theta):
    """
    Translate a vector of 3 elements and angle theta to a quaternion.

    Parameters
    ----------
    vec : :class:`~numpy.ndarray`
        Vector of the quaternion.
    theta : float
        Angle of the quaternion in radian.

    Returns
    -------
    :class:`~numpy.ndarray`
        The full quaternion (4 elements).
    """
    w = np.cos(theta/2)
    x, y, z = np.sin(theta/2) * u_vect(vec)
    return np.array([w, x, y, z])


@njit
def calc_rotation_matrix(quaternion):
    """
    Translate a quaternion to a rotation matrix.

    Parameters
    ----------
    quaternion : :class:`~numpy.ndarray`
        numpy 1D-array of 4 elements.

    Returns
    -------
    :class:`~numpy.ndarray`
        numpy 2D-array (dimension `[3, 3]`)
        The rotation matrix.
    """
    # Initialize rotation matrix.
    matrix = np.zeros((3, 3), dtype=np.float32)
    # Get quaternion elements.
    w, x, y, z = quaternion
    # Compute rotation matrix.
    matrix[0, 0] = w**2 + x**2 - y**2 - z**2
    matrix[1, 0] = 2 * (x*y + w*z)
    matrix[2, 0] = 2 * (x*z - w*y)
    matrix[0, 1] = 2 * (x*y - w*z)
    matrix[1, 1] = w**2 - x**2 + y**2 - z**2
    matrix[2, 1] = 2 * (y*z + w*x)
    matrix[0, 2] = 2 * (x*z + w*y)
    matrix[1, 2] = 2 * (y*z - w*x)
    matrix[2, 2] = w**2 - x**2 - y**2 + z**2
    return matrix


@njit
def apply_rotation(vec_to_rotate, rotation_axis, rad_angle):
    """Rotate a vector around an axis by a given angle.
    Note
    ----
    The rotation axis is a vector of 3 elements.
    Parameters
    ----------
    vec_to_rotate : :class:`~numpy.ndarray`
    rotation_axis : :class:`~numpy.ndarray`
    rad_angle : float
    Returns
    -------
    :class:`~numpy.ndarray`
        1D array
        The final rotated (normalized) vector.
    """
    # Generate a quaternion of the given angle (in radian).
    quaternion = vec2quaternion(rotation_axis, rad_angle)
    # Generate the rotation matrix.
    rotation_matrix = calc_rotation_matrix(quaternion)
    # Apply the rotation matrix on the vector to rotate.
    vec_rotated = np.dot(rotation_matrix, vec_to_rotate)
    return u_vect(vec_rotated)


@njit
def cross_product(A, B):
    """
    Return the cross product between vectors A & B.
    Source: http://hyperphysics.phy-astr.gsu.edu/hbase/vvec.html.

    Note
    ----
    On small vectors (i.e. of 3 elements), computing cross products
    with this functions is faster than `np.cross()`.

    Parameters
    ----------
    A : :class:`~numpy.ndarray`
        A vector of 3 elements.
    B : :class:`~numpy.ndarray`
        Another vector of 3 elements.

    Returns
    -------
    :class:`~numpy.ndarray`
        Cross product of A^B.
    """
    x = (A[1]*B[2]) - (A[2]*B[1])
    y = (A[0]*B[2]) - (A[2]*B[0])
    z = (A[0]*B[1]) - (A[1]*B[0])
    return np.array((x, -y, z))


def _dih(r1, r2, r3):
    cross1 = cross_product(r1, r2)
    cross2 = cross_product(r2, r3)
    n1 = u_vect(cross1)
    n2 = u_vect(cross2)
    dih = vector_angle_degrees(n1, n2)
    # GROMACS specific definition of the sign of the
    # dihedral.
    if np.dot(r1, cross2) < 0:
        dih = - dih
    return dih

dihedral_angle = jit(_dih, nopython=True)
