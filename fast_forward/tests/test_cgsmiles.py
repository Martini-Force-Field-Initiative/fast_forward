# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the cgsmiles module.
"""
import pysmiles
import pytest

from fast_forward.cgsmiles import _resolve_cg_match

# methyl-capped PEO: CH3ter-(EO)n-CH3ter, residue level on top
PEO_3LEVEL = ("{{[#CH3ter][#EO]|{n}[#CH3ter]}}."
              "{{#EO=[$][#EC][$],#CH3ter=[$][#C]}}."
              "{{#EC=[$]COC[$],#C=[$]C}}")


def _capped_peo(n):
    """All-atom graph of C(COC)nC with explicit hydrogens."""
    return pysmiles.read_smiles("C" + "COC" * n + "C", explicit_hydrogen=True)


@pytest.mark.parametrize('n', [1, 3, 10])
def test_resolve_three_level_residues(n):
    """
    A cgsmiles string with a residue level on top resolves, and every CG bead
    gets the resname and resid of the residue it belongs to.
    """
    resolved = _resolve_cg_match(PEO_3LEVEL.format(n=n), _capped_peo(n))
    assert resolved is not None
    cg, _, match = resolved
    assert match

    beads = sorted(cg.nodes)
    resnames = [cg.nodes[bead]["resname"] for bead in beads]
    resids = [cg.nodes[bead]["resid"] for bead in beads]
    assert resnames == ["CH3ter"] + ["EO"] * n + ["CH3ter"]
    assert resids == list(range(1, n + 3))


def test_resolve_two_level_unchanged():
    """
    Strings without a residue level do not go through residue annotation.
    """
    cgs_str = "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}"
    cg, _, match = _resolve_cg_match(cgs_str, _capped_peo(3))
    assert match
    assert all("resname" not in cg.nodes[bead] for bead in cg.nodes)
