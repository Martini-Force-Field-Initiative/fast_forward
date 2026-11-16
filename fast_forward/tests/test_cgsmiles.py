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
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pysmiles
import pytest
from cgsmiles.resolve import MoleculeResolver

from fast_forward.cgsmiles import (_annotate_vs, _collect_bead_data, _copy_offsets,
                                   _first_atom_by_molnum, _resolve_cg_match,
                                   cgsmiles_to_mapping)
from fast_forward.universe_handler import ResidueIter

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


def _bead_atoms(cg, match):
    """Per bead (in node order): fragname and the sorted matched atom indices."""
    return [(cg.nodes[bead]["fragname"],
             sorted(match[atom] for atom in cg.nodes[bead]["graph"].nodes))
            for bead in cg.nodes]


def test_bead_index_reorders_beads():
    """
    The 'index' bead annotation sets the bead order; bonds and the mapped
    atoms follow their beads.
    """
    plain = "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}"
    # reverse the order: last bead first
    indexed = ("{[#C;index=5][#EC;index=4][#EC;index=3][#EC;index=2][#C;index=1]}."
               "{#EC=[$]COC[$],#C=[$]C}")
    cg_ref, _, match_ref = _resolve_cg_match(plain, _capped_peo(3))
    cg, _, match = _resolve_cg_match(indexed, _capped_peo(3))

    assert list(cg.nodes) == [0, 1, 2, 3, 4]
    assert _bead_atoms(cg, match) == _bead_atoms(cg_ref, match_ref)[::-1]
    assert sorted(tuple(sorted(e)) for e in cg.edges) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_bead_index_ties_keep_string_order():
    """Equal indices keep the string order, so repeated beads can share one."""
    cgs_str = "{[#C;index=1][#EC;index=0]|3[#C;index=1]}.{#EC=[$]COC[$],#C=[$]C}"
    cg_ref, _, match_ref = _resolve_cg_match(
        "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}", _capped_peo(3))
    cg, _, match = _resolve_cg_match(cgs_str, _capped_peo(3))
    ref = _bead_atoms(cg_ref, match_ref)
    # EC beads first in chain order, then both caps in chain order
    assert _bead_atoms(cg, match) == ref[1:4] + [ref[0], ref[4]]


def test_bead_index_negative_counts_from_end():
    """
    Negative indices come after all non-negative ones, -1 last; beads
    without a reason to move can then keep a shared index of 0.
    """
    cgs_str = ("{[#C;index=-1][#EC;index=0][#EC;index=-2][#EC;index=0][#C;index=1]}."
               "{#EC=[$]COC[$],#C=[$]C}")
    cg_ref, _, match_ref = _resolve_cg_match(
        "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}", _capped_peo(3))
    cg, _, match = _resolve_cg_match(cgs_str, _capped_peo(3))
    ref = _bead_atoms(cg_ref, match_ref)
    # 0, 0, 1, -2, -1
    assert _bead_atoms(cg, match) == [ref[1], ref[3], ref[4], ref[2], ref[0]]


def test_bead_index_backbone_first():
    """Comb polymer: all backbone beads, then all side-chain beads."""
    cgs_str = ("{[#B;index=0]([#S;index=1])[#B;index=0]([#S;index=1])"
               "[#B;index=0]([#S;index=1])}.{#B=[$]CC[$][$],#S=[$]C}")
    # take the all-atom molecule the string itself describes
    mol = MoleculeResolver.from_string(cgs_str, last_all_atom=True).resolve()[1]
    cg, _, _ = _resolve_cg_match(cgs_str, mol)
    assert [cg.nodes[b]["fragname"] for b in cg.nodes] == ["B"] * 3 + ["S"] * 3
    assert sorted(tuple(sorted(e)) for e in cg.edges) == [
        (0, 1), (0, 3), (1, 2), (1, 4), (2, 5)]


def test_bead_index_sorts_within_residues():
    """
    With a residue level, beads are sorted within each residue and residues
    keep their sequence order, even when a fragment is repeated.
    """
    cgs_str = ("{[#CAP][#MON]|2[#CAP]}."
               "{#CAP=[$][#C],#MON=[$][#B;index=1]([#S;index=0])[$]}."
               "{#B=[$]CC[$][$],#S=[$]C,#C=[$]C}")
    # C-[CC(C)]-[CC(C)]-C
    mol = pysmiles.read_smiles("CCC(C)CC(C)C", explicit_hydrogen=True)
    resolved = _resolve_cg_match(cgs_str.replace("[#C]", "[#C;index=0]"), mol)
    assert resolved is not None
    cg, _, _ = resolved
    beads = [(cg.nodes[b]["resid"], cg.nodes[b]["fragname"]) for b in cg.nodes]
    assert beads == [(1, "C"), (2, "S"), (2, "B"), (3, "S"), (3, "B"), (4, "C")]


def test_bead_index_partial_defaults_to_zero(capsys):
    """Beads without an index sort as index 0, with a warning."""
    cgs_str = "{[#C;index=-1][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}"
    cg_ref, _, match_ref = _resolve_cg_match(
        "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}", _capped_peo(3))
    capsys.readouterr()
    cg, _, match = _resolve_cg_match(cgs_str, _capped_peo(3))
    ref = _bead_atoms(cg_ref, match_ref)
    assert _bead_atoms(cg, match) == ref[1:] + [ref[0]]
    out = capsys.readouterr().out
    assert "WARNING" in out and "4 of 5 beads" in out


def test_bead_index_full_no_warning(capsys):
    cgs_str = "{[#C;index=1][#EC;index=0][#C;index=1]}.{#EC=[$]COC[$],#C=[$]C}"
    _resolve_cg_match(cgs_str, _capped_peo(1))
    assert "WARNING" not in capsys.readouterr().out


def test_bead_index_not_integer_raises():
    cgs_str = "{[#C;index=a][#EC;index=2][#C;index=3]}.{#EC=[$]COC[$],#C=[$]C}"
    with pytest.raises(ValueError, match="not an integer"):
        _resolve_cg_match(cgs_str, _capped_peo(1))


def test_bead_index_not_checked_without_match():
    """Invalid indices on a non-matching candidate string are ignored."""
    cgs_str = "{[#C;index=a][#EC][#C;index=3]}.{#EC=[$]COC[$],#C=[$]C}"
    assert _resolve_cg_match(cgs_str, _capped_peo(2)) is None


@pytest.mark.parametrize('cgs_str, resids', [
    # 2-level: every bead has resid 1, one residue per copy
    ("{[#C][#EC][#C]}.{#EC=[$]COC[$],#C=[$]C}", [1]),
    # 3-level: first and last residue differ
    (PEO_3LEVEL.format(n=1), [1, 2, 3]),
])
def test_collect_bead_data_residue_per_copy(cgs_str, resids):
    """Each molecule copy starts its own residues, even if resids repeat."""
    mol = _capped_peo(1)
    cg, _, match = _resolve_cg_match(cgs_str, mol)
    for bead in cg.nodes:  # set by get_mappings for 2-level strings
        cg.nodes[bead].setdefault("resid", 1)
        cg.nodes[bead].setdefault("resname", "MOL")
    res_iter = ResidueIter()
    offsets = [0, len(mol), 2 * len(mol)]
    _, _, _, n_beads, res_iter = _collect_bead_data(cg, match, offsets, 0, res_iter)
    assert n_beads == 3 * len(cg)
    assert [res.resid for res in res_iter.res_iter()] == resids * 3


class _Atoms(list):
    """List of atoms with a molnums array, like an MDAnalysis AtomGroup."""
    def __init__(self, atoms, molnums):
        super().__init__(atoms)
        self.molnums = np.array(molnums)


def _fake_universe(layout):
    """
    Minimal stand-in for UniverseHandler with molecules in topology order.

    layout: list of (molname, all-atom graph). As in UniverseHandler, each
    molecule graph carries the global atom indices of the first copy.
    """
    atoms, molnums, mol_idxs_by_name, molecule_graphs = [], [], {}, {}
    for molnum, (mol_name, graph) in enumerate(layout):
        start = len(atoms)
        mol_idxs_by_name.setdefault(mol_name, []).append(molnum)
        if mol_name not in molecule_graphs:
            molecule_graphs[mol_name] = nx.convert_node_labels_to_integers(
                graph, first_label=start)
        atoms.extend(SimpleNamespace(name=f"{mol_name}{molnum}_{i}") for i in range(len(graph)))
        molnums.extend([molnum] * len(graph))
    return SimpleNamespace(molecule_graphs=molecule_graphs,
                           mol_idxs_by_name=mol_idxs_by_name,
                           atoms=_Atoms(atoms, molnums))


ETH_CGS = "{[#C][#EC][#C]}.{#EC=[$]COC[$],#C=[$]C}"
PEO2_CGS = "{[#C][#EC]|2[#C]}.{#EC=[$]COC[$],#C=[$]C}"


def test_copy_offsets_follow_first_atom():
    univ = _fake_universe([("A", _capped_peo(1)), ("B", _capped_peo(2)),
                           ("A", _capped_peo(1))])
    first_atom = _first_atom_by_molnum(univ)
    n_a, n_b = len(_capped_peo(1)), len(_capped_peo(2))
    assert first_atom == {0: 0, 1: n_a, 2: n_a + n_b}
    assert _copy_offsets(first_atom, [0, 2]) == [0, n_a + n_b]
    assert _copy_offsets(first_atom, [1]) == [0]


def test_interleaved_molecules_map_their_own_atoms():
    """
    Copies of a molecule type that are not next to each other in the
    topology (ETH PEO ETH) map onto their own atoms.
    """
    layout = [("ETH", _capped_peo(1)), ("PEO", _capped_peo(2)), ("ETH", _capped_peo(1))]
    univ = _fake_universe(layout)
    mapped_atoms, bead_idxs, _, _, res_iter = cgsmiles_to_mapping(
        univ, [ETH_CGS, PEO2_CGS], ["ETH", "PEO"])

    # atoms of each molecule in the topology
    molnums = univ.atoms.molnums
    atoms_of = [set(np.flatnonzero(molnums == m).tolist()) for m in range(len(layout))]
    # beads come per molecule type: ETH copy 1, ETH copy 2, then PEO
    beads_per_copy = [list(mapped_atoms[0:3]), list(mapped_atoms[3:6]), list(mapped_atoms[6:10])]
    for molnum, beads in zip([0, 2, 1], beads_per_copy):
        mapped = set().union(*(set(bead) for bead in beads))
        assert mapped == atoms_of[molnum]
    assert len(bead_idxs) == 10


def test_two_level_resname_is_molecule_name():
    """
    Without a residue level, each molecule type gets its own resname, so
    mappings of different molecules are not merged.
    """
    univ = _fake_universe([("ETH", _capped_peo(1)), ("ETH", _capped_peo(1)),
                           ("PEO", _capped_peo(2))])
    cgs_strs = ["{[#C][#EC][#C]}.{#EC=[$]COC[$],#C=[$]C}",
                "{[#C][#EC]|2[#C]}.{#EC=[$]COC[$],#C=[$]C}"]
    _, bead_idxs, mappings, _, res_iter = cgsmiles_to_mapping(
        univ, cgs_strs, ["ETH", "PEO"])

    assert set(mappings) == {"ETH", "PEO"}
    assert len(mappings["ETH"].beads) == 3
    assert len(mappings["PEO"].beads) == 4
    assert [res.resname for res in res_iter.res_iter()] == ["ETH", "ETH", "PEO"]
    assert len(bead_idxs) == 2 * 3 + 4


def test_vs_inherits_weights():
    """A virtual site maps onto its neighbor's atoms, with their weights."""
    cgs_str = "{[#TC4].[#SP4][#SP4]}.{#SP4=OC[$]C[$][O;w=0.5]}"
    mol = MoleculeResolver.from_string(cgs_str, last_all_atom=True).resolve()[1]
    cg, _, _ = _resolve_cg_match(cgs_str, mol)
    vs = next(b for b in cg.nodes if cg.nodes[b]["fragname"] == "TC4")
    neigh, = cg.neighbors(vs)
    vs_weights = dict(cg.nodes[vs]["graph"].nodes(data="weight"))
    neigh_weights = {atom: data.get("weight", 1.0)
                     for atom, data in cg.nodes[neigh]["graph"].nodes(data=True)}
    assert vs_weights == neigh_weights
    assert 0.5 in vs_weights.values()
    # other atom attributes are kept too
    assert all("element" in data for _, data in cg.nodes[vs]["graph"].nodes(data=True))


def test_vs_sums_weights_of_shared_atoms():
    """An atom split between two neighbors counts with its total weight."""
    bead_a, bead_b = nx.Graph(), nx.Graph()
    bead_a.add_nodes_from([(0, {"element": "C"}), (1, {"element": "C", "weight": 0.5})])
    bead_b.add_nodes_from([(1, {"element": "C", "weight": 0.5}), (2, {"element": "C"})])
    cg = nx.Graph()
    cg.add_node(0, graph=bead_a)
    cg.add_node(1, graph=bead_b)
    cg.add_node(2, graph=nx.Graph())  # virtual site
    cg.add_edges_from([(0, 1), (0, 2), (1, 2)])
    _annotate_vs(cg)
    assert dict(cg.nodes[2]["graph"].nodes(data="weight")) == {0: 1.0, 1: 1.0, 2: 1.0}
    # the neighbors' own weights are untouched
    assert bead_a.nodes[1]["weight"] == 0.5 and "weight" not in bead_a.nodes[0]


def _vs_and_parent_weights(cgs_str):
    """Resolve a string with one VS 'U'; return {element: weights} for VS and parent."""
    mol = MoleculeResolver.from_string(cgs_str, last_all_atom=True).resolve()[1]
    cg, _, _ = _resolve_cg_match(cgs_str, mol)
    vs = next(b for b in cg.nodes if cg.nodes[b]["fragname"] == "U")
    parent, = cg.neighbors(vs)

    def by_element(bead_graph):
        out = {}
        for atom, data in sorted(bead_graph.nodes(data=True)):
            out.setdefault(data["element"], []).append(float(data.get("weight", 1.0)))
        return out
    return by_element(cg.nodes[vs]["graph"]), by_element(cg.nodes[parent]["graph"])


def test_wvs_heavy_atom_and_inherited_by_hydrogens():
    """
    wVS sets the weight for the VS only; hydrogens inherit it like w,
    and atoms without wVS keep their normal weight.
    """
    vs, parent = _vs_and_parent_weights(r"{[#C].[#U]}.{#C=N[C;w=1;wVS=0]}")
    # the parent bead is unchanged
    assert parent == {"N": [1.0], "C": [1.0], "H": [1.0] * 5}
    # N and its 2 H keep 1, C and its 3 H get 0 -> VS on the NH2 group
    assert vs == {"N": [1.0], "C": [0.0], "H": [1.0, 1.0, 0.0, 0.0, 0.0]}


def test_wvs_place_vs_on_one_atom():
    vs, _ = _vs_and_parent_weights(r"{[#C].[#U]}.{#C=[N;wVS=0][C;wVS=1]}")
    assert vs == {"N": [0.0], "C": [1.0], "H": [0.0, 0.0, 1.0, 1.0, 1.0]}


def test_wvs_falls_back_to_w_and_explicit_h_wins():
    """Without wVS the VS uses w; an explicit wVS on a hydrogen is kept."""
    vs, _ = _vs_and_parent_weights(r"{[#C].[#U]}.{#C=N[C;w=0.3;wVS=0.5][H;wVS=2]}")
    # N + its H's: w default 1; C + implicit H's: wVS 0.5; explicit H: 2
    assert vs == {"N": [1.0], "C": [0.5], "H": [2.0, 1.0, 1.0, 0.5, 0.5]}


def test_wvs_not_a_number_raises():
    cgs_str = r"{[#C].[#U]}.{#C=N[C;wVS=x]}"
    mol = MoleculeResolver.from_string(cgs_str, last_all_atom=True).resolve()[1]
    with pytest.raises(ValueError, match="not a number"):
        _resolve_cg_match(cgs_str, mol)


def test_collect_bead_data_rejects_split_residues():
    """Beads of one residue that are not contiguous are rejected."""
    cg, _, match = _resolve_cg_match(PEO_3LEVEL.format(n=1), _capped_peo(1))
    cg.nodes[2]["resid"] = cg.nodes[0]["resid"]
    with pytest.raises(ValueError, match="not contiguous"):
        _collect_bead_data(cg, match, [0], 0, ResidueIter())


def test_resolve_two_level_unchanged():
    """
    Strings without a residue level do not go through residue annotation.
    """
    cgs_str = "{[#C][#EC]|3[#C]}.{#EC=[$]COC[$],#C=[$]C}"
    cg, _, match = _resolve_cg_match(cgs_str, _capped_peo(3))
    assert match
    assert all("resname" not in cg.nodes[bead] for bead in cg.nodes)
