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
import numpy as np
import numba
import networkx as nx
import pysmiles
from cgsmiles.resolve import MoleculeResolver
from .map_file_parers import Mapping
from .universe_handler import ResidueIter


def remove_explicit_hydrogens(graph):
    """
    Returns a subgraph of a molecule sans hydrogen atoms and
    annotates the number of hydrogen atoms on each remaining node.
    """
    not_hnodes = [node for node in graph.nodes if graph.nodes[node]["element"] != "H"]
    not_h_graph = graph.subgraph(not_hnodes)
    for node in not_hnodes:
        hcount = sum(
            1 for neighbor in nx.neighbors(graph, node)
            if graph.nodes[neighbor]["element"] == "H"
        )
        not_h_graph.nodes[node]["hcount"] = hcount
    return not_h_graph


def load_cgsmiles_library(filepath):
    """
    Given a file that lists cgsmiles strings, read and return them as a list.

    Parameters
    ----------
    filepath: str

    Returns
    -------
    list[str]
    """
    cgsmiles_strs = []
    with open(filepath) as _file:
        for line in _file:
            cgsmiles_strs.append(line.strip())
    return cgsmiles_strs


def find_one_graph_match(graph1, graph2):
    """
    Returns one ISMAGS match when graphs are isomorphic, otherwise [].
    Matches are done based on element and bond order, which is sufficient
    to also account for charges implicitly via the orders.

    Parameters
    ----------
    graph1: networkx.Graph
    graph2: networkx.Graph

    Returns
    -------
    abc.Iterator
    """

    def node_match(n1, n2):
        for attr in ["element", "hcount"]:
            if n1[attr] != n2[attr]:
                return False
        return True

    if len(graph1) != len(graph2):
        raw_matches = iter([])
    else:
        graph1_no_hatoms = remove_explicit_hydrogens(graph1)
        graph2_no_hatoms = remove_explicit_hydrogens(graph2)
        GM = nx.isomorphism.GraphMatcher(
            graph1_no_hatoms,
            graph2_no_hatoms,
            node_match=node_match,
        )
        raw_matches = GM.subgraph_isomorphisms_iter()

    try:
        mapping = next(raw_matches)
        # Add hydrogen atoms to the mapping
        for node in graph1.nodes:
            if graph1.nodes[node].get("element") == "H" and node not in mapping:
                anchor = list(graph1.neighbors(node))[0]
                hatoms_target = [
                    n for n in graph1.neighbors(anchor)
                    if graph1.nodes[n]["element"] == "H"
                ]
                hatoms_ref = [
                    n for n in graph2.neighbors(mapping[anchor])
                    if graph2.nodes[n]["element"] == "H"
                ]
                # Enforced by graph matching, but assert to catch any zip-masking issues
                assert len(hatoms_target) == len(hatoms_ref)
                mapping.update(dict(zip(hatoms_target, hatoms_ref)))
    except StopIteration:
        mapping = []

    return mapping


def _most_common(_list):
    return max(set(_list), key=_list.count)


def get_mappings(cg, univ, _match, mappings):
    """
    Assigns each coarse CG node a resname and resid based on the all-atom
    universe. If a bead is split between residues, the resname and resid of
    the majority of atoms is used.

    Parameters
    ----------
    cg: networkx.Graph
        The coarse molecule graph.
    univ: :class:`fast_forward.universe_handler.UniverseHandler`
        The universe storing the fine-grained molecule information.
    _match: dict
        A dict mapping the coarse atoms to fine-grained atoms.
    mappings: dict[:class:`fast_forward.map_file_parser.Mapping`]
        A dict of mapping objects.

    Returns
    -------
    dict[:class:`fast_forward.map_file_parser.Mapping`]
        Updated mappings.
    """
    target_resids = {}
    for bead in cg.nodes:
        atoms = cg.nodes[bead]['graph'].nodes
        # if no resname mapping is provided everything is defaulted
        # to RES
        resname_default = "RES"
        # we cannot simply take the resids provided in the input
        # because we allow mapping across residues in which case
        # there is no unique resid and the most common one is not
        # safe enough
        resid_default = 1 
        resname = cg.nodes[bead].get('resname', resname_default)
        resid = cg.nodes[bead].get('resid', resid_default)
        cg.nodes[bead]['resid'] = resid
        cg.nodes[bead]['resname'] = resname
        mapping = mappings.get(resname, Mapping(resname, resname))
        target_resid = target_resids.get(resname, resid)
        target_resids[resname] = target_resid
        if resid != target_resids[resname]:
            continue
        for atom in atoms:
            weight = np.float32(cg.nodes[bead]['graph'].nodes[atom].get('weight', 1))
            mapping.add_atom(
                # we need to append the bead index here because in a cgsmiles string
                # the bead names do not need to be unique but the mapping infrastructure
                # assumes this. Making this more flexible should be a future objective
                cg.nodes[bead]["fragname"]+f"{bead}",
                _match[atom],
                atom=univ.atoms[_match[atom]].name,
                weight=weight,
            )
        mappings[resname] = mapping

    return mappings


def _annotate_vs(cg_graph):
    """
    Identifies virtual sites in the CG graph and annotates them.

    If a node in cg_graph has no mapped atom positions, it is treated as a
    virtual site. Its mapped position is composed as the union of fragment
    graphs of neighboring atoms, which is equivalent to a virtual_sitesn
    mapping.

    Parameters
    ----------
    cg_graph: networkx.Graph
        The graph describing a coarse molecule.
    """
    for node in cg_graph.nodes:
        if len(cg_graph.nodes[node].get('graph', [])) == 0:
            g = nx.Graph()
            for neigh in cg_graph.neighbors(node):
                g.add_nodes_from(list(cg_graph.nodes[neigh]['graph'].nodes))
            cg_graph.nodes[node]['graph'] = g


def _annotate_residues(cg, res):
    """
    Propagates resname and resid from a residue-level graph down to CG nodes.

    Parameters
    ----------
    cg: networkx.Graph
        The coarse-grained molecule graph whose nodes will be annotated.
    res: networkx.Graph
        The residue-level graph produced by an extra resolver pass.
    """
    for node in res:
        resname = res.nodes[node]["fragname"]
        resid = res.nodes[node]["fragid"]
        for cgnode in res.nodes[node]['graph'].nodes:
            cg.nodes[cgnode]["resname"] = resname
            cg.nodes[cgnode]["resid"] = resid


def _resolve_cg_match(cgs_str, mol_graph):
    """
    Resolves a cgsmiles string and returns the matched (cg, aa, match) triple,
    or None if the string does not match mol_graph.

    Parameters
    ----------
    cgs_str: str
        A cgsmiles string encoding the CG mapping.
    mol_graph: networkx.Graph
        The all-atom molecule graph to match against.

    Returns
    -------
    tuple[networkx.Graph, networkx.Graph, dict] or None
    """
    resolver = MoleculeResolver.from_string(cgs_str, last_all_atom=True)
    nres = cgs_str.count("{")
    assert nres > 1, "cgsmiles string must contain at least two resolution levels."

    if nres > 2:
        res, _ = resolver.resolve()

    cg, aa = resolver.resolve()

    if nres > 2:
        _annotate_residues(cg, res)

    _annotate_vs(cg)
    match = find_one_graph_match(aa, mol_graph)
    return (cg, aa, match) if match else None


def _collect_bead_data(cg, match, mol_idxs, mol_graph, bead_count, res_iter):
    """
    Iterates over all molecule instances and collects per-bead mapped atoms,
    weights, and bead indices.

    Parameters
    ----------
    cg: networkx.Graph
    match: dict
        Mapping from CG atom indices to all-atom indices (for one molecule instance).
    mol_idxs: list[int]
        Indices of all instances of this molecule in the universe.
    mol_graph: networkx.Graph
    bead_count: int
        Running bead index to continue from.
    res_iter: ResidueIter

    Returns
    -------
    tuple[list, list, list, int, ResidueIter]
        mapped_atoms, weights, bead_idxs, updated bead_count, res_iter
    """
    mapped_atoms, weights, bead_idxs = [], [], []
    prev_resid = -1
    offset = 0

    for _ in mol_idxs:
        for bead in cg.nodes:
            resid = cg.nodes[bead]['resid']
            resname = cg.nodes[bead]['resname']
            if prev_resid != resid:
                prev_resid = resid
                res_iter.add_residue(resid=resid, resname=resname)

            atoms = cg.nodes[bead]['graph'].nodes
            mapped_atoms.append(
                numba.typed.List([match[atom] + offset for atom in atoms])
            )
            bead_weights = nx.get_node_attributes(cg.nodes[bead]['graph'], 'weight')
            weights.append(
                np.array([bead_weights.get(atom, 1.0) for atom in atoms], dtype=np.float32)
            )
            bead_idxs.append(bead_count)
            bead_count += 1

        offset += len(mol_graph)

    return mapped_atoms, weights, bead_idxs, bead_count, res_iter


def cgsmiles_to_mapping(univ, cgsmiles_strs, mol_names, mol_matching=True):
    """
    Given a list of mappings described by cgsmiles strings, maps atom indices
    to CG beads using graph isomorphism.

    Parameters
    ----------
    univ: :class:`fast_forward.universe_handler.UniverseHandler`
    cgsmiles_strs: list[str]
    mol_names: list[str]
    mol_matching: bool
        If False, the order of cgsmiles strings is assumed to match the order
        of molecules.

    Returns
    -------
    list, list, dict
    """
    all_mapped_atoms, all_bead_idxs, all_weights = [], [], []
    mappings = {}
    bead_count = 0
    res_iter = ResidueIter()

    for idx, mol_name in enumerate(mol_names):
        mol_graph = univ.molecule_graphs[mol_name]
        candidates = [cgsmiles_strs[idx]] if mol_matching else cgsmiles_strs

        # Find the first cgsmiles string that matches this molecule
        resolved = None
        for cgs_str in candidates:
            resolved = _resolve_cg_match(cgs_str, mol_graph)
            if resolved is not None:
                break
        if resolved is None:
            raise SyntaxError(
                f'No matching cgsmiles string found for molecule {mol_name}.'
            )
        cg, aa, match = resolved

        mappings = get_mappings(cg, univ, match, mappings)
        mol_idxs = univ.mol_idxs_by_name[mol_name]
        mapped_atoms, weights, bead_idxs, bead_count, res_iter = _collect_bead_data(
            cg, match, mol_idxs, mol_graph, bead_count, res_iter
        )
        all_mapped_atoms.extend(mapped_atoms)
        all_weights.extend(weights)
        all_bead_idxs.extend(bead_idxs)

    return (
        numba.typed.List(all_mapped_atoms),
        numba.typed.List(all_bead_idxs),
        mappings,
        numba.typed.List(all_weights),
        res_iter,
    )
