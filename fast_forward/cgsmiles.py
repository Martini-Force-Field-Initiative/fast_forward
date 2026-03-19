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
from cgsmiles.resolve import MoleculeResolver
from .map_file_parers import Mapping
from .universe_handler import ResidueIter

def load_cgsmiles_library(filepath):
    """
    Given a file that lists cgsmiles strings,
    read and return a list.

    Parameters
    ----------
    filepath: str

    Returns
    -------
    list[str]
    """
    cgsmiles_strs = []
    with open(filepath) as _file:
        for line in _file.readlines():
            cgsmiles_strs.append(line.strip())
    return cgsmiles_strs

def find_one_graph_match(graph1, graph2):
    """
    Returns one ismags match when graphs are isomorphic
    otherwise []. Matches are done based on element and
    bond order. That should be sufficent to also account
    for charges implicitly via the orders.

    Parameters
    ----------
    graph1: networkx.Graph
    graph2: networkx.Graph

    Returns
    -------
    abc.iteratable
    """

    def node_match(n1, n2):
        return n1["element"] == n2['element']

    def edge_match(e1, e2):
        return e1['order'] == e2['order']

    if len(graph1) != len(graph2):
        raw_matches = iter([])
    else:
        GM = nx.isomorphism.GraphMatcher(graph1,
                                         graph2,
                                         node_match=node_match,
                                         edge_match=edge_match)

        raw_matches = GM.subgraph_isomorphisms_iter()
    try:
        mapping = next(raw_matches)
    except StopIteration:
        mapping = []
    return mapping

def _most_common(_list):
    return max(set(_list), key=_list.count)

def get_mappings(cg, univ, _match, mappings):
    """
    We assing each coarse CG node a resname and resid
    based on the all-atom univ. If a bead is split between
    residues we simply assing the resname and id of the
    majority of atoms.

    Parameters
    ----------
    cg: networkx.graph
        the coarse molecule
    univ: :class:fast_forward.universe_handler.UniverseHandler
        the universe stroing the fine grained molecule information
    _match: dict
        a dict mapping the coarse atoms to fine grained atoms
    mappings:
        dict[:class:fast_forward.map_file_parser.Mapping]
        a dict of mapping objects

    Return
    ------
    dict[:class:fast_forward.map_file_parser.Mapping]
        updated mappings
    """
    target_resids = {}
    for bead in cg.nodes:
        atoms = cg.nodes[bead]['graph'].nodes
        resnames = [univ.atoms[_match[atom]].resname for atom in atoms]
        resname = _most_common(resnames)
        resids = [univ.atoms[_match[atom]].resid for atom in atoms]
        resid = _most_common(resids)
        cg.nodes[bead]['resname'] = resname
        cg.nodes[bead]['resid'] = resid
        mapping = mappings.get(resname, Mapping(resname, resname))
        target_resid = target_resids.get(resname, resid)
        target_resids[resname] = target_resid
        if resid != target_resids[resname]:
            continue
        for adx, atom in enumerate(atoms):
            weight = np.float32(cg.nodes[bead]['graph'].nodes[atom].get('weight', 1))
            mapping.add_atom(cg.nodes[bead]["fragname"], #+f"{bead}",
                             _match[atom],
                             atom=univ.atoms[_match[atom]].name,
                             weight=weight)
        mappings[resname] = mapping
    return mappings

def _annotate_vs(cg_graph):
    """
    If a node in the cg_graph is not mapped from atom positions,
    it is a virtual site. Thus, we will compose the mapped position
    as the union framgment graphs of neihgboring atoms. This appraoch
    should be equivalent to a virtual_sitesn mapping.

    Parameters
    ----------
    cg_graph: netwrokx.Graph
        the graph describing a coarse molecule
    """
    for node in cg_graph.nodes:
        if len(cg_graph.nodes[node].get('graph',[])) == 0:
            g = nx.Graph()
            for neigh in cg_graph.neighbors(node):
                g.add_nodes_from(list(cg_graph.nodes[neigh]['graph'].nodes))
            cg_graph.nodes[node]['graph'] = g

def cgsmiles_to_mapping(univ, cgsmiles_strs, mol_names, mol_matching=True):
    """
    Given a list of mappings described by cgsmiles strings
    this function maps the atom indices to CG beads using,
    graph isomorphism.

    Parameters
    ----------
    univ: :class:`fast_forward.UniverseHandler`
    cgsmiles_strs: list[str]
    mol_names: list[str]
    mol_matching: bool
        if is False the order of cgsmiles strings is equivalent
        to the order of molecules

    Retunrs
    -------
    list, list, dict
    """
    mapped_atoms, bead_idxs, mappings, weights = [], [], {}, []
    bead_count = 0
    for idx, mol_name in enumerate(mol_names):
        mol_graph = univ.molecule_graphs[mol_name]
        # names match the order of cgs strings
        if mol_matching:
            possible_cgs = [cgsmiles_strs[idx]]
        else:
            possible_cgs = cgsmiles_strs

        for cgs_str in possible_cgs:
            # read the cgsmiles string
            resolver = MoleculeResolver.from_string(cgs_str, last_all_atom=True)
            cg, aa = resolver.resolve_all()
            _annotate_vs(cg)
            _match = find_one_graph_match(aa, mol_graph)
            if _match:
                break
        else:
            raise SyntaxError('No matching cgsmiles string found for '
                              f'molecule {mol_name}.')

        # assgin resids to the beads
        mappings = get_mappings(cg, univ, _match, mappings)

        offset=0
        target_resids = {}
        res_iter = ResidueIter()
        prev_resid = -1
        for mol_idx in univ.mol_idxs_by_name[mol_name]:
            for bead in cg.nodes:
                resid = cg.nodes[bead]['resid']
                resname = cg.nodes[bead]['resname']
                if prev_resid != resid:
                    prev_resid = resid
                    res_iter.add_residue(resid=resid, resname=resname)
                atoms = cg.nodes[bead]['graph'].nodes
                mapped_atoms.append(numba.typed.List([_match[atom]+offset for atom in atoms]))
                set_weights = nx.get_node_attributes(cg.nodes[bead]['graph'], 'weight')
                weights.append(np.array([set_weights.get(node, 1.0) for node in atoms], dtype=np.float32))
                bead_idxs.append(bead_count)
                bead_count += 1
            offset += len(mol_graph)

    mapped_atoms = numba.typed.List(mapped_atoms)
    bead_idxs = numba.typed.List(bead_idxs)
    weights = numba.typed.List(weights)
    return mapped_atoms, bead_idxs, mappings, weights, res_iter
