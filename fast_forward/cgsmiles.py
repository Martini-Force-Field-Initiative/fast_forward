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
from collections import Counter, namedtuple
import numpy as np
import numba
import networkx as nx
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

    One string per line. Empty lines and lines starting with ``;`` or ``#``
    (comments) are skipped. Inline comments are not allowed, as both characters
    are used in cgsmiles strings.

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
            line = line.strip()
            if not line or line.startswith((';', '#')):
                continue
            cgsmiles_strs.append(line)
    return cgsmiles_strs


def find_one_graph_match(graph1, graph2):
    """
    Returns one match when the graphs are isomorphic, otherwise [].
    The heavy-atom graphs are matched with VF2 (networkx GraphMatcher),
    comparing only the element and hcount of each node.
    Hydrogens are then assigned to the hydrogens of the matched heavy atom
    (in arbitrary order).

    Parameters
    ----------
    graph1: networkx.Graph
    graph2: networkx.Graph

    Returns
    -------
    dict or list
        Mapping from graph1 nodes to graph2 nodes, or [] if there is no match.
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


def get_mappings(cg, univ, _match, mappings):
    """
    Builds the mapping objects of a matched molecule. The CG graph is not
    changed.

    Every CG node must have a resname and resid (set by _match_resolved).
    Each resname gets one Mapping, filled from the first residue with that
    name; the beads are named fragname + bead index.

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
        # resids come from the cgsmiles string, not the topology: a bead can
        # span several topology residues, so it has no unique topology resid
        resname = cg.nodes[bead]['resname']
        resid = cg.nodes[bead]['resid']
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


def _parse_wvs(bead_graph, atom, key='wVS'):
    """
    Returns the ``wVS`` annotation (or another annotation given by `key`,
    e.g. ``wVS_TN6``) of an atom as float, or None if it has none.
    """
    value = bead_graph.nodes[atom].get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}='{value}' of atom {bead_graph.nodes[atom].get('element')} "
                         "is not a number.") from None


def _own_wvs(bead_graph, atom, vs_name):
    """
    The wVS of an atom for the virtual site `vs_name` from its own
    annotations: ``wVS_<vs_name>`` if given, else ``wVS``, else None.
    """
    wvs = _parse_wvs(bead_graph, atom, f'wVS_{vs_name}')
    if wvs is None:
        wvs = _parse_wvs(bead_graph, atom)
    return wvs


def _vs_weight(bead_graph, atom, vs_name):
    """
    Weight of an atom of a parent bead when it is used for the virtual
    site with fragment name `vs_name`.

    Mirrors how cgsmiles treats w. The first of these that is given wins:

    1. the atom's own ``wVS_<vs_name>``
    2. the atom's own ``wVS``
    3. for a hydrogen: 1. and 2. of the heavy atom it is bonded to
    4. the atom's normal mapping weight (w, default 1)
    """
    wvs = _own_wvs(bead_graph, atom, vs_name)

    if wvs is None and bead_graph.nodes[atom].get('element') == 'H':
        for anchor in bead_graph.neighbors(atom):
            if bead_graph.nodes[anchor].get('element') != 'H':
                wvs = _own_wvs(bead_graph, anchor, vs_name)
                break

    if wvs is None:
        return bead_graph.nodes[atom].get('weight', 1.0)
    return wvs


def _warn_unused_wvs_keys(cg_graph, vs_nodes):
    """
    Prints a warning for every ``wVS_<name>`` annotation on the atoms of a
    bead that does not border a virtual site with fragment name <name>,
    since such an annotation (e.g. a typo) has no effect.
    """
    reported = set()
    for bead in cg_graph.nodes:
        if bead in vs_nodes:
            continue
        vs_names = {cg_graph.nodes[neigh].get('fragname')
                    for neigh in cg_graph.neighbors(bead) if neigh in vs_nodes}
        fragname = cg_graph.nodes[bead].get('fragname')
        for _, data in cg_graph.nodes[bead]['graph'].nodes(data=True):
            for key in data:
                if not key.startswith('wVS_') or key[len('wVS_'):] in vs_names:
                    continue
                if (fragname, key) in reported:
                    continue
                reported.add((fragname, key))
                print(f"WARNING - bead {fragname} has an atom annotated with {key}, "
                      f"but no virtual site named {key[len('wVS_'):]} is bonded to it; "
                      "the annotation is ignored.")


def _annotate_vs(cg_graph):
    """
    Identifies virtual sites in the CG graph and annotates them.

    If a node in cg_graph has no mapped atom positions, it is treated as a
    virtual site. Its mapped position is composed as the union of fragment
    graphs of its neighboring beads, which is equivalent to a virtual_sitesn
    mapping. Neighboring virtual sites are not parents and are ignored.

    The atoms keep their attributes. Their weight for the virtual site is
    the wVS atom annotation if given (e.g. [C;w=1;wVS=0]), which
    hydrogens inherit from their heavy atom just like w; otherwise it
    is the normal mapping weight.
    If several virtual sites are built on the same bead, an atom can set
    its weight for one of them with ``wVS_<name>``, where <name> is the
    fragment name of the virtual site (e.g. [C;wVS=0;wVS_TN6=1]). It takes
    precedence over ``wVS``, which then acts as the default for the other
    virtual sites.
    An atom that is shared between several neighboring beads (e.g. ``w=0.5``
    in two beads) gets the sum of its weights, so it counts as much as an
    atom that belongs to one bead.

    Parameters
    ----------
    cg_graph: networkx.Graph
        The graph describing a coarse molecule.

    Raises
    ------
    ValueError
        If a virtual site has no neighboring bead with atoms.
    """
    # identify all virtual sites first, so that neighboring virtual sites are
    # never used as parents, whatever order the nodes are processed in
    vs_nodes = {node for node in cg_graph.nodes
                if len(cg_graph.nodes[node].get('graph', [])) == 0}
    _warn_unused_wvs_keys(cg_graph, vs_nodes)
    for node in vs_nodes:
        vs_name = cg_graph.nodes[node].get('fragname')
        parents = [neigh for neigh in cg_graph.neighbors(node) if neigh not in vs_nodes]
        if not parents:
            raise ValueError(
                f"Virtual site {vs_name} (bead {node}) "
                "is not bonded to any bead with atoms, so its position is undefined."
            )
        g = nx.Graph()
        for neigh in parents:
            bead_graph = cg_graph.nodes[neigh]['graph']
            for atom, data in bead_graph.nodes(data=True):
                weight = _vs_weight(bead_graph, atom, vs_name)
                if atom in g:
                    g.nodes[atom]['weight'] += weight
                else:
                    g.add_node(atom, **{**data, 'weight': weight})
        cg_graph.nodes[node]['graph'] = g


def _annotate_residues(cg, res):
    """
    Propagates resname and resid from a residue-level graph down to CG nodes.

    Parameters
    ----------
    cg: networkx.Graph
        The coarse-grained molecule graph whose nodes will be annotated.
    res: networkx.Graph
        The residue-level graph, i.e. the level directly above the CG level.
        The 'graph' attribute of each node holds its CG nodes.
    """
    for node in res:
        resname = res.nodes[node]["fragname"]
        # residue nodes are numbered in sequence order, which gives the resid.
        # Their fragid cannot be used: the top level has none, and on lower
        # levels it is the id of the parent fragment, shared by its children.
        resid = node + 1
        for cgnode in res.nodes[node]['graph'].nodes:
            cg.nodes[cgnode]["resname"] = resname
            cg.nodes[cgnode]["resid"] = resid


def _order_by_bead_index(cg):
    """
    Returns a copy of the CG graph whose nodes are relabeled 0..N-1 in the
    order given by the bead annotation index (e.g. [#A;index=2]).

    The index is a sort key, not a position: beads are sorted by it, and
    beads with equal index keep their order from the cgsmiles string.
    As in python indexing, negative indices count from the end.
    If the beads carry a residue annotation (3-level strings), sorting
    happens within each residue, so the beads of a residue stay contiguous.
    """
    
    raw = {node: cg.nodes[node].get('index') for node in cg.nodes}
    missing = [cg.nodes[node]['fragname'] for node, val in raw.items() if val is None]
    if len(missing) == len(raw): # no beads have an index
        return cg
    if missing: # some beads have no index
        print(f"WARNING - {len(missing)} of {len(raw)} beads have no 'index' annotation "
              f"(bead types {sorted(set(missing))}); using index 0 for them.")
    indices = {}
    for node, val in raw.items():
        if val is None: # default to 0
            indices[node] = 0
            continue
        try:
            indices[node] = int(val)
        except (TypeError, ValueError):
            raise ValueError(f"Bead index '{val}' of bead {cg.nodes[node]['fragname']} "
                             "is not an integer.") from None

    # residues in order of first appearance; all 0 without a residue level
    residue_rank = {}
    for node in cg.nodes:
        residue_rank.setdefault(cg.nodes[node].get('resid'), len(residue_rank))

    # create a new graph with reordered nodes
    order = sorted(cg.nodes, key=lambda node: (residue_rank[cg.nodes[node].get('resid')],
                                               indices[node] < 0, #puts negative indices after all non-negative ones
                                               indices[node]))
    relabel = {old: new for new, old in enumerate(order)}
    ordered = nx.Graph(**cg.graph)
    ordered.add_nodes_from((relabel[node], cg.nodes[node]) for node in order)
    ordered.add_edges_from((relabel[u], relabel[v], data)
                           for u, v, data in cg.edges(data=True))
    return ordered


ResolvedCGS = namedtuple(
    'ResolvedCGS', ['cgs_str', 'cg', 'aa', 'has_residue_level', 'element_counts']
)
ResolvedCGS.__doc__ = """
The molecule-independent result of resolving one cgsmiles string.

cg is shared by every molecule matched with this string and must not be
modified; _match_resolved works on a copy.
"""


def _element_counts(graph):
    """Number of atoms per element of a graph."""
    return Counter(nx.get_node_attributes(graph, 'element').values())


def _resolve_cgsmiles(cgs_str):
    """
    Resolves a cgsmiles string. Depends only on the string, so it is done
    once per string and reused for every molecule (see _match_resolved).

    For strings with three or more levels, the CG nodes get resname and resid
    from the residue level. Virtual sites are annotated.

    Parameters
    ----------
    cgs_str: str
        A cgsmiles string encoding the CG mapping.

    Returns
    -------
    ResolvedCGS
    """
    resolver = MoleculeResolver.from_string(cgs_str, last_all_atom=True)
    if resolver.resolutions < 1:
        raise ValueError("cgsmiles string must contain at least two resolution "
                         f"levels: {cgs_str}")

    # each resolve() step returns (level k, level k+1); the last step is
    # (CG, all-atom). With more than two levels, the level directly above
    # the CG level holds the residues; levels further up only group them.
    metas = [meta for meta, _ in resolver.resolve_iter()]
    cg, aa = resolver.meta_graph, resolver.molecule

    has_residue_level = len(metas) > 1
    if has_residue_level:
        _annotate_residues(cg, metas[-2])

    _annotate_vs(cg)
    return ResolvedCGS(cgs_str, cg, aa, has_residue_level, _element_counts(aa))


def _match_resolved(resolved, mol_graph, resname_default="RES"):
    """
    Matches a resolved cgsmiles string against a molecule graph and returns
    the (cg, aa, match) triple, or None if they do not match.

    Every node of the returned CG graph has a resname and resid: from the
    residue level for strings with three or more levels; otherwise the whole
    molecule is one residue named resname_default with resid 1.
    The returned cg is a copy; resolved is not modified.

    Parameters
    ----------
    resolved: ResolvedCGS
        Result of _resolve_cgsmiles.
    mol_graph: networkx.Graph
        The all-atom molecule graph to match against.
    resname_default: str
        Resname for strings without a residue level. Should be unique per
        molecule type (e.g. the molecule name), otherwise the mappings of
        different molecules are merged under one resname.

    Returns
    -------
    tuple[networkx.Graph, networkx.Graph, dict] or None
    """
    aa = resolved.aa
    
    # cheap rejection before the graph matching
    if len(aa) != len(mol_graph):
        return None
    if resolved.element_counts != _element_counts(mol_graph):
        return None

    match = find_one_graph_match(aa, mol_graph)
    if not match:
        return None

    # the resolved graph is shared between molecules: annotate a copy.
    # Node attribute dicts are copied; the bead graphs are only read.
    cg = resolved.cg.copy()
    if not resolved.has_residue_level:
        # no residue level: the whole molecule is one residue
        for node in cg.nodes:
            cg.nodes[node]["resname"] = resname_default
            cg.nodes[node]["resid"] = 1
    # reorder last: residue annotation relies on the original node labels.
    # Only after a match, so non-matching candidate strings are not validated.
    cg = _order_by_bead_index(cg)
    return cg, aa, match


def _first_atom_by_molnum(univ):
    """
    Returns a dict mapping each molecule number to the index of its first atom.
    """
    molnums, first_atoms = np.unique(np.asarray(univ.atoms.molnums), return_index=True)
    return dict(zip(molnums.tolist(), first_atoms.tolist()))


def _copy_offsets(first_atom, mol_idxs):
    """
    Atom index shift of each copy of a molecule relative to its first copy.

    Parameters
    ----------
    first_atom: dict[int, int]
        Index of the first atom of each molecule number.
    mol_idxs: list[int]
        Molecule numbers of all copies of one molecule type.

    Returns
    -------
    list[int]
    """
    base = first_atom[mol_idxs[0]]
    return [first_atom[mol_idx] - base for mol_idx in mol_idxs]


def _check_contiguous_residues(cg):
    """
    Raises a ValueError if the beads of a residue are not contiguous in the
    bead order. Each residue becomes one block of consecutive beads in the
    CG universe, so split residues cannot be represented.
    """
    seen_resids = set()
    prev_resid = None
    for bead in cg.nodes:
        resid = cg.nodes[bead]['resid']
        if resid != prev_resid:
            if resid in seen_resids:
                raise ValueError(
                    f"The beads of residue {cg.nodes[bead]['resname']} {resid} are "
                    "not contiguous; check the bead 'index' annotations."
                )
            seen_resids.add(resid)
            prev_resid = resid


def _check_bead_weights(cg, mol_name):
    """
    Raises a ValueError if the atom weights of a bead sum to zero (or less),
    since its position would be a division by zero.
    """
    for bead in cg.nodes:
        bead_graph = cg.nodes[bead]['graph']
        total = sum(weight for _, weight in bead_graph.nodes(data='weight', default=1.0))
        if total <= 0:
            raise ValueError(
                f"The atom weights of bead {cg.nodes[bead]['fragname']} (bead {bead}) "
                f"in molecule {mol_name} sum to {total}, so its position is undefined. "
                "Check the w, wVS and wVS_<name> annotations."
            )


def _collect_bead_data(cg, match, offsets, bead_count, res_iter):
    """
    Iterates over the given molecule instances and collects per-bead mapped
    atoms, weights, and bead indices.

    Parameters
    ----------
    cg: networkx.Graph
    match: dict
        Mapping from CG atom indices to all-atom indices (for the first
        molecule instance).
    offsets: list[int]
        Atom index shift of each molecule.
    bead_count: int
        Running bead index to continue from.
    res_iter: ResidueIter

    Returns
    -------
    tuple[list, list, list, int, ResidueIter]
        mapped_atoms, weights, bead_idxs, updated bead_count, res_iter
    """
    mapped_atoms, weights, bead_idxs = [], [], []

    for offset in offsets:
        prev_resid = None
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

    return mapped_atoms, weights, bead_idxs, bead_count, res_iter


def _counter_diff(string_counts, topology_counts):
    """Keys whose counts differ, as (key, string count, topology count)."""
    keys = sorted(set(string_counts) | set(topology_counts), key=str)
    return [(key, string_counts[key], topology_counts[key]) for key in keys
            if string_counts[key] != topology_counts[key]]


def _heavy_atom_classes(graph):
    """Per heavy atom, its (element, number of bonded hydrogens)."""
    classes = {}
    for node in graph.nodes:
        if graph.nodes[node]['element'] == 'H':
            continue
        n_h = sum(1 for neigh in graph.neighbors(node)
                  if graph.nodes[neigh]['element'] == 'H')
        classes[node] = (graph.nodes[node]['element'], n_h)
    return classes


def _describe_mismatch(aa, mol_graph, atom_names=None, max_names=8):
    """
    Explains why the all-atom graph of a cgsmiles string does not match a
    molecule graph from the topology.

    Checks, in order: element composition, the number of hydrogens on each
    heavy atom, and the bonds between heavy atoms; reports the first that
    differs.

    Parameters
    ----------
    aa: networkx.Graph
        All-atom graph of the cgsmiles string.
    mol_graph: networkx.Graph
        Molecule graph from the topology.
    atom_names: dict, optional
        Topology atom names by node, used to point at differing atoms.
    max_names: int
        Maximum number of atom names listed.

    Returns
    -------
    tuple[tuple[int, int], list[str]]
        A score (lower is closer) and the lines of the explanation.
    """
    lines = [f"atoms (string/topology): {len(aa)}/{len(mol_graph)}"]
    elements = _counter_diff(Counter(nx.get_node_attributes(aa, 'element').values()),
                             Counter(nx.get_node_attributes(mol_graph, 'element').values()))
    element_score = sum(abs(s - t) for _, s, t in elements)
    if elements:
        lines.append("elements differ (string/topology): "
                     + ", ".join(f"{el} {s}/{t}" for el, s, t in elements))
        return (element_score, 0), lines

    aa_classes = _heavy_atom_classes(aa)
    mol_classes = _heavy_atom_classes(mol_graph)
    classes = _counter_diff(Counter(aa_classes.values()), Counter(mol_classes.values()))
    if classes:
        lines.append("same elements, but the hydrogens are on different atoms. "
                     "Heavy atoms by number of bonded H (string/topology): "
                     + ", ".join(f"{el} with {n_h} H {s}/{t}" for (el, n_h), s, t in classes))
        surplus = {key for key, s, t in classes if t > s}
        if atom_names:
            names = [atom_names[node] for node, key in mol_classes.items() if key in surplus]
            more = f" ... ({len(names)} in total)" if len(names) > max_names else ""
            lines.append("topology atoms in the classes that are more common than in "
                         f"the string: {' '.join(map(str, names[:max_names]))}{more}")
        return (0, sum(abs(s - t) for _, s, t in classes)), lines

    aa_bonds = aa.subgraph(aa_classes).number_of_edges()
    mol_bonds = mol_graph.subgraph(mol_classes).number_of_edges()
    if aa_bonds != mol_bonds:
        lines.append("same atoms and hydrogens, but a different number of bonds between "
                     f"heavy atoms (string/topology): {aa_bonds}/{mol_bonds}. "
                     "Check for missing bonds or ring closures in the topology or string.")
    else:
        lines.append("same atoms, hydrogens and number of bonds, but the atoms are "
                     "connected differently (e.g. a different isomer).")
    return (0, 0), lines


def _no_match_report(candidates, mol_graph, univ):
    """
    Explanation for a molecule that no cgsmiles string matches, based on the
    candidate string closest to the molecule.

    Parameters
    ----------
    candidates: list[ResolvedCGS]
    mol_graph: networkx.Graph
    univ: :class:`fast_forward.universe_handler.UniverseHandler`
    """
    atom_names = {node: univ.atoms[node].name for node in mol_graph.nodes}
    scored = []
    for resolved in candidates:
        score, lines = _describe_mismatch(resolved.aa, mol_graph, atom_names)
        scored.append((score, resolved.cgs_str, lines))
    score, cgs_str, lines = min(scored, key=lambda item: item[0])
    if len(candidates) == 1:
        header = f"cgsmiles string: {cgs_str}"
    else:
        header = f"closest of the {len(candidates)} cgsmiles strings: {cgs_str}"
    report = [header] + [f"  - {line}" for line in lines]
    return "\n".join(report)


def cgsmiles_to_mapping(univ, cgsmiles_strs, mol_names, mol_matching=True):
    """
    Given a list of mappings described by cgsmiles strings, maps atom indices
    to CG beads using graph isomorphism. The beads are returned in topology
    order, i.e. molecules appear in the same order as in the universe.

    Parameters
    ----------
    univ: :class:`fast_forward.universe_handler.UniverseHandler`
    cgsmiles_strs: list[str]
    mol_names: list[str]
    mol_matching: bool
        If True, the cgsmiles strings are in the same order as mol_names
        (one string per molecule type). If False, every molecule is matched
        against all strings and the first matching string is used.

    Returns
    -------
    mapped_atoms: numba.typed.List
        Per bead, the indices of its atoms in the universe.
    bead_idxs: numba.typed.List
        Index of each bead in the CG system.
    mappings: dict[:class:`fast_forward.map_file_parers.Mapping`]
        Mapping objects by resname; used for the bead names.
    weights: numba.typed.List
        Per bead, the weight of each of its atoms.
    res_iter: :class:`fast_forward.universe_handler.ResidueIter`
        The CG residues in topology order.
    """
    if mol_matching and len(cgsmiles_strs) != len(mol_names):
        raise ValueError(
            f"Got {len(cgsmiles_strs)} cgsmiles strings for {len(mol_names)} molecule "
            f"types ({' '.join(mol_names)}). With ordered matching, give exactly one "
            "string per molecule type, in the same order, or use -nocgsorder to "
            "match each molecule against all strings."
        )

    all_mapped_atoms, all_bead_idxs, all_weights = [], [], []
    mappings = {}
    bead_count = 0
    res_iter = ResidueIter()
    first_atom = _first_atom_by_molnum(univ)
    resolved_by_name = {}
    # (first atom index, molecule name, offset) of every molecule copy
    copies = []

    # resolving is the expensive step and depends only on the string, so each
    # distinct string is resolved once and reused for every molecule type
    resolved_strs = {cgs_str: _resolve_cgsmiles(cgs_str)
                     for cgs_str in dict.fromkeys(cgsmiles_strs)}

    for idx, mol_name in enumerate(mol_names):
        mol_graph = univ.molecule_graphs[mol_name]
        candidates = [cgsmiles_strs[idx]] if mol_matching else list(resolved_strs)
        candidates = [resolved_strs[cgs_str] for cgs_str in candidates]

        # Find the first cgsmiles string that matches this molecule
        resolved = None
        for candidate in candidates:
            # without a residue level, name the residue after the molecule
            resolved = _match_resolved(candidate, mol_graph, resname_default=mol_name)
            if resolved is not None:
                break
        if resolved is None:
            raise ValueError(
                f'No matching cgsmiles string found for molecule {mol_name}.\n'
                + _no_match_report(candidates, mol_graph, univ)
            )
        cg, aa, match = resolved

        mappings = get_mappings(cg, univ, match, mappings)
        _check_contiguous_residues(cg)
        _check_bead_weights(cg, mol_name)
        resolved_by_name[mol_name] = (cg, match)
        mol_idxs = univ.mol_idxs_by_name[mol_name]
        offsets = _copy_offsets(first_atom, mol_idxs)
        copies.extend((first_atom[mol_idx], mol_name, offset)
                      for mol_idx, offset in zip(mol_idxs, offsets))

    # emit the beads molecule by molecule in topology order, so the CG output
    # follows the topology (as with .map files), not the order of mol_names
    copies.sort(key=lambda copy: copy[0])
    for _, mol_name, offset in copies:
        cg, match = resolved_by_name[mol_name]
        mapped_atoms, weights, bead_idxs, bead_count, res_iter = _collect_bead_data(
            cg, match, [offset], bead_count, res_iter
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
