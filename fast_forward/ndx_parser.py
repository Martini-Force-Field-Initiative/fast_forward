import numpy as np
import MDAnalysis as mda

TYPES = {2: "bonds",
         3: "angles",
         4: "dihedrals"}

def ndx_to_ag(atoms_or_universe, infile):
    """
    Create MDAnalysis AtomGroups from a Universe and an index file.

    Examples
    --------
    .. code-block::

        u = mda.Universe(TPR, XTC)
        with open('index.ndx') as infile:
            for group_name, atom_group in ndx_to_ag(u, infile):
                print(group_name, atom_group)

    Parameters
    ----------
    atoms_or_universe: :class:`~MDAnalysis.core.universe.Universe` or :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to select from.
    infile: str
        The open index file to read.

    Yields
    ------
    group_name: str
        The name of the group as written in the index file.
    atoms: :class:`~MDAnalysis.core.groups.AtomGroup`
        An atom group containing the atoms for that group.

    """
    with open(infile) as _file:
        lines = _file.readlines()

    atoms = atoms_or_universe.atoms
    group_name = None
    indices = []
    for line in lines:
        comment_idx = line.find(';')
        if comment_idx >= 0:
            line = line[comment_idx:]
        line = line.strip()
        if line.startswith('['):
            # we do this after a section is completed unless we start
            if group_name is not None:
                indices = np.array(indices, dtype=int) - 1
                inter_type = TYPES[indices.shape[1]]
                yield (group_name, inter_type, indices)
            group_name = line[1:-1].strip()
            indices = []
        else:
            indices.append(line.split())

    # we do this at the end of the file
    if group_name is not None:
        indices = np.array(indices, dtype=int) - 1
        inter_type = TYPES[indices.shape[1]]
        yield (group_name, inter_type, indices)
