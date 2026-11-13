"""
Solvent accessible surface area (SASA) of molecules over a trajectory.

The SASA itself is computed with ``sassy`` (Shrake-Rupley, every molecule as a body
on its own, so the SASA describes the shape of a molecule and not its exposure in the
surrounding system):

- ``ff_map -sasa`` computes the SASA of the atomistic (AA) trajectory, with the
  atomistic van der Waals radii, and saves its distribution as the reference.
- ``ff_assess -sasa`` computes the SASA of the coarse-grained (CG) trajectory, with
  the Martini bead radii, and scores it against that reference.

Time series and distributions are written in the same format as the interaction
distributions (``<prefix><molname>_sasa.dat`` and ``<prefix><molname>_sasa_distr.dat``),
so the reference SASA lives next to the reference bonded distributions.

The SASA is also split over the beads (``<prefix><molname>_<resid>_<bead>_sasa[_distr].dat``):
``ff_map`` assigns the SASA of every atom to the beads of the mapping (an atom shared by k
beads gives 1/k to each), leaving out dummy beads, i.e. beads whose atoms all belong to
other beads; ``ff_assess`` takes the SASA of every CG bead, virtual sites included, except
dummy beads (type ``U``, radius 0), which have no SASA.
"""
from collections import ChainMap
from pathlib import Path
import warnings
import numpy as np
import MDAnalysis as mda
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from fast_forward import DATA_PATH
from fast_forward.interaction_distribution import distribution


REFERENCE = 'reference' # ff_assess -sasa-surface: the frame closest to the mapped structure of ff_map
FALLBACK_STATISTIC = 'mode' # without a usable mapped structure


def _sassy():
    """
    Import sassy
    """
    try:
        import sassy
    except ImportError as err:
        msg = ("The SASA calculation (-sasa) requires the 'sassy' package, which is not "
               "installed. Install it (e.g. 'pip install git+https://github.com/ThallmairLab/sassy') "
               "and try again.")
        raise ImportError(msg) from err
    return sassy


def surface_statistic(value, allow_reference=False):
    """
    The statistic choosing the representative frame of a surface (``-sasa-stat``): a
    name of ``sassy.STATISTICS`` ('median', 'mean', 'mode', 'q1', 'q3', 'min', 'max',
    'mean+std', 'mean-std') or a fixed SASA (nm^2); with ``allow_reference`` also
    'reference' (the frame closest to the mapped structure of ``ff_map``).

    Raises
    ------
    ValueError
        if ``value`` is none of them
    """
    statistics = _sassy().STATISTICS
    if value in statistics or (allow_reference and value == REFERENCE):
        return value
    try:
        return float(value)
    except ValueError:
        known = [REFERENCE] * allow_reference + list(statistics)
        raise ValueError(f"unknown statistic {value!r}: use one of {', '.join(known)} "
                         "or a SASA in nm^2") from None


def _citation_map():
    with open(DATA_PATH/'citations.bib') as citation_file:
        ff_citations = read_bib(citation_file)
    return ChainMap(ff_citations, COMMON_CITATIONS)


def cite_aa_radii(ag: mda.AtomGroup):
    """
    Print the citations of the atomistic van der Waals radii used for the SASA of
    ag: Rowland & Taylor (1996), and Bondi (1964) for phosphorus if ag has any.
    """
    sassy = _sassy()
    citation_map = _citation_map()
    cite_string = citation_formatter(citation_map['VDW_radii'])
    print(f"\nSASA calculation uses atomistic van der Waals radii from:\n{cite_string}\n")
    if 'P' in sassy.atom_symbols(ag, cg=False):
        cite_string = citation_formatter(citation_map['VDW_radii_P'])
        print(f"\nPhosphorus atoms use atomistic van der Waals radii from:\n{cite_string}\n")


def mapped_beads(mapped_atoms, atom_indices):
    """
    Indices of the beads built only from the AA atoms ``atom_indices``.

    Parameters
    ----------
    mapped_atoms : list of list of int
        AA atom indices of every bead
    atom_indices : array of int
        AA atom indices, e.g. of one molecule or of all molecules of a type
    """
    atom_indices = set(int(idx) for idx in atom_indices)
    return [bdx for bdx, atoms in enumerate(mapped_atoms)
            if len(atoms) and set(int(idx) for idx in atoms) <= atom_indices]


def map_sasa(atom_sasa, beads):
    """
    Map the SASA of atoms to beads: every atom contributes equally (mapping weights
    are not used), and an atom shared between k beads contributes 1/k of its SASA to
    each of them.

    Dummy beads get no SASA: beads whose atoms all belong to other beads, placed from
    the atoms of their neighbours (e.g. the beads without atoms of their own in a
    CGsmiles mapping). Beads are checked from the largest to the smallest (a dummy
    built from its neighbours holds more atoms than each of them, so it is found
    before a neighbour could look like a dummy because of it); of beads with identical
    atoms, the first one is kept. k counts the remaining beads only.

    Parameters
    ----------
    atom_sasa : np.ndarray
        (n_atoms, n_frames) SASA of every atom
    beads : list of array of int
        rows of ``atom_sasa`` of the atoms of every bead (of one molecule)

    Returns
    -------
    bead_sasa : np.ndarray
        (n_beads - n_dummies, n_frames) SASA of every bead that is not a dummy, in the
        order of ``beads``
    dummy : np.ndarray
        (n_beads,) bool, True for the dummy beads
    unassigned : np.ndarray
        rows of ``atom_sasa`` that are in no bead
    """
    beads = [np.unique(np.asarray(atoms, dtype=np.int64)) for atoms in beads] # remove duplicates

    # count how many beads every atom belongs to
    count = np.zeros(len(atom_sasa), dtype=np.int64)
    for atoms in beads:
        count[atoms] += 1

    # dummy beads: all atoms also belong to other (remaining) beads
    dummy = np.zeros(len(beads), dtype=bool)
    for b in sorted(range(len(beads)), key=lambda b: (-len(beads[b]), -b)):
        atoms = beads[b]
        if len(atoms) and (count[atoms] > 1).all():
            dummy[b] = True
            count[atoms] -= 1 # count is now the number of non-dummy beads of every atom

    # split the SASA between the non-dummy beads
    shared = atom_sasa / np.maximum(count, 1)[:, None] # divide the SASA of each atom by the number of beads it belongs to
    bead_sasa = np.array([shared[atoms].sum(axis=0) for atoms, d in zip(beads, dummy) if not d]) # sum the atom SASA of every bead
    bead_sasa = bead_sasa.reshape(int(np.count_nonzero(~dummy)), atom_sasa.shape[1])
    unassigned = np.flatnonzero(count == 0) # indices of the atoms that are in no bead
    return bead_sasa, dummy, unassigned


class MoleculeSASA:
    """
    SASA of all molecules of one type over a trajectory, computed once with a single
    :class:`sassy.SASA` (every molecule a body on its own), which is then also used for
    the surface structures.

    As for the interaction distributions, the time series of all molecules are
    concatenated (all frames of the first molecule, then of the second, ...) and
    pooled into one distribution.

    Parameters
    ----------
    ag : MDAnalysis.AtomGroup
        atoms of the molecules of one type; molecules must be whole
    mol_name : str
        name of the molecule type, used for the file names
    cg : bool
        use the Martini bead radii (from the bead types) instead of the atomistic
        van der Waals radii (from the elements, cited when computed)
    prefix : str
        common prefix of the file names
    ndots : int, optional
        minimum number of surface dots per particle (default: that of sassy)

    Attributes
    ----------
    sasa : sassy.SASA
        the SASA after its run; its :class:`sassy.Results` in ``sasa.results``
    series : np.ndarray
        (n_molecules * n_frames,) SASA of every molecule in every frame (nm^2)
    distribution : np.ndarray
        (n_bins, 2) distribution of :attr:`series`, columns: bin centre (nm^2), probability density
    bead_names : list of str or None
        ``<resid>_<bead name>`` of every bead of a molecule (without the beads left out), after
        :meth:`map_to_beads` or :meth:`cg_bead_sasa`
    bead_series : np.ndarray or None
        (n_beads, n_molecules * n_frames) SASA of every bead, the molecules concatenated
        as for :attr:`series`
    bead_distributions : list of np.ndarray or None
        distribution of every bead, as :attr:`distribution`
    """
    def __init__(self, ag: mda.AtomGroup, mol_name: str, cg: bool = False, prefix: str = "",
                 ndots: int | None = None):
        if ag.n_atoms == 0:
            raise ValueError(f"No atoms selected for the SASA of {mol_name}.")
        
        self.mol_name = mol_name
        self.cg = cg
        self.prefix = prefix
        kwargs = {} if ndots is None else {'ndots': ndots}
        
        self.sasa = _sassy().SASA(ag, cg=cg, bodies='molecule', keep_atoms=True, **kwargs)
        results = self.sasa.run(verbose=True, name=mol_name)

        if not cg: # cite the AA radii papers
            cite_aa_radii(ag)

        self.series = results.body_sasa.ravel() # (n_molecules * n_frames,) SASA of every molecule in every frame
        self.distribution = distribution(self.series, 'sasa', mol_name) 
        self.bead_names = self.bead_series = self.bead_distributions = None

    @property
    def molecule_type(self):
        """sassy body type of the molecules (all of ag are of one type)"""
        return self.sasa.body_type_labels[0]

    def file_name(self, suffix, outdir="."):
        """``<outdir>/<prefix><mol_name>_<suffix>``"""
        return Path(outdir) / f"{self.prefix}{self.mol_name}_{suffix}"

    def save(self, dir):
        """
        Save the time series and distribution of the whole molecule and for each bead.
        """
        np.savetxt(self.file_name("sasa.dat", dir), self.series)
        np.savetxt(self.file_name("sasa_distr.dat", dir), self.distribution)
        if self.bead_series is not None:
            for name, series, distr in zip(self.bead_names, self.bead_series, self.bead_distributions):
                np.savetxt(self.file_name(f"{name}_sasa.dat", dir), series)
                np.savetxt(self.file_name(f"{name}_sasa_distr.dat", dir), distr)

    def _molecule_rows(self):
        """Rows of the atom SASA (positions in ag) of the atoms of every molecule."""
        molecules = self.sasa.results.atom_body # (n_atoms,) molecule (body) of every atom
        return [np.flatnonzero(molecules == m) for m in range(self.sasa.n_bodies)]

    def _set_bead_sasa(self, names, sasa_per_molecule):
        """
        Store the SASA of the beads, (n_beads, n_frames) for every molecule, and free the
        SASA of the atoms.
        """
        self.bead_names = list(names)
        n_frames = len(self.series) // max(1, self.sasa.n_bodies)
        self.bead_series = (np.concatenate(sasa_per_molecule, axis=1) if sasa_per_molecule
                            else np.zeros((len(names), n_frames))) # (n_beads, n_molecules * n_frames) one row per bead, the molecules concatenated
        self.bead_distributions = [distribution(series, 'sasa', f"{self.mol_name} {name}")
                                   for name, series in zip(self.bead_names, self.bead_series)]
        self.sasa.results.atom_sasa = None # the SASA of the atoms is no longer needed

    def map_to_beads(self, mapped_atoms, cg_universe):
        """
        Split the SASA of the atoms (AA) over the beads of the mapping, molecule by molecule.

        Every atom contributes equally, regardless of the mapping weights; an atom shared
        by k of the remaining beads gives 1/k of its SASA to each of them. Dummy beads, i.e.
        beads whose atoms all belong to other beads (see :func:`map_sasa`), get no SASA
        and are left out. Atoms in no bead count for the
        molecule only; a warning reports them. Beads are named ``<resid>_<bead name>``, the residue counted from 1
        within the molecule, as the atoms of the CG itp in ``ff_inter`` and ``ff_assess``.

        Parameters
        ----------
        mapped_atoms : list of list of int
            AA atom (universe) indices of every bead
        cg_universe : MDAnalysis.Universe
            the mapped universe, bead i built from the AA atoms ``mapped_atoms[i]``

        Raises
        ------
        ValueError
            if the molecules do not all have the same beads in the same order
        """
        ag = self.sasa.ag # atomistic ag 
        
        row = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64)
        row[ag.indices] = np.arange(ag.n_atoms) # universe index -> position in ag, -1 for atoms not in ag

		# molecule index of each atom
        molecule_rows = self._molecule_rows()
        molecule_of_row = np.empty(ag.n_atoms, dtype=np.int64)
        for m, rows in enumerate(molecule_rows):
            molecule_of_row[rows] = m
            
        # beads built only from the atoms of one molecule of this type, in the order of the mapping
        mol_beads = [[] for _ in molecule_rows]
        for bdx, atoms in enumerate(mapped_atoms):
            rows = row[np.asarray(list(atoms), dtype=np.int64)]
            if len(rows) and (rows >= 0).all() and (molecule_of_row[rows] == molecule_of_row[rows[0]]).all(): # all atoms of this bead belong to one molecule of this type
                mol_beads[molecule_of_row[rows[0]]].append((bdx, rows))

        names, sasa_per_molecule, n_unassigned, unassigned_names = None, [], 0, {}
        for m, beads in enumerate(mol_beads):
            bdxs = [bdx for bdx, _ in beads]
            # SASA of the atoms of this molecule, split over its beads (dummies get none)
            rows = molecule_rows[m]
            local = np.full(ag.n_atoms, -1, dtype=np.int64) # universe index -> position in rows, -1 for atoms not in this molecule
            local[rows] = np.arange(len(rows))
            
            bead_sasa, dummy, unassigned = map_sasa(self.sasa.results.atom_sasa[rows], [local[r] for _, r in beads])
            resindices = cg_universe.atoms.resindices[bdxs] if bdxs else np.zeros(0, dtype=int)
            all_names = [f"{res - resindices.min() + 1}_{cg_universe.atoms[bdx].name}"
                         for bdx, res in zip(bdxs, resindices)]
            mol_names = [name for name, d in zip(all_names, dummy) if not d]
            if names is None:
                names = mol_names
                skipped = [name for name, d in zip(all_names, dummy) if d]
                if skipped:
                    print(f"INFO - No SASA assigned to the dummy bead(s) {', '.join(skipped)} of "
                          f"{self.mol_name}: all their atoms belong to other beads")
            elif mol_names != names:
                raise ValueError(f"All {self.mol_name} molecules must have the same beads in the same order "
                                 f"for the bead SASA: molecule 1 has {', '.join(names)}, molecule {m + 1} has "
                                 f"{', '.join(mol_names)}.")
            sasa_per_molecule.append(bead_sasa)
            n_unassigned += len(unassigned)
            for atom in ag[rows[unassigned]]:
                unassigned_names.setdefault(f"{atom.resname}{atom.resid}:{atom.name}", None)
        if n_unassigned:
            listed = list(unassigned_names)
            more = f" and {len(listed) - 10} more" if len(listed) > 10 else ""
            print(f"WARNING - {n_unassigned} atom(s) of {self.mol_name} are in no bead of the mapping; their "
                  f"SASA counts for the molecule but not for any bead: {', '.join(listed[:10])}{more}")
        self._set_bead_sasa(names or [], sasa_per_molecule)

    def cg_bead_sasa(self, block):
        """
        Save the SASA of all CG beads in the ff block.

        Parameters
        ----------
        block : vermouth.molecule.Block
            the molecule type, its atoms in the order of every molecule of ag

        Raises
        ------
        ValueError
            if a molecule has a different number of beads than ``block``
        """
        nodes = list(block.nodes) # atoms from ff block
        all_names = [f"{block.nodes[node]['resid']}_{block.nodes[node]['atomname']}" for node in nodes] 
        molecule_rows = self._molecule_rows() # indices of the molecules in ag/sasa array

        # check that all molecules have the same number of beads as the block, and get the radii of the beads
        for m, rows in enumerate(molecule_rows):
            if len(rows) != len(nodes):
                raise ValueError(f"Molecule {m + 1} of {self.mol_name} has {len(rows)} beads, but the itp "
                                 f"has {len(nodes)}.")

        # remove dummy beads (radius 0) 
        radii = self.sasa.radii[molecule_rows[0]] if molecule_rows else np.zeros(len(nodes))
        keep = [i for i in range(len(nodes)) if radii[i] > 0]
        skipped = [all_names[i] for i in range(len(nodes)) if i not in keep] 
        if skipped:
            print(f"INFO - No SASA saved for the dummy bead(s) (radius 0) {', '.join(skipped)} of "
                  f"{self.mol_name}")

        # save the sasa of the remaining beads in the class attributes
        self._set_bead_sasa([all_names[i] for i in keep],
                        [self.sasa.results.atom_sasa[rows[keep]] for rows in molecule_rows])

    def write_surface(self, outdir=".", reference=None, statistic=FALLBACK_STATISTIC):
        """
        Write the structure and its surface dots of one molecule as
        ``<outdir>/<prefix><mol_name>_sasa_dots_<AA|CG>.pdb``. The B-factor of every atom and of its dots is the SASA of the atom (nm^2), and REMARK
        records tell how the molecule and frame were chosen and which radii were used.

        Without ``reference``, the molecule and frame whose SASA is closest to ``statistic``
        of the SASA distribution of the molecule type are used, from the SASA already
        computed. With ``reference`` (an AtomGroup or (n_atoms, 3) positions in nm, as many
        atoms as one molecule), the frame of the first molecule closest to it (lowest RMSD)
        is used, superposed onto it. Only the surface dots of the written frame are computed
        (fewer dots per particle than for the SASA, as they are for viewing).

        Parameters
        ----------
        outdir : str or Path
            directory of the file
        reference : MDAnalysis.AtomGroup or np.ndarray, optional
            structure to match the frame to
        statistic : str or float
            without ``reference``: a name of ``sassy.STATISTICS`` ('median', 'mean', 'mode',
            'q1', 'q3', 'min', 'max', 'mean+std', 'mean-std') or a fixed SASA (nm^2);
            default the mode (most probable SASA)

        Returns
        -------
        sassy.Surface
            the written surface (structure, dots, area in nm^2 from the dots, frame, ...)
        """
        fname = self.file_name(f"sasa_dots_{'CG' if self.cg else 'AA'}.pdb", outdir)
        surface = self.sasa.surface(body_type=self.molecule_type, reference=reference, statistic=statistic)
        surface.write(fname)

        if surface.rmsd is not None: # a reference is given: report the RMSD to it
            chosen = f"RMSD {surface.rmsd:.3f} nm to the mapped structure"
        else:
            chosen = f"closest to the {statistic if isinstance(statistic, str) else f'value {statistic:g} nm^2'}"
        print(f"INFO - Wrote SASA surface of {self.sasa.body_labels[surface.body]} in frame "
              f"{surface.frame} ({surface.area:.2f} nm^2, {chosen}) to {fname}")
        return surface

    def write_mapped_structure(self, surface, cg_universe, mapped_atoms, outdir="."):
        """
        Write the mapped (CG) structure of the AA molecule and frame of ``surface`` as
        ``<outdir>/<prefix><mol_name>_sasa_mapped.pdb``; ``ff_assess -sasa-surface``
        uses it to pick the matching CG frame.

        Parameters
        ----------
        surface : sassy.Surface
            AA surface from :meth:`write_surface`
        cg_universe : MDAnalysis.Universe
            the mapped universe, bead i built from the AA atoms ``mapped_atoms[i]``
        mapped_atoms : list of list of int
            AA atom indices of every bead
        """
        beads = mapped_beads(mapped_atoms, surface.atoms.indices)
        if not beads:
            print(f"WARNING - No mapped beads found for the SASA surface molecule of {self.mol_name}; "
                  "the mapped structure is not written.")
            return None
        fname = self.file_name("sasa_mapped.pdb", outdir)
        cg_universe.trajectory[surface.frame]
        with warnings.catch_warnings(): # attributes missing in the mapped topology (elements, chainIDs, ...)
            warnings.simplefilter("ignore", UserWarning)
            cg_universe.atoms[beads].write(str(fname))
        print(f"INFO - Wrote mapped structure of {self.mol_name} in frame {surface.frame} to {fname}")
        return fname

    def write_cg_surface(self, reference_dir=None, outdir=".", statistic=REFERENCE):
        """
        Write the CG surface (see :meth:`write_surface`). With ``statistic='reference'``
        (default), in the frame closest to the mapped structure written by
        ``ff_map -sasa-surface`` (``<reference_dir>/<prefix><mol_name>_sasa_mapped.pdb``), so
        that the AA and CG surfaces share one coordinate frame; without a matching mapped
        structure, in the frame closest to the mode of the SASA. With any other statistic
        (see :meth:`write_surface`), in the frame closest to it.
        """
        if statistic != REFERENCE:
            return self.write_surface(outdir, statistic=statistic)
        statistic = FALLBACK_STATISTIC # if the mapped structure is missing or does not match
        reference = None
        mapped_file = self.file_name("sasa_mapped.pdb", reference_dir) if reference_dir else None
        if mapped_file is not None and mapped_file.is_file():
            reference = mda.Universe(str(mapped_file)).atoms
            n_atoms = self.sasa.body_atoms(int(self.sasa.body_type_indices[self.molecule_type][0])).n_atoms
            if reference.n_atoms != n_atoms:
                print(f"WARNING - The mapped structure {mapped_file} has {reference.n_atoms} beads, but "
                      f"{self.mol_name} has {n_atoms}; writing the CG surface of the frame closest to the "
                      f"{statistic} SASA instead.")
                reference = None
        else:
            print(f"INFO - No mapped structure {mapped_file} found; writing the CG surface of the frame "
                  f"closest to the {statistic} SASA instead.")
        return self.write_surface(outdir, reference=reference, statistic=statistic)
