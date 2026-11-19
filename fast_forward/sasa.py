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
``ff_map`` assigns the SASA of every
atom to the beads of the mapping (an atom shared by k beads gives 1/k to each), leaving out
dummy beads, i.e. beads whose atoms all belong to other beads. The beads are set before the SASA
is computed and passed to sassy as groups of atoms, so their SASA is summed while the trajectory
is read and the SASA of every atom in every frame is never kept. ``ff_assess`` takes the SASA of
every CG bead, virtual sites included, except dummy beads (type ``U``, radius 0), which have no
SASA. The bead table ``<prefix><molname>_sasa_beads.dat`` lists every bead (position, name,
real or dummy, file); ``ff_assess`` matches its beads to those of the reference by their
position within the molecule (counted from 1), as the bonded interactions, and only checks the
names, so the file names of the two sides need not agree.
"""
import argparse
from collections import ChainMap, namedtuple
from pathlib import Path
import warnings
import numpy as np
import MDAnalysis as mda
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from fast_forward import DATA_PATH
from fast_forward.interaction_distribution import distribution


# a bead of a molecule: position within the molecule (from 1), name for messages
# (``<resid>_<label>``), label (the bead name of the mapping or the itp atom name), dummy (no SASA)
Bead = namedtuple('Bead', 'position name label dummy')
# a row of the bead table: position, label, dummy, file name of the distribution (None for a dummy)
BeadRow = namedtuple('BeadRow', 'position label dummy file')
BEAD_TABLE_HEADER = "# position  name  status  file"

REFERENCE = 'reference' # ff_assess -sasa-surface: the frame closest to the mapped structure of ff_map
FALLBACK_STATISTIC = 'mode' # without a usable mapped structure
# the names of sassy.STATISTICS, kept here so -sasa-stat is checked without importing sassy
STATISTICS = ('median', 'mean', 'mode', 'q1', 'q3', 'min', 'max', 'mean+std', 'mean-std')


class StatisticError(ValueError, argparse.ArgumentTypeError):
    """
    An unknown -sasa-stat (an ArgumentTypeError, so argparse prints the message as is)
    """


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
    if not hasattr(sassy.SASA, 'set_groups'):
        raise ImportError("The SASA calculation (-sasa) needs a newer 'sassy' (with groups of atoms, "
                          "SASA.set_groups). Update it (e.g. 'pip install -U git+https://github.com/ThallmairLab/sassy') "
                          "and try again.")
    return sassy


def surface_statistic(value, allow_reference=False):
    """
    The statistic choosing the representative frame of a surface (``-sasa-stat``): a
    name of ``STATISTICS`` ('median', 'mean', 'mode', 'q1', 'q3', 'min', 'max',
    'mean+std', 'mean-std') or a fixed SASA (nm^2); with ``allow_reference`` also
    'reference' (the frame closest to the mapped structure of ``ff_map``). Used as the
    argparse ``type`` of ``-sasa-stat``.

    Raises
    ------
    StatisticError
        if ``value`` is none of them
    """
    if value in STATISTICS or (allow_reference and value == REFERENCE):
        return value
    try:
        return float(value)
    except ValueError:
        known = [REFERENCE] * allow_reference + list(STATISTICS)
        raise StatisticError(f"unknown statistic {value!r}: use one of {', '.join(known)} "
                         "or a SASA in nm^2") from None


def _citation_map():
    with open(DATA_PATH/'citations.bib') as citation_file:
        ff_citations = read_bib(citation_file)
    return ChainMap(ff_citations, COMMON_CITATIONS)


# sassy.radii_references key: (key in the citation map, message)
RADII_CITATIONS = {'rowland_taylor_1996': ('VDW_radii', "SASA calculation uses atomistic van der Waals radii from"),
                   'bondi_1964': ('VDW_radii_P', "Phosphorus atoms use atomistic van der Waals radii from")}


def cite_aa_radii(ag: mda.AtomGroup):
    """
    Print the citations of the atomistic van der Waals radii used for the SASA of ag
    (see ``sassy.radii_references``): Rowland & Taylor (1996), and Bondi (1964) for
    phosphorus if ag has any.
    """
    sassy = _sassy()
    citation_map = _citation_map()
    for reference in sassy.radii_references(ag, cg=False):
        key, message = RADII_CITATIONS.get(reference.key, (None, f"{reference.scope} from"))
        cite_string = (citation_formatter(citation_map[key]) if key in citation_map
                       else f"{reference.citation}, doi:{reference.doi}")
        print(f"\n{message}:\n{cite_string}\n")


def find_dummies(beads):
    """
    The dummy beads of a molecule: beads whose atoms all belong to other beads, placed from
    the atoms of their neighbours (e.g. the beads without atoms of their own in a CGsmiles
    mapping). They get no SASA; the SASA of every atom is split equally over its remaining
    beads (an atom shared by k of them gives 1/k to each, mapping weights are not used).

    Beads are checked from the largest to the smallest (a dummy built from its neighbours
    holds more atoms than each of them, so it is found before a neighbour could look like a
    dummy because of it); of beads with identical atoms, the first one is kept. Empty beads
    are not dummies.

    Parameters
    ----------
    beads : list of array of int
        atoms of every bead (of one molecule)

    Returns
    -------
    np.ndarray
        (n_beads,) bool, True for the dummy beads
    """
    beads = [np.unique(np.asarray(atoms, dtype=np.int64)) for atoms in beads] # remove duplicates
    n_atoms = max((int(atoms.max()) + 1 for atoms in beads if len(atoms)), default=0)

    # count how many beads every atom belongs to
    count = np.zeros(n_atoms, dtype=np.int64)
    for atoms in beads:
        count[atoms] += 1

    # dummy beads: all atoms also belong to other (remaining) beads
    dummy = np.zeros(len(beads), dtype=bool)
    for b in sorted(range(len(beads)), key=lambda b: (-len(beads[b]), -b)):
        atoms = beads[b]
        if len(atoms) and (count[atoms] > 1).all():
            dummy[b] = True
            count[atoms] -= 1 # count is now the number of non-dummy beads of every atom
    return dummy


def _bead_list(beads):
    """``<position>:<label>`` of every bead, dummies marked, for messages"""
    return ", ".join(f"{bead.position}:{bead.label}{' (dummy)' if bead.dummy else ''}" for bead in beads)


def read_bead_table(path):
    """
    Read a bead table ``<prefix><molname>_sasa_beads.dat`` (see :meth:`MoleculeSASA.save`).

    Returns
    -------
    list of BeadRow
        every bead of the molecule, in the order of the positions 1, 2, ...

    Raises
    ------
    ValueError
        if a line is malformed or the positions are not 1, 2, ...
    """
    rows = []
    with open(path) as table:
        for lineno, line in enumerate(table, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) != 4 or tokens[2] not in ("real", "dummy") or not tokens[0].isdigit():
                raise ValueError(f"{path}, line {lineno}: expected 'position name real|dummy file', got {line.strip()!r}")
            rows.append(BeadRow(int(tokens[0]), tokens[1], tokens[2] == "dummy",
                                None if tokens[3] == "-" else tokens[3]))
    if [row.position for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: the bead positions are not 1, 2, ..., {len(rows)}")
    return rows


class MoleculeSASA:
    """
    SASA of all molecules of one type over a trajectory, computed with a single
    :class:`sassy.SASA` (every molecule a body on its own), which is then also used for
    the surface structures.

    Atomistic (``cg=False``): set the beads of the mapping first (:meth:`set_mapped_beads`,
    optional), then compute the SASA with :meth:`run`. The beads are passed to sassy as
    groups of atoms, so their SASA is summed while the trajectory is read and the SASA of
    every atom in every frame is not kept.

    CG (``cg=True``): every bead is an atom to sassy, so the SASA of every atom is kept
    (``sasa.results.atom_sasa``); after :meth:`run`, :meth:`set_cg_beads` takes the SASA of
    the beads of the itp from it.

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
        the SASA; after :meth:`run` its :class:`sassy.Results` in ``sasa.results``
    series : np.ndarray or None
        (n_molecules * n_frames,) SASA of every molecule in every frame (nm^2), after :meth:`run`
    distribution : np.ndarray or None
        (n_bins, 2) distribution of :attr:`series`, columns: bin centre (nm^2), probability density
    beads : list of Bead or None
        every bead of a molecule, dummies included, in the order of the molecule, after
        :meth:`set_mapped_beads` or :meth:`set_cg_beads`
    bead_names : list of str or None
        ``<resid>_<bead name>`` of every bead with SASA (without the dummies), for messages
    bead_positions : list of int or None
        position within the molecule (from 1) of every bead with SASA
    bead_series : np.ndarray or None
        (n_beads, n_molecules * n_frames) SASA of every bead, the molecules concatenated
        as for :attr:`series`, after :meth:`run` (AA) or :meth:`set_cg_beads` (CG)
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
        # AA: the bead SASA is summed during the run (groups), the SASA of every atom is not kept;
        # CG: every bead is an atom, so the SASA of every atom is the SASA of every bead
        self.sasa = _sassy().SASA(ag, cg=cg, bodies='molecule', keep_atoms=cg, **kwargs)
        self.series = self.distribution = None
        self.beads = self.bead_series = self.bead_distributions = None
        self.molecule_beads = None # (AA) bead indices in the mapped universe of every molecule, from set_mapped_beads

    def run(self, verbose=True):
        """
        Compute the SASA of every molecule and, if set, of its beads, and their distributions.

        Returns
        -------
        MoleculeSASA
            self, e.g. ``MoleculeSASA(ag, name).run()``
        """
        results = self.sasa.run(verbose=verbose, name=self.mol_name)
        if not self.cg: # cite the AA radii papers
            cite_aa_radii(self.sasa.ag)
        self.series = results.body_sasa.ravel() # (n_molecules * n_frames,) SASA of every molecule in every frame
        self.distribution = distribution(self.series, 'sasa', self.mol_name)
        if results.group_sasa is not None: # the beads of set_mapped_beads
            # groups: the real beads of molecule 1, of molecule 2, ... -> one row per bead, the molecules concatenated
            n_beads = len(self.bead_names)
            n_molecules, n_frames = self.sasa.n_bodies, results.n_frames
            self.set_bead_series(self.beads, results.group_sasa.reshape(n_molecules, n_beads, n_frames)
                                 .transpose(1, 0, 2).reshape(n_beads, n_molecules * n_frames))
        return self

    def set_bead_series(self, beads, bead_series):
        """
        Store the beads of a molecule and the SASA of its real beads, and compute their distributions.

        Parameters
        ----------
        beads : list of Bead
            every bead of a molecule, dummies included, in the order of the molecule
        bead_series : np.ndarray
            (n_real_beads, n_molecules * n_frames) SASA of every bead that is not a dummy, in the
            order of ``beads``, the molecules concatenated as for :attr:`series`
        """
        self.beads = list(beads)
        names = self.bead_names
        self.bead_series = np.asarray(bead_series, dtype=float).reshape(len(names), -1)
        self.bead_distributions = [distribution(series, 'sasa', f"{self.mol_name} {name}")
                                   for name, series in zip(names, self.bead_series)]

    def _require_run(self, method):
        if self.series is None:
            raise RuntimeError(f"{method} needs the SASA: call run() first")

    @property
    def molecule_type(self):
        """sassy body type of the molecules (all of ag are of one type)"""
        return self.sasa.body_type_labels[0]

    def file_name(self, suffix, outdir="."):
        """``<outdir>/<prefix><mol_name>_<suffix>``"""
        return Path(outdir) / f"{self.prefix}{self.mol_name}_{suffix}"

    @property
    def bead_names(self):
        return None if self.beads is None else [bead.name for bead in self.beads if not bead.dummy]

    @property
    def bead_positions(self):
        return None if self.beads is None else [bead.position for bead in self.beads if not bead.dummy]

    def bead_file_name(self, bead, suffix, outdir="."):
        """``<outdir>/<prefix><mol_name>_<resid>_<bead name>_<suffix>`` of a :class:`Bead`"""
        return self.file_name(f"{bead.name}_{suffix}", outdir)

    def save(self, dir):
        """
        Save the time series and distribution of the whole molecule and of every bead
        (``<prefix><mol_name>_<resid>_<bead name>_sasa[_distr].dat``), and the bead table
        ``<prefix><mol_name>_sasa_beads.dat``: one line per bead, dummies included, with
        its position, name, status (real or dummy) and distribution file.
        """
        self._require_run("save")
        np.savetxt(self.file_name("sasa.dat", dir), self.series)
        np.savetxt(self.file_name("sasa_distr.dat", dir), self.distribution)
        if self.beads is None:
            return
        real_beads = [bead for bead in self.beads if not bead.dummy]
        for bead, series, distr in zip(real_beads, self.bead_series, self.bead_distributions):
            np.savetxt(self.bead_file_name(bead, "sasa.dat", dir), series)
            np.savetxt(self.bead_file_name(bead, "sasa_distr.dat", dir), distr)
        with open(self.file_name("sasa_beads.dat", dir), "w") as table:
            table.write(BEAD_TABLE_HEADER + "\n")
            for bead in self.beads:
                file = "-" if bead.dummy else self.bead_file_name(bead, "sasa_distr.dat").name
                table.write(f"{bead.position}  {bead.label}  {'dummy' if bead.dummy else 'real'}  {file}\n")

    def set_mapped_beads(self, mapped_atoms, cg_universe, bead_labels=None, ignore=None):
        """
        Set the beads of the mapping (AA), molecule by molecule, before :meth:`run`, which
        splits the SASA of the atoms over them (as groups of atoms, see ``sassy.SASA.set_groups``).

        Every atom contributes equally, regardless of the mapping weights; an atom shared
        by k of the remaining beads gives 1/k of its SASA to each of them. Dummy beads, i.e.
        beads whose atoms all belong to other beads (see :func:`find_dummies`), get no SASA
        and are left out. Atoms in no bead count for the
        molecule only; a warning reports them. Every bead keeps its position within the
        molecule (from 1, dummies included), by which ``ff_assess`` matches it to the beads
        of the CG itp; its label is the bead name of the mapping (``bead_labels``).

        Parameters
        ----------
        mapped_atoms : list of list of int
            AA atom (universe) indices of every bead
        cg_universe : MDAnalysis.Universe
            the mapped universe, bead i built from the AA atoms ``mapped_atoms[i]``
        bead_labels : dict, optional
            {name in the mapped universe: bead name}, e.g. the CGsmiles fragment name of a
            bead (named ``<fragment name><node index>`` in the mapped universe); names not in
            it are used as they are
        ignore : list of int, optional
            positions within the molecule (from 1, ``ff_map -sasa-ignore``) of the beads that get
            no SASA: they replace the automatic detection of dummy beads, exactly these beads are
            dummies (in every molecule)

        Raises
        ------
        ValueError
            if the molecules do not all have the same beads in the same order, if a position of
            ``ignore`` is not a bead of the molecule, or for a CG SASA
        """
        if self.cg:
            raise ValueError("set_mapped_beads splits the SASA of atoms over the beads of a mapping: "
                             "it needs an atomistic SASA (cg=False)")
        ag = self.sasa.ag # atomistic ag

        row = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64)
        row[ag.indices] = np.arange(ag.n_atoms) # universe index -> position in ag, -1 for atoms not in ag

        # molecule index of each atom
        molecule_rows = self.sasa.body_members
        molecule_of_row = np.empty(ag.n_atoms, dtype=np.int64)
        for m, rows in enumerate(molecule_rows):
            molecule_of_row[rows] = m

        # beads built only from the atoms of one molecule of this type, in the order of the mapping
        mol_beads = [[] for _ in molecule_rows]
        for bdx, atoms in enumerate(mapped_atoms):
            rows = row[np.asarray(list(atoms), dtype=np.int64)]
            if len(rows) and (rows >= 0).all() and (molecule_of_row[rows] == molecule_of_row[rows[0]]).all(): # all atoms of this bead belong to one molecule of this type
                mol_beads[molecule_of_row[rows[0]]].append((bdx, rows))

        # every bead of every molecule, dummies included (write_mapped_structure)
        self.molecule_beads = [[bdx for bdx, _ in beads] for beads in mol_beads]

        bead_labels = bead_labels or {}
        mol_table, groups = None, []
        for m, beads in enumerate(mol_beads):
            bdxs = [bdx for bdx, _ in beads]
            if ignore is None:
                dummy = find_dummies([rows for _, rows in beads])
            else:
                outside = [index for index in ignore if index >= len(beads)]
                if outside:
                    raise ValueError(f"-sasa-ignore: {self.mol_name} has {len(beads)} bead(s), so there is no bead at "
                                     f"position(s) {', '.join(map(str, outside))}")
                dummy = np.isin(np.arange(1, len(beads) + 1), ignore)
            resindices = cg_universe.atoms.resindices[bdxs] if bdxs else np.zeros(0, dtype=int)
            labels = [bead_labels.get(name, name) for name in cg_universe.atoms[bdxs].names] if bdxs else []
            table = [Bead(index, f"{res - resindices.min() + 1}_{label}", label, bool(d))
                     for index, (res, label, d) in enumerate(zip(resindices, labels, dummy), 1)]
            if mol_table is None:
                mol_table = table
                skipped = [bead.name for bead in table if bead.dummy]
                if skipped and ignore is not None:
                    print(f"INFO - No SASA assigned to the bead(s) {', '.join(skipped)} of {self.mol_name}, "
                          f"ignored with -sasa-ignore")
                elif skipped:
                    print(f"INFO - No SASA assigned to the dummy bead(s) {', '.join(skipped)} of "
                          f"{self.mol_name}: all their atoms belong to other beads")
            elif table != mol_table:
                raise ValueError(f"All {self.mol_name} molecules must have the same beads in the same order "
                                 f"for the bead SASA: molecule 1 has {_bead_list(mol_table)}, molecule {m + 1} "
                                 f"has {_bead_list(table)}.")
            groups += [rows for (_, rows), d in zip(beads, dummy) if not d]
        self.beads = mol_table or []
        self.bead_series = self.bead_distributions = None
        self.sasa.set_groups(groups, share='equal')

        unassigned = ag[self.sasa.ungrouped]
        if unassigned.n_atoms:
            listed = list(dict.fromkeys(f"{atom.resname}{atom.resid}:{atom.name}" for atom in unassigned))
            more = f" and {len(listed) - 10} more" if len(listed) > 10 else ""
            print(f"WARNING - {unassigned.n_atoms} atom(s) of {self.mol_name} are in no bead of the mapping; their "
                  f"SASA counts for the molecule but not for any bead: {', '.join(listed[:10])}{more}")

    def set_cg_beads(self, block):
        """
        Set the CG beads of the itp ``block`` and their SASA, after :meth:`run`, from the SASA of
        the atoms (every bead is an atom to sassy): every bead except dummy beads (radius 0),
        virtual sites included. Every bead is named ``<resid>_<atomname>`` as in the itp and
        keeps its index within the molecule (from 1, dummies included), by which it is
        matched to the reference. Sets :attr:`beads` and the bead SASA (:meth:`set_bead_series`).

        Parameters
        ----------
        block : vermouth.molecule.Block
            the molecule type, its atoms in the order of every molecule of ag

        Raises
        ------
        ValueError
            if a molecule has a different number of beads than ``block``
        """
        results = self.sasa.results
        if results is None or results.atom_sasa is None:
            raise ValueError("set_cg_beads needs the SASA of every bead: a CG MoleculeSASA (cg=True) after run()")
        nodes = list(block.nodes) # atoms from the ff block, in the order of the molecule
        labels = [block.nodes[node]['atomname'] for node in nodes]
        names = [f"{block.nodes[node]['resid']}_{label}" for node, label in zip(nodes, labels)]
        molecule_rows = self.sasa.body_members # positions in ag (rows of atom_sasa) of the atoms of every molecule

        # all molecules have the same number of beads as the block
        for m, rows in enumerate(molecule_rows):
            if len(rows) != len(nodes):
                raise ValueError(f"Molecule {m + 1} of {self.mol_name} has {len(rows)} beads, but the itp "
                                 f"has {len(nodes)}.")

        # leave out dummy beads (radius 0), which have no SASA
        radii = self.sasa.radii[molecule_rows[0]] if molecule_rows else np.zeros(len(nodes))
        dummy = radii <= 0
        if dummy.any():
            print(f"INFO - No SASA saved for the dummy bead(s) (radius 0) "
                  f"{', '.join(name for name, d in zip(names, dummy) if d)} of {self.mol_name}")

        # one row per real bead, the molecules concatenated (all frames of the first, then of the second, ...)
        keep = np.flatnonzero(~dummy)
        series = (np.concatenate([results.atom_sasa[rows[keep]] for rows in molecule_rows], axis=1)
                  if molecule_rows else np.zeros((len(keep), 0)))
        beads = [Bead(i + 1, names[i], labels[i], bool(dummy[i])) for i in range(len(nodes))]
        self.set_bead_series(beads, series)

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
        self._require_run("write_surface")
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

    def write_mapped_structure(self, surface, cg_universe, outdir="."):
        """
        Write the mapped (CG) structure of the AA molecule and frame of ``surface`` as
        ``<outdir>/<prefix><mol_name>_sasa_mapped.pdb``; ``ff_assess -sasa-surface``
        uses it to pick the matching CG frame.

        Parameters
        ----------
        surface : sassy.Surface
            AA surface from :meth:`write_surface`
        cg_universe : MDAnalysis.Universe
            the mapped universe given to :meth:`set_mapped_beads`, which must be called first

        Raises
        ------
        RuntimeError
            if :meth:`set_mapped_beads` was not called
        """
        if self.molecule_beads is None:
            raise RuntimeError("write_mapped_structure needs the beads of every molecule: call set_mapped_beads first")
        beads = self.molecule_beads[surface.body]
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
