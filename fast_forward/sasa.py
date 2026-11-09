"""
Solvent accessible surface area (SASA) of molecules over a trajectory.

The SASA itself is computed with ``fastsasa`` (Shrake-Rupley,
every molecule on its own, so the SASA describes the shape of a molecule and not its
exposure in the surrounding system):

- ``ff_map -sasa`` computes the SASA of the atomistic (AA) trajectory, with the
  atomistic van der Waals radii, and saves its distribution as the reference.
- ``ff_assess -sasa`` computes the SASA of the coarse-grained (CG) trajectory, with
  the Martini bead radii, and scores it against that reference.

Time series and distributions are written in the same format as the interaction
distributions (``<prefix><molname>_sasa.dat`` and ``<prefix><molname>_sasa_distr.dat``),
so the reference SASA lives next to the reference bonded distributions.
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


def _fastsasa():
    """
    Import fastsasa on first use, so that fast_forward works without it unless
    the SASA is requested.
    """
    try:
        import fastsasa
    except ImportError as err:
        msg = ("The SASA calculation (-sasa) requires the 'fastsasa' package, which is not "
               "installed. Install it (e.g. 'pip install /path/to/fastsasa') and try again.")
        raise ImportError(msg) from err
    return fastsasa


def _citation_map():
    with open(DATA_PATH/'citations.bib') as citation_file:
        ff_citations = read_bib(citation_file)
    return ChainMap(ff_citations, COMMON_CITATIONS)


def cite_aa_radii(ag: mda.AtomGroup):
    """
    Print the citations of the atomistic van der Waals radii used for the SASA of
    ``ag``: Rowland & Taylor (1996), and Bondi (1964) for phosphorus if ``ag`` has any.
    """
    fastsasa = _fastsasa()
    citation_map = _citation_map()
    cite_string = citation_formatter(citation_map['VDW_radii'])
    print(f"\nSASA calculation uses atomistic van der Waals radii from:\n{cite_string}\n")
    if 'P' in fastsasa.atom_symbols(ag, cg=False):
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


class MoleculeSASA:
    """
    SASA of all molecules of one type over a trajectory, computed once with a single
    :class:`fastsasa.SASA`, which is then also used for the surface structures.

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
        minimum number of surface dots per particle (default: that of fastsasa)

    Attributes
    ----------
    sasa : fastsasa.SASA
        the SASA after its run
    series : np.ndarray
        (n_molecules * n_frames,) SASA of every molecule in every frame (nm^2)
    distribution : np.ndarray
        (n_bins, 2) distribution of :attr:`series`, columns: bin centre (nm^2), probability density
    """
    def __init__(self, ag: mda.AtomGroup, mol_name: str, cg: bool = False, prefix: str = "",
                 ndots: int | None = None):
        if ag.n_atoms == 0:
            raise ValueError(f"No atoms selected for the SASA of {mol_name}.")
        self.mol_name = mol_name
        self.cg = cg
        self.prefix = prefix
        kwargs = {} if ndots is None else {'ndots': ndots}
        self.sasa = _fastsasa().SASA(ag, cg=cg, per_molecule=True, save_atoms=False, **kwargs)
        self.sasa.run()
        if not cg: # cite the AA radii papers
            cite_aa_radii(ag)
        self.series = np.asarray(self.sasa.molecule_sasa).ravel()
        self.distribution = distribution(self.series, 'sasa', mol_name)

    @property
    def molecule_type(self):
        """fastsasa type of the molecules (all of ``ag`` are of one type)"""
        return self.sasa.molecule_types[0]

    def file_name(self, suffix, outdir="."):
        """``<outdir>/<prefix><mol_name>_<suffix>``"""
        return Path(outdir) / f"{self.prefix}{self.mol_name}_{suffix}"

    def save(self, dir):
        """
        Save the time series and distribution as ``<prefix><mol_name>_sasa.dat`` and
        ``<prefix><mol_name>_sasa_distr.dat`` in ``dir``.
        """
        np.savetxt(self.file_name("sasa.dat", dir), self.series)
        np.savetxt(self.file_name("sasa_distr.dat", dir), self.distribution)

    def write_surface(self, outdir=".", reference=None):
        """
        Write the structure and its surface dots of one molecule as
        ``<outdir>/<prefix><mol_name>_sasa_dots_<AA|CG>.pdb``.

        Without ``reference``, the molecule and frame whose SASA is closest to the median
        of the molecule type are used, from the SASA already computed. With ``reference``
        (an AtomGroup or (n_atoms, 3) positions in nm, as many atoms as one molecule), the
        frame of the first molecule closest to it (lowest RMSD) is used, superposed onto it.
        Only the surface dots of the written frame are computed (fewer dots per particle
        than for the SASA, as they are for viewing).

        Returns
        -------
        fastsasa.Surface
            the written surface (structure, dots, area in nm^2 from the dots, frame, ...)
        """
        fname = self.file_name(f"sasa_dots_{'CG' if self.cg else 'AA'}.pdb", outdir)
        surface = self.sasa.write_surface(fname, molecule_type=self.molecule_type, reference=reference,
                                          return_surface=True)

		# If a reference is given, report the RMSD to it
        matched = f", RMSD {surface.rmsd:.3f} nm to the mapped structure" if surface.rmsd is not None else ""
        print(f"INFO - Wrote SASA surface of {self.sasa.molecule_labels[surface.molecule]} in frame "
              f"{surface.frame} ({surface.area:.2f} nm^2{matched}) to {fname}")
        return surface

    def write_mapped_structure(self, surface, cg_universe, mapped_atoms, outdir="."):
        """
        Write the mapped (CG) structure of the AA molecule and frame of ``surface`` as
        ``<outdir>/<prefix><mol_name>_sasa_mapped.pdb``; ``ff_assess -sasa-surface``
        uses it to pick the matching CG frame.

        Parameters
        ----------
        surface : fastsasa.Surface
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

    def write_cg_surface(self, reference_dir=None, outdir="."):
        """
        Write the CG surface (see :meth:`write_surface`) in the frame closest to the mapped
        structure written by ``ff_map -sasa-surface``
        (``<reference_dir>/<prefix><mol_name>_sasa_mapped.pdb``), so that the AA and CG
        surfaces share one coordinate frame; without a matching mapped structure, in the
        frame with median SASA.
        """
        reference = None
        mapped_file = self.file_name("sasa_mapped.pdb", reference_dir) if reference_dir else None
        if mapped_file is not None and mapped_file.is_file():
            reference = mda.Universe(str(mapped_file)).atoms
            n_atoms = self.sasa.molecule_atoms(int(self.sasa.types[self.molecule_type][0])).n_atoms
            if reference.n_atoms != n_atoms:
                print(f"WARNING - The mapped structure {mapped_file} has {reference.n_atoms} beads, but "
                      f"{self.mol_name} has {n_atoms}; writing the CG surface of the frame closest to the "
                      "median SASA instead.")
                reference = None
        else:
            print(f"INFO - No mapped structure {mapped_file} found; writing the CG surface of the frame "
                  "closest to the median SASA instead.")
        return self.write_surface(outdir, reference=reference)
