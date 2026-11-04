"""
Dot representation of the solvent accessible surface, for visual comparison of
atomistic, mapped and coarse-grained structures.
"""
from pathlib import Path
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.util import NamedStream
from MDAnalysis.analysis.align import rotation_matrix
from vermouth.file_writer import deferred_open
from fast_forward.sasa import PROBE_RADIUS, atom_radii, sasa_citation_remarks, split_molecules, surface_dots

def _write_pdb(fname, positions, names, resnames, resids, chains, record_types, remarks):
    """
    Write per-record arrays as a PDB file with MDAnalysis, through vermouth's
    DeferredFileWriter. Positions are in Å.
    """
    n_records = len(positions)
    # one residue per consecutive (chain, resid, resname) block
    keys = list(zip(chains, resids, resnames))
    resindex, res_keys = [], {}
    for key in keys:
        resindex.append(res_keys.setdefault(key, len(res_keys)))
    universe = mda.Universe.empty(n_records, n_residues=len(res_keys), atom_resindex=resindex,
                                  trajectory=True)
    universe.add_TopologyAttr('names', list(names))
    universe.add_TopologyAttr('resnames', [key[2] for key in res_keys])
    universe.add_TopologyAttr('resids', [int(key[1]) for key in res_keys])
    universe.add_TopologyAttr('chainIDs', list(chains))
    universe.add_TopologyAttr('record_types', list(record_types))
    # explicit defaults for the remaining PDB fields, so the writer does not warn about them
    universe.add_TopologyAttr('altLocs', [''] * n_records)
    universe.add_TopologyAttr('icodes', [''] * len(res_keys))
    universe.add_TopologyAttr('segids', [''])
    universe.add_TopologyAttr('occupancies', np.ones(n_records))
    universe.add_TopologyAttr('tempfactors', np.zeros(n_records))
    universe.add_TopologyAttr('elements', [''] * n_records)
    universe.add_TopologyAttr('formalcharges', np.zeros(n_records, dtype=int))
    universe.atoms.positions = positions
    universe.trajectory.remarks = list(remarks)
    with deferred_open(fname, 'w') as handle, warnings.catch_warnings():
        # there is no meaningful box for a single (superposed) molecule: the
        # standard unitary CRYST1 record is intended
        warnings.filterwarnings('ignore', message='Unit cell dimensions not found')
        with mda.Writer(NamedStream(handle, str(fname)), n_atoms=n_records) as writer:
            writer.write(universe.atoms)

def write_structure_pdb(ag, positions, fname, remarks=None):
    """
    Write the particles of ``ag`` at ``positions`` (Å) as a PDB file, with
    ``remarks`` as REMARK records.
    Written with vermouth's DeferredFileWriter: call ``DeferredFileWriter().write()`` to finalise.
    """
    _write_pdb(fname, positions=positions, names=ag.names, resnames=ag.resnames, resids=ag.resids,
               chains=['A'] * len(positions), record_types=['ATOM'] * len(positions),
               remarks=["Mapped structure of the SASA surface frame (fast_forward)"] + list(remarks or []))

def closest_to_median(values):
    """
    Index of the value closest to the median, i.e. a representative frame.
    """
    values = np.asarray(values)
    return int(np.argmin(np.abs(values - np.median(values))))

def best_matching_frame(positions, reference):
    """
    Frame of ``positions`` (n_frames, n_atoms, 3) with the lowest RMSD to
    ``reference`` (n_atoms, 3) after optimal superposition, and that frame
    superposed onto the reference.

    Returns
    -------
    frame : int
    aligned : np.ndarray
        positions of the best frame superposed onto ``reference``
    rmsd : float
        RMSD (same units as the input) of the aligned frame
    """
    ref_com = reference.mean(axis=0)
    ref = (reference - ref_com).astype(np.float64)
    best = (None, np.inf, None)
    for frame, pos in enumerate(positions):
        mobile = (pos - pos.mean(axis=0)).astype(np.float64)
        rot, rmsd = rotation_matrix(mobile, ref)
        if rmsd < best[1]:
            best = (frame, rmsd, mobile @ np.asarray(rot).T + ref_com)
    frame, rmsd, aligned = best
    return frame, aligned, float(rmsd)

def write_sasa_surface(ag, positions, fname, cg=False, ndots=240):
    """
    Write the structure in ``ag`` at ``positions`` (Å) together with its solvent
    accessible surface dots to a PDB file and return the surface area in nm^2.

    The particles come first as ATOM records (chain A); the dots follow as
    HETATM records (chain X) with atom and residue name DOT, carrying the
    residue id of the particle they belong to. The citations of fast_forward and the radii are added as REMARK records.
    Written with vermouth's DeferredFileWriter: call ``DeferredFileWriter().write()`` to finalise.
    """
    dots, owners, area = surface_dots(positions / 10.0, atom_radii(ag, cg=cg),
                                      probe=PROBE_RADIUS, ndots=ndots)
    n_atoms, n_dots = len(positions), len(dots)
    _write_pdb(fname,
               positions=np.vstack([positions, dots * 10]),  # nm -> Å
               names=np.concatenate([ag.names, ['DOT'] * n_dots]),
               resnames=np.concatenate([ag.resnames, ['DOT'] * n_dots]),
               resids=np.concatenate([ag.resids, ag.resids[owners]]),
               chains=['A'] * n_atoms + ['X'] * n_dots,
               record_types=['ATOM'] * n_atoms + ['HETATM'] * n_dots,
               remarks=["Structure (ATOM) and solvent accessible surface dots (HETATM)"]
                       + sasa_citation_remarks(ag, cg=cg))
    return area

def write_aa_sasa_surface(mol_atoms, mol_name, sasa, outdir=".", prefix="", ndots=240):
    """
    Write the atomistic structure and surface dots of the first molecule of
    ``mol_atoms`` in the frame whose SASA is closest to the median, as
    ``<outdir>/<prefix><mol_name>_sasa_dots_AA.pdb``.

    ``sasa`` is the SASA series of ``mol_atoms`` from ``calc_SASA``; the values
    of the first molecule are reused to select the frame, so the SASA is not
    computed twice.

    Returns
    -------
    tuple(int, set)
        the selected frame and the indices of the atoms of that molecule
    """
    molecule = split_molecules(mol_atoms)[0]
    # the first n_frames values of the concatenated series belong to the first molecule
    frame = closest_to_median(sasa[:molecule.universe.trajectory.n_frames])
    trajectory = molecule.universe.trajectory
    current = trajectory.ts.frame
    trajectory[frame]
    positions = molecule.positions.copy()
    trajectory[current]  # leave the universe on the frame it was on
    print(f"INFO - Writing atomistic SASA surface of frame {frame} for {mol_name}")
    area = write_sasa_surface(molecule, positions, Path(outdir) / f"{prefix}{mol_name}_sasa_dots_AA.pdb",
                              cg=False, ndots=ndots)
    print(f"INFO - SASA of the written {mol_name} surface: {area:.3f} nm^2")
    return frame, set(molecule.indices.tolist())

def write_cg_sasa_surface(mol_atoms, molname, reference, sasa, outdir=".", prefix="", ndots=240):
    """
    Write the surface dots of a simulated CG structure that is as similar as
    possible to the mapped atomistic structure written by 'ff_map -sasa-surface'
    (``<reference>/<prefix><molname>_sasa_mapped.pdb``), as
    ``<outdir>/<prefix><molname>_sasa_dots_CG.pdb``.

    The frame of the first molecule with the lowest RMSD to the mapped structure
    is superposed onto it, so that the CG and atomistic surfaces share the same
    coordinate frame. Without a mapped structure in the reference directory,
    the frame whose SASA is closest to the median is used; the values of the
    first molecule in ``sasa`` (the SASA series of ``mol_atoms`` from
    ``calc_SASA``) are reused for this.
    """
    molecule = split_molecules(mol_atoms)[0]
    positions = np.array([molecule.positions.copy() for _ in molecule.universe.trajectory])
    mapped_file = Path(reference) / f"{prefix}{molname}_sasa_mapped.pdb"

    mapped = None
    if mapped_file.is_file():
        # only bead names and positions are needed; silence MDAnalysis' warnings about the
        # missing elements/masses and the placeholder unit cell of this file
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mapped = mda.Universe(str(mapped_file)).atoms
        if mapped.n_atoms != molecule.n_atoms:
            print(f"WARNING - {mapped_file} has {mapped.n_atoms} beads, but {molname} has "
                  f"{molecule.n_atoms}; ignoring the mapped structure.")
            mapped = None
        elif list(mapped.names) != list(molecule.names):
            print(f"WARNING - bead names in {mapped_file} differ from those of {molname}; "
                  "matching beads by their order.")
    else:
        print(f"WARNING - {mapped_file} not found (run 'ff_map -sasa -sasa-surface' to create it); "
              f"using the frame of {molname} whose SASA is closest to the median.")

    if mapped is not None:
        frame, frame_positions, rmsd = best_matching_frame(positions, mapped.positions)
        print(f"INFO - Frame {frame} of {molname} is closest to the mapped structure "
              f"(RMSD {rmsd / 10:.3f} nm); superposing it onto the mapped structure.")
    else:
        frame = closest_to_median(sasa[:molecule.universe.trajectory.n_frames])
        frame_positions = positions[frame]

    area = write_sasa_surface(molecule, frame_positions, Path(outdir) / f"{prefix}{molname}_sasa_dots_CG.pdb",
                              cg=True, ndots=ndots)
    print(f"INFO - SASA of the CG {molname} surface (frame {frame}): {area:.3f} nm^2")
