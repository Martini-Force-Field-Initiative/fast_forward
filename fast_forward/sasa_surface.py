"""
Dot representation of the solvent accessible surface, for visual comparison of
atomistic, mapped and coarse-grained structures.

The unit-sphere dots are generated as in ``gmx sasa``.
"""
from functools import lru_cache
from pathlib import Path
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.util import NamedStream
from MDAnalysis.analysis.align import rotation_matrix
from vermouth.file_writer import deferred_open
from scipy.spatial import cKDTree
from fast_forward.sasa import (PROBE_RADIUS, atom_radii, sasa_citation_remarks, sasa_per_molecule,
                               split_molecules)

def _divarc(p1, p2, div1, div2):
    """Point at fraction div1/div2 along the great-circle arc p1 -> p2 (gmx divarc)."""
    axis = np.cross(p1, p2)
    phi = np.arctan2(np.linalg.norm(axis), np.dot(p1, p2)) * div1 / div2
    axis /= np.linalg.norm(axis)
    p1 = p1 / np.linalg.norm(p1)
    return np.cos(phi) * p1 + np.sin(phi) * np.cross(axis, p1)

def _icosahedron():
    c72 = np.cos(np.radians(72.0))
    rh = np.sqrt(1.0 - 2.0 * c72) / (1.0 - c72)
    rg = c72 / (1.0 - c72)
    v = [[0, 0, 1]]
    v += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), rg] for a in (72, 144, 216, 288, 0)]
    v += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), -rg] for a in (36, 108, 180, 252, 324)]
    v += [[0, 0, -1]]
    return np.array(v, float)

def _fill_face(vi, vj, vk, tess):
    """Interior dots of the spherical triangle (vi, vj, vk) at tessellation level tess."""
    out = []
    for tl in range(1, tess - 1):
        pji = _divarc(vj, vi, tl, tess)
        pki = _divarc(vk, vi, tl, tess)
        for tl2 in range(1, tess - tl):
            pij = _divarc(vi, vj, tl2, tess)
            pkj = _divarc(vk, vj, tl2, tess)
            pik = _divarc(vi, vk, tess - tl - tl2, tess)
            pjk = _divarc(vj, vk, tess - tl - tl2, tess)
            p = (_divarc(pki, pji, tl2, tess - tl)
                 + _divarc(pkj, pij, tl, tess - tl2)
                 + _divarc(pjk, pik, tl, tl + tl2))
            out.append(p / np.linalg.norm(p))
    return out

def _d2(a, b):
    return np.sum((a - b) ** 2)

def ico_dot_arc(densit):
    """Geodesic icosahedron: 10*tess^2 + 2 dots."""
    tess = max(int(np.ceil(np.sqrt((densit - 2.0) / 10.0))), 1)
    v = _icosahedron()
    pts = list(v)
    if tess > 1:
        a = _d2(v[0], v[1])                       # squared icosahedron edge
        e = lambda i, j: abs(_d2(v[i], v[j]) - a) < 1e-4
        for i in range(11):                       # 30 edges
            for j in range(i + 1, 12):
                if e(i, j):
                    pts += [_divarc(v[i], v[j], tl, tess) for tl in range(1, tess)]
        for i in range(10):                       # 20 faces
            for j in range(i + 1, 11):
                for k in range(j + 1, 12):
                    if e(i, j) and e(i, k) and e(j, k):
                        pts += _fill_face(v[i], v[j], v[k], tess)
    return np.array(pts)

def ico_dot_dod(densit):
    """Geodesic pentakis-dodecahedron: 30*tess^2 + 2 dots."""
    tess = max(int(np.ceil(np.sqrt((densit - 2.0) / 30.0))), 1)
    v = _icosahedron()
    a = _d2(v[0], v[1])
    e = lambda i, j: abs(_d2(v[i], v[j]) - a) < 1e-4
    cen = []                                      # 20 face centres = dodecahedron vertices
    for i in range(10):
        for j in range(i + 1, 11):
            for k in range(j + 1, 12):
                if e(i, j) and e(i, k) and e(j, k):
                    c = v[i] + v[j] + v[k]
                    cen.append(c / np.linalg.norm(c))
    w = np.vstack([v, cen])                       # 32 vertices
    pts = list(w)
    if tess > 1:
        ai_d = min(_d2(w[0], w[j]) for j in range(12, 32))   # ico vertex - face centre
        adod = min(_d2(w[12], w[j]) for j in range(13, 32))  # dodecahedron edge
        def is_e(i, j):
            if i < 12 and j < 12:
                return False
            return abs(_d2(w[i], w[j]) - (ai_d if (i < 12) != (j < 12) else adod)) < 1e-4
        for i in range(31):                       # 90 edges
            for j in range(max(12, i + 1), 32):
                if is_e(i, j):
                    pts += [_divarc(w[i], w[j], tl, tess) for tl in range(1, tess)]
        for i in range(12):                       # 60 faces (vertex, centre, centre)
            for j in range(12, 31):
                for k in range(j + 1, 32):
                    if is_e(i, j) and is_e(i, k) and is_e(j, k):
                        pts += _fill_face(w[i], w[j], w[k], tess)
    return np.array(pts)

@lru_cache(maxsize=None)
def gmx_unit_dots(ndots):
    """
    Dots as gmx sasa -ndots: whichever scheme gives fewer dots >= ndots
    (e.g. 24 -> 32 (dod), 50 -> 92 (arc), 100 -> 122 (dod), 240 -> 252 (arc)).
    The array is cached and read-only.
    """
    arc, dod = ico_dot_arc(ndots), ico_dot_dod(ndots)
    unit = arc if len(arc) <= len(dod) else dod
    unit.flags.writeable = False
    return unit

def surface_dots(xyz, radii, probe=0.191, ndots=240):
    """
    Generate dots of the solvent accessible surface of a set of spheres.

    Parameters
    ----------
    xyz : np.ndarray
        (n_atoms, 3) coordinates in nm.
    radii : np.ndarray
        (n_atoms,) van der Waals radii in nm. Particles with radius 0 (dummy
        beads, virtual sites) neither carry dots nor bury dots of others.
    probe : float
        probe radius in nm
    ndots : int
        minimum number of dots per sphere

    Returns
    -------
    dots : np.ndarray
        (n_dots, 3) dot positions in nm
    owners : np.ndarray
        (n_dots,) index of the particle each dot belongs to
    area : float
        solvent accessible surface area in nm^2 estimated from the dots
    """
    xyz = np.asarray(xyz, dtype=float)
    radii = np.asarray(radii, dtype=float)
    unit = gmx_unit_dots(int(ndots))
    n_pts = len(unit)

    active = np.flatnonzero(radii > 0)
    if active.size == 0:
        return np.zeros((0, 3)), np.zeros(0, int), 0.0
    ext = radii[active] + probe
    centres = xyz[active]
    # all candidate dots on the probe-expanded spheres
    pts = (centres[:, None, :] + ext[:, None, None] * unit[None]).reshape(-1, 3)
    owner_local = np.repeat(np.arange(active.size), n_pts)

    # a dot is buried if it lies strictly inside the expanded sphere of another particle
    keep = np.ones(len(pts), bool)
    tree = cKDTree(pts)
    for j, hits in enumerate(tree.query_ball_point(centres, ext)):
        if not hits:
            continue
        hits = np.asarray(hits)
        hits = hits[owner_local[hits] != j]
        if hits.size:
            d = np.linalg.norm(pts[hits] - centres[j], axis=1)
            keep[hits[d < ext[j]]] = False

    keep = keep.reshape(active.size, n_pts)
    area = float(np.sum(keep.sum(axis=1) / n_pts * 4 * np.pi * ext**2))

    dots = pts.reshape(active.size, n_pts, 3)[keep]
    owners = active[np.nonzero(keep)[0]]
    return dots, owners, area

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

def _first_molecule_sasa(molecule, sasa, cg):
    """
    SASA of ``molecule``, the first molecule of a selection, in every frame.
    ``sasa`` is the concatenated SASA series of that selection from ``calc_SASA``,
    whose first n_frames values belong to the first molecule; without it the
    SASA is computed.
    """
    if sasa is not None:
        return sasa[:molecule.universe.trajectory.n_frames]
    return sasa_per_molecule(molecule, cg=cg)[0]

def write_aa_sasa_surface(mol_atoms, mol_name, outdir=".", prefix="", ndots=240, sasa=None):
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
    frame = closest_to_median(_first_molecule_sasa(molecule, sasa, cg=False))
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

def write_cg_sasa_surface(mol_atoms, molname, reference, outdir=".", prefix="", ndots=240, sasa=None):
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
        frame = closest_to_median(_first_molecule_sasa(molecule, sasa, cg=True))
        frame_positions = positions[frame]

    area = write_sasa_surface(molecule, frame_positions, Path(outdir) / f"{prefix}{molname}_sasa_dots_CG.pdb",
                              cg=True, ndots=ndots)
    print(f"INFO - SASA of the CG {molname} surface (frame {frame}): {area:.3f} nm^2")
