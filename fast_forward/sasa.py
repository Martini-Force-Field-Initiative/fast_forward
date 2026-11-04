import numpy as np
try:
	import mdtraj as md
except ImportError:
	md = None
from . import DATA_PATH
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from collections import ChainMap
from fast_forward.interaction_distribution import INTERACTIONS
import MDAnalysis as mda

AA_RADII = {
	'H': 0.109,
	'C': 0.175,
	'N': 0.161,
	'O': 0.156,
	'F': 0.144,
	'P': 0.180,
	'S': 0.179,
	'Cl': 0.174,
	'Br': 0.185,
	'I': 0.200,
	}

CG_RADII= {
'T': 0.191, # tiny bead
'S': 0.230, # small bead
'R': 0.264, # regular bead
'U': 0.000, # Dummy bead, no radius
'VS': 0.000,# Virtual site, no radius
}

def check_mdtraj():
	'''
	Raise an ImportError if mdtraj, which is needed for SASA calculations, is not installed.
	'''
	if md is None:
		raise ImportError("SASA calculations require mdtraj, which is not installed. "
						  "Install it with `pip install mdtraj`.")

PROBE_RADIUS = 0.191 # nm
N_SPHERE_POINTS = 4800

def atom_symbols(ag: mda.AtomGroup, cg: bool = False):
	'''
	Symbols used to look up the radius of each particle
	- respective element for AA
	- bead size (R = regular, S = small, T = tiny, U = dummy) derived from atom type for CG.
	'''
	if cg and not hasattr(ag.atoms, "elements"):
		return [at[0] if at[0] in ("S", "T", "U") else "R" for at in ag.types]
	elements = ag.elements if hasattr(ag.atoms, "elements") else [None] * ag.n_atoms
	return [el if el else a.name[0] for el, a in zip(elements, ag)]

def atom_radii(ag: mda.AtomGroup, cg: bool = False):
	'''
	Van der Waals radii (nm) of the particles in ``ag``, as used for the SASA.
	'''
	table = CG_RADII if cg else AA_RADII
	symbols = atom_symbols(ag, cg=cg)
	missing = sorted({sym for sym in symbols if sym not in table})
	if missing:
		raise KeyError(f"No {'CG' if cg else 'atomistic'} SASA radius known for: {', '.join(missing)}")
	return np.array([table[sym] for sym in symbols])

def _molecule_keys(ag: mda.AtomGroup):
	'''
	Molecule of every atom in ``ag``: its molnum, or its residue if there are no molnums.
	'''
	return ag.molnums if hasattr(ag.atoms, 'molnums') else ag.resindices

def split_molecules(ag: mda.AtomGroup):
	'''
	Split ``ag`` into its molecules (by molnum, or by residue if there are no
	molnums), in the order in which they appear.
	'''
	keys = _molecule_keys(ag)
	_, first = np.unique(keys, return_index=True)
	return [ag[keys == keys[i]] for i in sorted(first)]

def sasa_per_molecule(ag: mda.AtomGroup, cg: bool = False):
	'''
	SASA (nm^2) of every molecule in ``ag`` in every frame of its trajectory,
	each molecule taken on its own (other molecules do not bury its surface).

	Returns
	-------
	np.ndarray
		(n_molecules, n_frames), molecules in the order of :func:`split_molecules`
	'''
	check_mdtraj()
	traj = mda_to_mdtraj(ag, cg=cg)
	radii = CG_RADII if cg else AA_RADII
	keys = _molecule_keys(ag)
	_, first = np.unique(keys, return_index=True)
	sasa = []
	for i in sorted(first):
		molecule = traj.atom_slice(np.flatnonzero(keys == keys[i]))
		sasa.append(md.shrake_rupley(molecule, probe_radius=PROBE_RADIUS, n_sphere_points=N_SPHERE_POINTS,
									 change_radii=radii, mode='atom').sum(axis=1))
	return np.array(sasa)

def _citation_map():
	with open(DATA_PATH/'citations.bib') as citation_file:
		ff_citations = read_bib(citation_file)
	return ChainMap(ff_citations, COMMON_CITATIONS)

def sasa_citation_remarks(ag: mda.AtomGroup = None, cg: bool = False):
	'''
	Citations to put in PDB files written for the SASA surface: fast_forward
	and, if ``ag`` is given, the radii used for the surface.
	'''
	citation_map = _citation_map()
	remarks = [f"Generated with fast_forward: {citation_formatter(citation_map['fast_forward'])}"]
	if ag is not None:
		if cg:
			remarks.append(f"CG radii: tiny {CG_RADII['T']}, small {CG_RADII['S']}, regular {CG_RADII['R']} nm; "
						   f"probe radius {PROBE_RADIUS} nm")
		else:
			remarks.append(f"Atomistic van der Waals radii: {citation_formatter(citation_map['VDW_radii'])}")
			if 'P' in atom_symbols(ag, cg=False):
				remarks.append(f"Phosphorus radius: {citation_formatter(citation_map['VDW_radii_P'])}")
			remarks.append(f"Probe radius {PROBE_RADIUS} nm")
	return remarks

def mda_to_mdtraj(ag: mda.AtomGroup, cg: bool = False):
	'''
	Convert an MDAnalysis AtomGroup to an MDTraj Trajectory.
	'''
	top = md.Topology()
	chains, residues, atom_map = {}, {}, {}
	for a, sym in zip(ag, atom_symbols(ag, cg=cg)):
		seg = a.segid if hasattr(a, "segid") else "A"
		if seg not in chains:
			chains[seg] = top.add_chain()
		rkey = (seg, a.resid, a.resname)
		if rkey not in residues:
			residues[rkey] = top.add_residue(a.resname, chains[seg], resSeq=int(a.resid))
		# register unknown symbols (e.g. bead sizes "T", "R") so shrake_rupley's change_radii can match them
		try:
			el = md.element.get_by_symbol(sym)
		except KeyError:
			el = md.element.Element(0, f"bead_{sym}", sym, a.mass, 0.0)
		atom_map[a.index] = top.add_atom(a.name, el, residues[rkey])
	if hasattr(ag, "bonds"):
		for b in ag.bonds:
			i, j = b.indices
			if i in atom_map and j in atom_map:
				top.add_bond(atom_map[i], atom_map[j])

	xyz, lengths, angles, times = [], [], [], []
	for ts in ag.universe.trajectory:
		xyz.append(ag.positions / 10.0)		  # Å -> nm
		times.append(ts.time)
		if ts.dimensions is not None:
			lengths.append(ts.dimensions[:3] / 10.0)
			angles.append(ts.dimensions[3:])
	return md.Trajectory(np.array(xyz), top, time=np.array(times),
						 unitcell_lengths=np.array(lengths) if lengths else None,
						 unitcell_angles=np.array(angles) if angles else None)

def calc_SASA(ag: mda.AtomGroup, mol_name: str, prefix: str, cg: bool = False, save: bool = False):
	'''
	Calculate the solvent accessible surface area (SASA) of every molecule in an MDAnalysis AtomGroup in
	every frame using MDTraj's shrake_rupley algorithm, and optionally save the time series and its
	distribution. As for the interaction distributions, the time series of all molecules are concatenated
	(all frames of the first molecule, then of the second, ...).

	Returns
	-------
	sasa : np.ndarray
		the SASA of every molecule in every frame (nm^2), length n_molecules * n_frames
	distr : np.ndarray
		its distribution (columns: bin centre, probability density)
	'''
	sasa = sasa_per_molecule(ag, cg=cg).ravel()

	if not cg: # cite AA radii paper
		citation_map = _citation_map()
		cite_string = citation_formatter(
			citation_map[ 'VDW_radii']
		)
		print(f"\nSASA calculation uses atomistic van der Waals radii from:\n{cite_string}\n")
		if 'P' in atom_symbols(ag, cg=False):
			cite_string = citation_formatter(
				citation_map[ 'VDW_radii_P']
			)
			print(f"\nPhosphorus atoms use atomistic van der Waals radii from:\n{cite_string}\n")

	if save:
		np.savetxt("{dir}/{prefix}{name}_{inter_type}.dat".format(dir=save,
	                                                	name=mol_name,
														inter_type='sasa',
														prefix=prefix),
				sasa)
	probs, edges = np.histogram(sasa, density=True, bins=INTERACTIONS['sasa'].get('bins'))
	center_points = edges[:-1] + np.diff(edges)/2.
	distr = np.transpose((center_points, probs))
	if save:
		np.savetxt("{dir}/{prefix}{name}_{inter_type}_distr.dat".format(dir=save,
																	name=mol_name,
																	inter_type='sasa',
																	prefix=prefix),
					distr)
	return sasa, distr
