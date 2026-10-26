import numpy as np
try:
	import mdtraj as md
except ImportError:
	md = None
from . import DATA_PATH
import numpy as np
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from collections import ChainMap
from fast_forward.interaction_distribution import INTERACTIONS

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

def mda_to_mdtraj(ag, cg=False):
	'''
	Convert an MDAnalysis AtomGroup to an MDTraj Trajectory.
	'''
	if not hasattr(ag.atoms, "elements") and cg:
		# assign elements as bead sizes based on atom types (R = regular, S = small, T = tiny, U = dummy)
		ag.universe.add_TopologyAttr("elements", ["R"] * ag.universe.atoms.n_atoms)
		ag.elements = [at[0] if at[0] in ("S", "T", "U") else "R" for at in ag.types]
	top = md.Topology()
	chains, residues, atom_map = {}, {}, {}
	for a in ag:
		seg = a.segid if hasattr(a, "segid") else "A"
		if seg not in chains:
			chains[seg] = top.add_chain()
		rkey = (seg, a.resid, a.resname)
		if rkey not in residues:
			residues[rkey] = top.add_residue(a.resname, chains[seg], resSeq=int(a.resid))
		sym = a.element if hasattr(a, "element") and a.element else a.name[0]
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

def calc_SASA(ag, mol_name, prefix, cg=False, save=False):
	'''
	Calculate the solvent accessible surface area (SASA) of an MDAnalysis AtomGroup using MDTraj's shrake_rupley algorithm.
	'''
	check_mdtraj()
	traj = mda_to_mdtraj(ag, cg=cg)
	radii = CG_RADII if cg else AA_RADII
	sasa = md.shrake_rupley(traj, probe_radius=0.191, n_sphere_points=4800, change_radii=radii, mode='residue')
	sasa = sasa.flatten()

	if not cg: # cite AA radii paper
		with open(DATA_PATH/'citations.bib') as citation_file:
			ff_citations = read_bib(citation_file)
		citation_map = ChainMap(ff_citations, COMMON_CITATIONS)
		cite_string = citation_formatter(
			citation_map[ 'VDW_radii']
		)
		print(f"\nSASA calculation uses atomistic van der Waals radii from:\n{cite_string}\n")
		if any(a.element.symbol == 'P' for a in traj.topology.atoms):
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
	return distr
