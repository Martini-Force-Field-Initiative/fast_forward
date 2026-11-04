"""
Solvent accessible surface area (SASA) in the style of ``gmx sasa``.

Dots are placed on the probe-expanded sphere of every particle, using the
geodesic dot sets of ``gmx sasa``, and a dot counts as exposed if it is not
inside the expanded sphere of any other particle of the same molecule
(Shrake-Rupley). The burial test is vectorised: for particle i with expanded
radius R_i and a neighbour j at d_ij = x_j - x_i, the dot at x_i + R_i u lies
inside sphere j exactly if

	u . d_ij > (R_i^2 + |d_ij|^2 - R_j^2) / (2 R_i),

which is tested in a compiled (numba) loop over all dots, starting with the
neighbour that buried the previous dot. Neighbour pairs of many frames and
molecules are found with one k-d tree, after placing every (frame, molecule)
block in its own region of space.
"""
from functools import lru_cache
from collections import ChainMap
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange
import MDAnalysis as mda
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from . import DATA_PATH
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

PROBE_RADIUS = 0.191 # nm, tiny bead radius
N_DOTS = 4800 # minimum number of dots per particle for the SASA time series (-> 4842 dots)

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
	unknown = [i for i, sym in enumerate(symbols) if sym not in table]
	if unknown:
		kind = 'bead size' if cg else 'element'
		atoms = ', '.join(f"{ag[i].resname}{ag[i].resid}:{ag[i].name} ({kind} {symbols[i]!r})"
						  for i in unknown[:10])
		more = f" and {len(unknown) - 10} more" if len(unknown) > 10 else ""
		raise ValueError(f"No {'CG' if cg else 'atomistic'} SASA radius known for {len(unknown)} "
						 f"{'bead' if cg else 'atom'}(s): {atoms}{more}. "
						 f"Known {kind}s: {', '.join(table)}.")
	return np.array([table[sym] for sym in symbols])

def _molecule_index(ag: mda.AtomGroup):
	'''
	Index (0, 1, ...) of the molecule of every atom in ``ag`` (by molnum, or by
	residue if there are no molnums), molecules numbered in the order in which
	they appear.
	'''
	keys = ag.molnums if hasattr(ag.atoms, 'molnums') else ag.resindices
	_, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
	rank = np.empty(len(first), dtype=int)
	rank[np.argsort(first)] = np.arange(len(first))
	return rank[inverse.ravel()]

def split_molecules(ag: mda.AtomGroup):
	'''
	Split ``ag`` into its molecules (by molnum, or by residue if there are no
	molnums), in the order in which they appear.
	'''
	molecule = _molecule_index(ag)
	return [ag[molecule == i] for i in range(molecule.max() + 1 if len(molecule) else 0)]

# ---------------------------------------------------------------------------
# unit-sphere dots as in gmx sasa (nsc.cpp)
# ---------------------------------------------------------------------------

def _divarc(p1, p2, div1, div2):
	"""
	Point at fraction div1/div2 along the great-circle arc p1 -> p2 (gmx divarc).
	Works on arrays of points (..., 3) and fractions (...).
	"""
	p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
	axis = np.cross(p1, p2)
	norm = np.linalg.norm(axis, axis=-1, keepdims=True)
	phi = np.arctan2(norm, np.sum(p1 * p2, axis=-1, keepdims=True)) * (np.asarray(div1, float) / div2)[..., None]
	axis = axis / norm
	p1 = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
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
	tl, tl2 = np.array([(tl, tl2) for tl in range(1, tess - 1) for tl2 in range(1, tess - tl)]).reshape(-1, 2).T
	if not len(tl):
		return []
	pji = _divarc(vj, vi, tl, tess)
	pki = _divarc(vk, vi, tl, tess)
	pij = _divarc(vi, vj, tl2, tess)
	pkj = _divarc(vk, vj, tl2, tess)
	pik = _divarc(vi, vk, tess - tl - tl2, tess)
	pjk = _divarc(vj, vk, tess - tl - tl2, tess)
	p = (_divarc(pki, pji, tl2, tess - tl)
		 + _divarc(pkj, pij, tl, tess - tl2)
		 + _divarc(pjk, pik, tl, tl + tl2))
	return list(p / np.linalg.norm(p, axis=-1, keepdims=True))

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
					pts += list(_divarc(v[i], v[j], np.arange(1, tess), tess))
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
					pts += list(_divarc(w[i], w[j], np.arange(1, tess), tess))
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

# ---------------------------------------------------------------------------
# burial test
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _test_dots(starts, vec, threshold, unit, exposed):
	"""
	Mark the dots of every sphere that are not inside any neighbouring sphere.

	The neighbours of sphere a are ``starts[a]:starts[a + 1]``, closest first;
	dot u of sphere a is inside neighbour m if ``u . vec[m] > threshold[m]``.
	As in gmx sasa, the neighbour that buried the previous dot is tested first.
	"""
	for a in prange(len(starts) - 1):
		first, end = starts[a], starts[a + 1]
		last = first
		for k in range(unit.shape[0]):
			if first == end:
				exposed[a, k] = True
				continue
			ux, uy, uz = unit[k, 0], unit[k, 1], unit[k, 2]
			if ux * vec[last, 0] + uy * vec[last, 1] + uz * vec[last, 2] > threshold[last]:
				exposed[a, k] = False
				continue
			free = True
			for m in range(first, end):
				if ux * vec[m, 0] + uy * vec[m, 1] + uz * vec[m, 2] > threshold[m]:
					last = m
					free = False
					break
			exposed[a, k] = free

@njit(parallel=True, cache=True)
def _neighbour_lists(xyz, ext, block, pairs):
	"""
	Neighbour lists of the spheres from candidate ``pairs`` (i < j): only pairs
	of overlapping spheres in the same block are kept, every pair in both
	directions, the neighbours of every sphere sorted closest first (they bury
	the most dots).

	Returns
	-------
	starts : np.ndarray
		(n + 1,) the neighbours of sphere a are ``starts[a]:starts[a + 1]``
	vec : np.ndarray
		(n_neighbours, 3) vector from the sphere to its neighbour
	threshold : np.ndarray
		(n_neighbours,) dot u of the sphere is inside the neighbour if ``u . vec > threshold``
	"""
	n = len(xyz)
	keep = np.zeros(len(pairs), dtype=np.bool_)
	pair_dist2 = np.empty(len(pairs))
	starts = np.zeros(n + 1, dtype=np.int64)
	for p in range(len(pairs)):
		i, j = pairs[p, 0], pairs[p, 1]
		if block[i] != block[j]:
			continue
		d2 = 0.0
		for c in range(3):
			d2 += (xyz[j, c] - xyz[i, c])**2
		if d2 < (ext[i] + ext[j])**2:
			keep[p] = True
			pair_dist2[p] = d2
			starts[i + 1] += 1
			starts[j + 1] += 1
	for a in range(n):
		starts[a + 1] += starts[a]
	fill = starts[:-1].copy()
	vec = np.empty((starts[n], 3))
	dist2 = np.empty(starts[n])
	threshold = np.empty(starts[n])
	for p in range(len(pairs)):
		if not keep[p]:
			continue
		i, j = pairs[p, 0], pairs[p, 1]
		d2 = pair_dist2[p]
		for own, other, sign in ((i, j, 1.0), (j, i, -1.0)):
			m = fill[own]
			fill[own] += 1
			for c in range(3):
				vec[m, c] = sign * (xyz[j, c] - xyz[i, c])
			dist2[m] = d2
			threshold[m] = (ext[own]**2 + d2 - ext[other]**2) / (2 * ext[own])
	# insertion sort of every (short) neighbour list by distance
	for a in prange(n):
		for m in range(starts[a] + 1, starts[a + 1]):
			k = m
			while k > starts[a] and dist2[k - 1] > dist2[k]:
				dist2[k - 1], dist2[k] = dist2[k], dist2[k - 1]
				threshold[k - 1], threshold[k] = threshold[k], threshold[k - 1]
				for c in range(3):
					vec[k - 1, c], vec[k, c] = vec[k, c], vec[k - 1, c]
				k -= 1
	return starts, vec, threshold

def _exposed_dots(xyz, ext, block, unit):
	"""
	Find the exposed dots of a set of spheres.

	Parameters
	----------
	xyz : np.ndarray
		(n, 3) sphere centres
	ext : np.ndarray
		(n,) radii of the spheres (van der Waals + probe radius), all > 0
	block : np.ndarray
		(n,) block of every sphere: spheres only bury dots of spheres in the same block
		(the same molecule in the same frame)
	unit : np.ndarray
		(n_dots, 3) unit-sphere dots

	Returns
	-------
	np.ndarray
		(n, n_dots) bool, True for the dots that are not inside another sphere
	"""
	xyz = np.ascontiguousarray(xyz, dtype=np.float64)
	ext = np.ascontiguousarray(ext, dtype=np.float64)
	block = np.ascontiguousarray(block, dtype=np.int64)
	if len(xyz) > 1:
		pairs = cKDTree(xyz).query_pairs(2 * ext.max(), output_type='ndarray').astype(np.int64)
	else:
		pairs = np.zeros((0, 2), dtype=np.int64)
	starts, vec, threshold = _neighbour_lists(xyz, ext, block, pairs)
	exposed = np.empty((len(xyz), len(unit)), dtype=bool)
	_test_dots(starts, vec, threshold, np.ascontiguousarray(unit, dtype=np.float64), exposed)
	return exposed

def surface_dots(xyz, radii, probe=PROBE_RADIUS, ndots=240):
	"""
	Generate dots of the solvent accessible surface of a set of spheres.

	Parameters
	----------
	xyz : np.ndarray
		(n_atoms, 3) coordinates
	radii : np.ndarray
		(n_atoms,) van der Waals radii, in the units of ``xyz``. Particles with
		radius 0 (dummy beads, virtual sites) neither carry dots nor bury dots of others.
	probe : float
		probe radius, in the units of ``xyz`` (default in nm)
	ndots : int
		minimum number of dots per sphere

	Returns
	-------
	dots : np.ndarray
		(n_dots, 3) dot positions
	owners : np.ndarray
		(n_dots,) index of the particle each dot belongs to
	area : float
		solvent accessible surface area estimated from the dots (units of ``xyz`` squared)
	"""
	xyz = np.asarray(xyz, dtype=float)
	radii = np.asarray(radii, dtype=float)
	unit = gmx_unit_dots(int(ndots))
	active = np.flatnonzero(radii > 0)
	ext = radii[active] + probe
	centres = xyz[active]
	exposed = _exposed_dots(centres, ext, np.zeros(len(active), dtype=int), unit)
	area = float(np.sum(exposed.mean(axis=1) * 4 * np.pi * ext**2))
	dots = (centres[:, None, :] + ext[:, None, None] * unit[None])[exposed]
	owners = active[np.nonzero(exposed)[0]]
	return dots, owners, area

class SASA:
	"""
	Solvent accessible surface area of every molecule in an MDAnalysis
	AtomGroup, calculated as in ``gmx sasa``.

	Every molecule (by molnum, or by residue if there are no molnums) is taken
	on its own, so other molecules do not bury its surface. Particles with
	radius 0 (dummy beads, virtual sites) are ignored. Molecules are assumed to
	be whole.

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
	cg : bool
		use the CG bead radii instead of the atomistic van der Waals radii
	probe : float
		probe radius (nm)
	ndots : int
		minimum number of dots per particle, see :func:`gmx_unit_dots`
	radii : np.ndarray, optional
		van der Waals radii (nm) of the particles, instead of the default radii
	max_frame_atoms : int
		frames are processed in blocks of about this many particles

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> series = sasa.run()  # (n_molecules, n_frames), nm^2
	"""
	def __init__(self, ag, cg=False, probe=PROBE_RADIUS, ndots=N_DOTS, radii=None, max_frame_atoms=20000):
		self.ag = ag
		radii = atom_radii(ag, cg=cg) if radii is None else np.asarray(radii, dtype=float)
		molecule = _molecule_index(ag)
		self.n_molecules = int(molecule.max()) + 1 if len(molecule) else 0
		self.unit = gmx_unit_dots(int(ndots))
		active = np.flatnonzero(radii > 0)
		self._indices = ag.indices[active] # universe indices of the particles with a radius
		self._molecule = molecule[active]
		self._ext = (radii[active] + probe) * 10.0 # Å
		self._dot_area = 4 * np.pi * (self._ext / 10.0)**2 / len(self.unit) # nm^2 per dot
		# every molecule is placed relative to its first particle
		_, first = np.unique(self._molecule, return_index=True)
		anchor = np.zeros(self.n_molecules, dtype=int)
		anchor[self._molecule[first]] = first
		self._anchor = anchor[self._molecule]
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(active)))

	def _molecule_areas(self, xyz):
		"""
		SASA (nm^2) of every molecule in a block of frames.

		Parameters
		----------
		xyz : np.ndarray
			(n_frames, n_particles, 3) positions (Å) of the particles with a radius

		Returns
		-------
		np.ndarray
			(n_frames, n_molecules)
		"""
		n_frames, n_atoms, _ = xyz.shape
		# put every (frame, molecule) block in its own slab along x, so a single
		# k-d tree finds the neighbours of all of them and no pairs between them
		xyz = xyz - xyz[:, self._anchor]
		span = 2 * np.abs(xyz).max() + 2 * self._ext.max() + 1.0
		block = (np.arange(n_frames)[:, None] * self.n_molecules + self._molecule[None]).ravel()
		xyz = xyz.reshape(-1, 3)
		xyz[:, 0] += block * span
		ext = np.tile(self._ext, n_frames)
		exposed = np.count_nonzero(_exposed_dots(xyz, ext, block, self.unit), axis=1)
		areas = exposed * np.tile(self._dot_area, n_frames)
		return np.bincount(block, weights=areas, minlength=n_frames * self.n_molecules).reshape(n_frames, -1)

	def run(self, start=None, stop=None, step=None):
		"""
		SASA (nm^2) of every molecule in the frames ``start:stop:step`` of the trajectory.

		Returns
		-------
		np.ndarray
			(n_molecules, n_frames), molecules in the order of :func:`split_molecules`
		"""
		trajectory = self.ag.universe.trajectory
		if not len(self._indices): # only particles without radius
			return np.zeros((self.n_molecules, len(trajectory[start:stop:step])))
		current = trajectory.ts.frame
		results, buffer = [], []
		for ts in trajectory[start:stop:step]:
			buffer.append(ts.positions[self._indices])
			if len(buffer) == self._frames_per_block:
				results.append(self._molecule_areas(np.array(buffer, dtype=np.float64)))
				buffer = []
		if buffer:
			results.append(self._molecule_areas(np.array(buffer, dtype=np.float64)))
		trajectory[current] # leave the trajectory on the frame it was on
		if not results:
			return np.zeros((self.n_molecules, 0))
		return np.concatenate(results).T

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

def calc_SASA(ag: mda.AtomGroup, mol_name: str, prefix: str, cg: bool = False, save: bool = False,
			  ndots: int = N_DOTS):
	'''
	Calculate the solvent accessible surface area (SASA) of every molecule in an MDAnalysis AtomGroup in
	every frame (see :class:`SASA`), and optionally save the time series and its distribution. As for the
	interaction distributions, the time series of all molecules are concatenated (all frames of the first
	molecule, then of the second, ...).

	Returns
	-------
	sasa : np.ndarray
		the SASA of every molecule in every frame (nm^2), length n_molecules * n_frames
	distr : np.ndarray
		its distribution (columns: bin centre, probability density)
	'''
	sasa = SASA(ag, cg=cg, ndots=ndots).run().ravel()

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
