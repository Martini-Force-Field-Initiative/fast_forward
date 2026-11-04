"""
Tests for the SASA calculation and the dot representation of the solvent
accessible surface.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fast_forward.sasa import SASA, atom_radii, gmx_unit_dots, surface_dots
from fast_forward.sasa_surface import best_matching_frame, closest_to_median


@pytest.mark.parametrize('ndots, expected', [(24, 32), (50, 92), (100, 122), (240, 252)])
def test_gmx_unit_dots(ndots, expected):
    """
    The number of dots follows gmx sasa -ndots and all dots lie on the unit sphere.
    """
    dots = gmx_unit_dots(ndots)
    assert len(dots) == expected
    assert np.allclose(np.linalg.norm(dots, axis=1), 1.0)
    # no duplicated dots
    assert len(np.unique(np.round(dots, 6), axis=0)) == expected


def test_single_sphere():
    """
    An isolated sphere keeps all its dots on the probe-expanded sphere.
    """
    dots, owners, area = surface_dots(np.zeros((1, 3)), np.array([0.2]), probe=0.1, ndots=240)
    assert len(dots) == 252
    assert np.all(owners == 0)
    assert np.allclose(np.linalg.norm(dots, axis=1), 0.3)
    assert area == pytest.approx(4 * np.pi * 0.3**2)


def test_two_spheres_area():
    """
    Two overlapping spheres: the dot area approaches the analytic value and
    no dot lies inside the other sphere.
    """
    radius, probe, dist = 0.2, 0.1, 0.4
    xyz = np.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]])
    dots, owners, area = surface_dots(xyz, np.array([radius, radius]), probe=probe, ndots=4000)
    ext = radius + probe
    cap = 2 * np.pi * ext * (ext - dist / 2)
    assert area == pytest.approx(2 * (4 * np.pi * ext**2 - cap), rel=0.01)
    for i in range(2):
        other = xyz[1 - i]
        assert np.all(np.linalg.norm(dots[owners == i] - other, axis=1) >= ext - 1e-9)


def test_zero_radius_ignored():
    """
    Particles without radius (dummies, virtual sites) neither carry dots nor bury others.
    """
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    dots, owners, area = surface_dots(xyz, np.array([0.2, 0.0]), probe=0.1, ndots=240)
    assert len(dots) == 252
    assert np.all(owners == 0)
    assert area == pytest.approx(4 * np.pi * 0.3**2)


def test_best_matching_frame():
    """
    The frame that is a rigid-body copy of the reference is found and superposed onto it.
    """
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(8, 3)) * 5
    frames = np.array([reference @ Rotation.random(random_state=i).as_matrix().T + rng.normal(size=3) * 10
                       for i in range(5)])
    frames[[0, 1, 3, 4]] += rng.normal(size=(4, 8, 3))
    frame, aligned, rmsd = best_matching_frame(frames, reference)
    assert frame == 2
    assert rmsd == pytest.approx(0, abs=1e-4)
    assert np.allclose(aligned, reference, atol=1e-4)


def test_closest_to_median():
    assert closest_to_median([5.0, 1.0, 3.2, 9.0, 2.0]) == 2


@pytest.fixture
def gsh_aa():
    """
    Atomistic GSH (a single molecule) for the first 20 frames.
    """
    from fast_forward.tests.datafiles import GSH_AA_TPR, GSH_AA_TRAJ, HAVE_EXAMPLES_DATA
    if not HAVE_EXAMPLES_DATA:
        pytest.skip('GSH example data not available')
    import MDAnalysis as mda
    universe = mda.Universe(str(GSH_AA_TPR), str(GSH_AA_TRAJ), in_memory=True, in_memory_step=25)
    return universe.select_atoms('moltype GSH')


def test_calc_sasa_returns(gsh_aa):
    """
    calc_SASA returns the SASA in every frame and its distribution.
    """
    from fast_forward.sasa import calc_SASA
    series, distr = calc_SASA(gsh_aa, 'GSH', '')
    assert series.shape == (gsh_aa.universe.trajectory.n_frames,)
    assert np.allclose(series, SASA(gsh_aa).run()[0])
    assert distr.shape[1] == 2
    assert np.isclose(np.sum(distr[:, 1]) * np.diff(distr[:2, 0])[0], 1.0)


def test_surface_reuses_sasa(gsh_aa, tmp_path):
    """
    The SASA series passed in selects the frame.
    """
    import fast_forward.sasa_surface as sasa_surface
    from vermouth.file_writer import DeferredFileWriter
    series = np.arange(gsh_aa.universe.trajectory.n_frames, dtype=float)  # median at the middle frame
    frame, atoms = sasa_surface.write_aa_sasa_surface(gsh_aa, 'GSH', series, outdir=tmp_path, ndots=24)
    DeferredFileWriter().write()
    assert frame == closest_to_median(series)
    assert atoms == set(gsh_aa.indices.tolist())
    assert (tmp_path / 'GSH_sasa_dots_AA.pdb').exists()


def test_sasa_per_copy(gsh_aa):
    """
    With several copies, every copy gets its own SASA (copies do not bury each
    other, even when they overlap), and calc_SASA concatenates the copies'
    time series like the interaction distributions; the surface uses the first.
    """
    import MDAnalysis as mda
    from fast_forward.sasa import calc_SASA, split_molecules
    n_frames = gsh_aa.universe.trajectory.n_frames
    # two copies of GSH on top of each other, the second shifted by 1 Å
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    two = mda.Merge(gsh_aa, gsh_aa)
    two.add_TopologyAttr('molnums', [0, 1])  # one residue per copy
    two.load_new(np.concatenate([positions, positions + 1.0], axis=1), order='fac')
    copies = split_molecules(two.atoms)
    assert [len(copy) for copy in copies] == [gsh_aa.n_atoms] * 2
    assert np.all(copies[0].indices == np.arange(gsh_aa.n_atoms))

    single = SASA(gsh_aa).run()[0]
    per_copy = SASA(two.atoms).run()
    assert per_copy.shape == (2, n_frames)
    # translation does not change the SASA (up to float32 rounding); burial by the
    # overlapping copy would change it by several nm^2
    assert np.allclose(per_copy, single[None], atol=2e-3)

    series, _ = calc_SASA(two.atoms, 'GSH', '')
    assert np.allclose(series, np.concatenate([single, single]), atol=2e-3)



def _brute_force_areas(xyz, radii, probe, unit):
    """
    Per-sphere exposed area, testing every dot against every other sphere.
    """
    ext = radii + probe
    areas = []
    for i in range(len(xyz)):
        dots = xyz[i] + ext[i] * unit
        dist = np.linalg.norm(dots[:, None] - xyz[None], axis=2)
        inside = dist < ext[None]
        inside[:, i] = False
        areas.append(np.mean(~inside.any(axis=1)) * 4 * np.pi * ext[i]**2)
    return np.array(areas)


def test_surface_dots_brute_force():
    """
    The neighbour lists and the compiled burial test agree with testing every
    dot against every sphere, for a dense random cluster with mixed radii.
    """
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 1.2, size=(60, 3))
    radii = rng.choice([0.109, 0.156, 0.175, 0.264], size=60)
    unit = gmx_unit_dots(240)
    _, owners, area = surface_dots(xyz, radii, probe=0.191, ndots=240)
    expected = _brute_force_areas(xyz, radii, 0.191, unit)
    assert area == pytest.approx(expected.sum(), rel=1e-12)
    counts = np.bincount(owners, minlength=60)
    assert np.allclose(counts / len(unit) * 4 * np.pi * (radii + 0.191)**2, expected)


def test_sasa_class(gsh_aa):
    """
    The SASA of every frame equals the area of the surface dots of that frame,
    independent of how many frames are processed at once.
    """
    sasa = SASA(gsh_aa, ndots=240).run()
    assert sasa.shape == (1, gsh_aa.universe.trajectory.n_frames)
    radii = atom_radii(gsh_aa)
    for frame in (0, 7, 19):
        gsh_aa.universe.trajectory[frame]
        _, _, area = surface_dots(gsh_aa.positions / 10, radii, ndots=240)
        assert sasa[0, frame] == pytest.approx(area, rel=1e-6)
    # one frame per block
    one_by_one = SASA(gsh_aa, ndots=240, max_frame_atoms=1).run()
    assert np.allclose(one_by_one, sasa, rtol=1e-6)
    assert np.allclose(SASA(gsh_aa, ndots=240).run(start=5, step=3), sasa[:, 5::3], rtol=1e-6)


def test_sasa_zero_radius(gsh_aa):
    """
    Particles with radius 0 are ignored: giving an atom radius 0 equals removing it.
    """
    radii = atom_radii(gsh_aa)
    radii[3] = 0
    without = SASA(gsh_aa, radii=radii, ndots=240).run()
    removed = gsh_aa[np.arange(gsh_aa.n_atoms) != 3]
    assert np.allclose(without, SASA(removed, ndots=240).run(), rtol=1e-6)


def test_unknown_element(gsh_aa):
    """
    An atom whose element has no atomistic radius raises an error naming the atom.
    """
    atoms = gsh_aa.universe.copy().select_atoms('moltype GSH')
    atoms[0].element = 'Na'
    with pytest.raises(ValueError, match=rf"{atoms[0].resname}{atoms[0].resid}:{atoms[0].name} \(element 'Na'\)"):
        atom_radii(atoms)
    with pytest.raises(ValueError, match='No atomistic SASA radius known for 1 atom'):
        SASA(atoms)
