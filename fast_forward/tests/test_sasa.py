"""
Tests for the SASA calculation (fast_forward.sasa) and its use in ff_map and ff_assess.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import MDAnalysis as mda

from fast_forward.interaction_distribution import INTERACTIONS, RECOGNISED_INTERACTIONS, distribution
from fast_forward.tests.datafiles import (GSH_AA_TRAJ, GSH_AA_TPR, GSH_MAP, GSH_ASSESS_TRAJ, GSH_ASSESS_TPR,
                                          GSH_ITP_OUTPUT, GSH_ASSESS_REFERENCE, HAVE_EXAMPLES_DATA,
                                           MISSING_EXAMPLES_DATA_REASON)

pytest.importorskip("sassy")
needs_data = pytest.mark.skipif(not HAVE_EXAMPLES_DATA, reason=MISSING_EXAMPLES_DATA_REASON)

from fast_forward.sasa import MoleculeSASA  # noqa: E402

# run the command line tools of this fast_forward (not of another installed copy)
ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [
    str(Path(__file__).resolve().parents[2]), os.environ.get('PYTHONPATH')]))}

SASA_LINE = re.compile(r'^\t(\S+)\s*:\s*([-\d.]+)\s*\(([-\d.]+)\)\s+([-\d.]+) / ([-\d.]+)')


def test_sasa_not_a_bonded_interaction():
    # ff_inter loops over the recognised interactions; the SASA only needs bins for scoring
    assert 'sasa' in INTERACTIONS
    assert 'sasa' not in RECOGNISED_INTERACTIONS


@pytest.mark.parametrize('inter_type, mean, std', [('bonds', 0.35, 0.02),
                                                   ('angles', 120, 10),
                                                   ('dihedrals', 0, 40),
                                                   ('distances', 5, 1),
                                                   ('sasa', 6.0, 0.2)])
def test_distribution_normalised(inter_type, mean, std):
    bins = INTERACTIONS[inter_type]['bins']
    series = np.random.default_rng(1).normal(mean, std, 1000)
    distr = distribution(series, inter_type)
    assert distr.shape == (len(bins) - 1, 2)
    assert np.allclose(distr[:, 0], bins[:-1] + np.diff(bins) / 2)
    assert np.isclose(np.sum(distr[:, 1] * np.diff(bins)), 1.0)


def test_distribution_warns_outside(capsys):
    distr = distribution(np.array([5.0, 6.0, 150.0]), 'sasa', 'GSH')
    assert "1 of 3 values of GSH sasa lie outside" in capsys.readouterr().out
    # the value outside is not counted
    bins = INTERACTIONS['sasa']['bins']
    assert np.isclose(np.sum(distr[:, 1] * np.diff(bins)), 1.0)


@needs_data
def test_molecule_sasa_aa(tmp_path):
    u = mda.Universe(str(GSH_AA_TPR), str(GSH_AA_TRAJ))
    ag = u.select_atoms("moltype LIG")
    mol_sasa = MoleculeSASA(ag, "GSH", prefix="test_", cg=False)
    mol_sasa.save(tmp_path)
    series, distr = mol_sasa.series, mol_sasa.distribution
    # one molecule: one SASA value per frame
    assert series.shape == (u.trajectory.n_frames,)
    # glutathione, as a single molecule, has a SASA of a few nm^2
    assert np.all((series > 3) & (series < 10))
    assert np.allclose(np.loadtxt(tmp_path / "test_GSH_sasa.dat"), series)
    assert np.allclose(np.loadtxt(tmp_path / "test_GSH_sasa_distr.dat"), distr)
    assert np.allclose(mol_sasa.sasa.results.body_sasa.ravel(), series)

    # the surface comes from the same SASA: by default a frame of the type's most probable SASA
    from sassy.statistics import mode
    surface = mol_sasa.write_surface(tmp_path)
    assert (tmp_path / "test_GSH_sasa_dots_AA.pdb").is_file()
    assert np.argmin(np.abs(series - mode(series))) == surface.frame
    # the B-factor of the atoms and their dots is the SASA of the atom
    assert "B-factor: SASA of the atom" in (tmp_path / "test_GSH_sasa_dots_AA.pdb").read_text()

    # another statistic
    surface = mol_sasa.write_surface(tmp_path, statistic="max")
    assert np.isclose(series[surface.frame], series.max())


def test_surface_statistic():
    from fast_forward.sasa import surface_statistic
    assert surface_statistic("mode") == "mode"
    assert surface_statistic("mean+std") == "mean+std"
    assert surface_statistic("5.5") == 5.5
    with pytest.raises(ValueError, match="unknown statistic 'average'"):
        surface_statistic("average")
    # 'reference' only for ff_assess
    assert surface_statistic("reference", allow_reference=True) == "reference"
    with pytest.raises(ValueError, match="unknown statistic 'reference'"):
        surface_statistic("reference")


@pytest.mark.parametrize('tool, stat', [('ff_map', 'average'), ('ff_map', 'reference'), ('ff_assess', 'average')])
def test_invalid_sasa_stat(tmp_path, tool, stat):
    cmd = [sys.executable, '-m', f'fast_forward.bin.{tool}', '-f', 'x.xtc', '-s', 'x.tpr',
           '-sasa', '-sasa-surface', '-sasa-stat', stat]
    cmd += ['-mols', 'LIG', '-o', 'out.gro'] if tool == 'ff_map' else ['-i', 'x.itp']
    proc = subprocess.run(cmd, cwd=tmp_path, timeout=60, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 2
    assert f"-sasa-stat: unknown statistic '{stat}'" in proc.stderr


@needs_data
def test_cg_surface_statistic(tmp_path):
    # an explicit statistic ignores the mapped structure
    u = mda.Universe(str(GSH_ASSESS_TPR), str(GSH_ASSESS_TRAJ))
    mol_sasa = MoleculeSASA(u.select_atoms("moltype GSH"), "GSH", cg=True)
    u.atoms[:6].write(str(tmp_path / "GSH_sasa_mapped.pdb"))
    surface = mol_sasa.write_cg_surface(reference_dir=tmp_path, outdir=tmp_path, statistic="min")
    assert surface.rmsd is None
    assert np.isclose(mol_sasa.series[surface.frame], mol_sasa.series.min())


@needs_data
def test_cg_surface_mismatched_reference(tmp_path, capsys):
    u = mda.Universe(str(GSH_ASSESS_TPR), str(GSH_ASSESS_TRAJ))
    mol_sasa = MoleculeSASA(u.select_atoms("moltype GSH"), "GSH", cg=True)
    # mapped structure with fewer beads than the CG molecule
    u.atoms[:3].write(str(tmp_path / "GSH_sasa_mapped.pdb"))
    surface = mol_sasa.write_cg_surface(reference_dir=tmp_path, outdir=tmp_path)
    assert "has 3 beads, but GSH has 6" in capsys.readouterr().out
    # fell back to the frame with median SASA, without a reference
    assert surface.rmsd is None
    assert (tmp_path / "GSH_sasa_dots_CG.pdb").is_file()


@needs_data
def test_ff_map_ff_assess_sasa(tmp_path):
    # reference: the bonded distributions (always assessed) + the SASA from ff_map
    ref = tmp_path / "ref"
    shutil.copytree(GSH_ASSESS_REFERENCE, ref)
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(GSH_MAP),
              '-mols', 'LIG:GSH', '-o', 'mapped.xtc',
              '-sasa', str(ref), '-sasa-surface']
    proc = subprocess.run(ff_map, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING - No CG molecule name" not in proc.stdout
    # the atomistic van der Waals radii are cited; GSH has no phosphorus
    assert "SASA calculation uses atomistic van der Waals radii from:\nRowland" in proc.stdout
    assert "Phosphorus atoms" not in proc.stdout
    for name in ("GSH_sasa.dat", "GSH_sasa_distr.dat", "GSH_sasa_dots_AA.pdb", "GSH_sasa_mapped.pdb"):
        assert (ref / name).is_file(), name
    # the SASA of every bead of the mapping, adding up to that of the molecule
    beads = ["1_CAC1", "1_AMC1", "1_AMD1", "1_SUL1", "1_AMD2", "1_CAC2"]
    for bead in beads:
        assert (ref / f"GSH_{bead}_sasa_distr.dat").is_file(), bead
    assert np.allclose(sum(np.loadtxt(ref / f"GSH_{bead}_sasa.dat") for bead in beads),
                       np.loadtxt(ref / "GSH_sasa.dat"))
    # the mapped structure has one bead per bead of the CG molecule
    assert mda.Universe(str(ref / "GSH_sasa_mapped.pdb")).atoms.n_atoms == 6

    ff_assess = [sys.executable, '-m', 'fast_forward.bin.ff_assess',
                 '-f', str(GSH_ASSESS_TRAJ), '-s', str(GSH_ASSESS_TPR), '-i', str(GSH_ITP_OUTPUT),
                 '-d', str(ref), '-interactions-only', '-sasa', '-sasa-surface']
    proc = subprocess.run(ff_assess, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = (tmp_path / "report_sasa.out").read_text()
    molecule_part, bead_part = report.split(" [ Bead SASA of GSH ]")
    matches = [SASA_LINE.match(line) for line in molecule_part.splitlines()]
    scores = {m.group(1): m.groups()[1:] for m in matches if m}
    assert set(scores) == {"GSH"}
    h_score, score, cg_mean, ref_mean = map(float, scores["GSH"])
    assert 0.0 <= h_score <= 1.0
    assert 0.0 <= score <= 1.0
    # the CG model reproduces the SASA of the atomistic molecule within ~10%
    assert abs(cg_mean - ref_mean) / ref_mean < 0.1

    # every bead is scored against its reference from ff_map
    assert "Overall Bead Score" in bead_part
    matches = [SASA_LINE.match(line) for line in bead_part.splitlines()]
    bead_scores = {m.group(1): tuple(map(float, m.groups()[1:])) for m in matches if m}
    assert set(bead_scores) == set(beads)
    for bead, (h_score, score, cg_mean, ref_mean) in bead_scores.items():
        assert 0.0 <= h_score <= 1.0 and 0.0 <= score <= 1.0
        # the reference mean is that of the bead SASA written by ff_map
        assert np.isclose(ref_mean, np.loadtxt(ref / f"GSH_{bead}_sasa.dat").mean(), atol=0.01)
    # CG surface, superposed onto the mapped structure
    assert "Superposed onto the reference structure" in (tmp_path / "GSH_sasa_dots_CG.pdb").read_text()

    # with -dists, the CG SASA of every bead is saved under the names of the reference;
    # with -plots, the bead distributions are plotted against the references
    dists = tmp_path / "dists"
    dists.mkdir()
    proc = subprocess.run(ff_assess[:-1] + ['-dists', str(dists), '-plots', str(tmp_path)], cwd=tmp_path,
                          timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "sasa_distributions.png").is_file()
    assert (tmp_path / "sasa_bead_distributions_GSH.png").is_file()
    cg_beads = sorted(p.name for p in dists.glob("GSH_1_*_sasa_distr.dat"))
    assert cg_beads == sorted(f"GSH_{bead}_sasa_distr.dat" for bead in beads)
    assert np.allclose(sum(np.loadtxt(dists / f"GSH_{bead}_sasa.dat") for bead in beads),
                       np.loadtxt(dists / "GSH_sasa.dat"))


@needs_data
def test_cite_aa_radii_phosphorus(capsys):
    from fast_forward.sasa import cite_aa_radii
    u = mda.Universe.empty(n_atoms=2, trajectory=True)
    u.add_TopologyAttr("names", values=["C1", "P1"])
    u.add_TopologyAttr("elements", values=["C", "P"])
    cite_aa_radii(u.atoms)
    out = capsys.readouterr().out
    assert "Rowland" in out
    assert "Phosphorus atoms use atomistic van der Waals radii from:\nBondi" in out
    cite_aa_radii(u.atoms[:1])
    assert "Phosphorus" not in capsys.readouterr().out


@needs_data
def test_ff_map_sasa_without_cg_name(tmp_path):
    # without a CG name, the SASA files are named after the AA molecule, with a warning
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(GSH_MAP),
              '-mols', 'LIG', '-o', 'mapped.xtc', '-sasa', str(tmp_path)]
    proc = subprocess.run(ff_map, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING - No CG molecule name given for LIG" in proc.stdout
    assert (tmp_path / "LIG_sasa_distr.dat").is_file()


def test_parse_mol_names():
    from fast_forward.bin.ff_map import parse_mol_names
    assert parse_mol_names(['LIG:GSH', 'POPC']) == (['LIG', 'POPC'], {'LIG': 'GSH'})
    assert parse_mol_names(['LIG']) == (['LIG'], {})
    for entry in ['LIG:', ':GSH', 'LIG:GSH:X', '']:
        with pytest.raises(ValueError, match="invalid -mols entry"):
            parse_mol_names([entry])


@pytest.mark.parametrize('mols', ['LIG:', ':GSH', 'LIG:GSH:X'])
def test_ff_map_invalid_mols(tmp_path, mols):
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map', '-f', 'x.xtc', '-s', 'x.tpr',
              '-mols', mols, '-o', 'out.gro']
    proc = subprocess.run(ff_map, cwd=tmp_path, timeout=60, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 2
    assert f"invalid -mols entry '{mols}'" in proc.stderr


@needs_data
def test_ff_assess_sasa_missing_reference(tmp_path):
    ref = tmp_path / "ref"
    shutil.copytree(GSH_ASSESS_REFERENCE, ref) # bonded references only
    ff_assess = [sys.executable, '-m', 'fast_forward.bin.ff_assess',
                 '-f', str(GSH_ASSESS_TRAJ), '-s', str(GSH_ASSESS_TPR), '-i', str(GSH_ITP_OUTPUT),
                 '-d', str(ref), '-interactions-only', '-sasa']
    proc = subprocess.run(ff_assess, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode != 0
    assert "GSH_sasa_distr.dat file not found" in proc.stderr


@pytest.mark.parametrize('beads, expected', [
    # dummy built from the atoms of its two neighbours, in any position
    ([[0, 1], [2, 3], [0, 1, 2, 3]], [False, False, True]),
    ([[0, 1, 2, 3], [0, 1], [2, 3]], [True, False, False]),
    # beads sharing an atom are not dummies
    ([[0, 1, 2], [2, 3, 4]], [False, False]),
    # of identical beads the first is kept
    ([[0, 1], [1, 0]], [False, True]),
    # empty beads are not dummies
    ([[0, 1], []], [False, False]),
])
def test_map_sasa_dummy_beads(beads, expected):
    from fast_forward.sasa import map_sasa
    atom_sasa = np.arange(1.0, 6.0)[:, None] # 5 atoms, 1 frame
    bead_sasa, dummy, _ = map_sasa(atom_sasa, beads)
    assert dummy.tolist() == expected
    # one row per non-dummy bead
    assert bead_sasa.shape == (expected.count(False), 1)


def test_map_sasa_shared():
    from fast_forward.sasa import map_sasa
    atom_sasa = np.array([[1.0, 2.0], [4.0, 8.0], [0.5, 0.5], [3.0, 3.0]]) # 4 atoms, 2 frames
    # atom 1 is shared (and listed twice in bead 0), atom 3 is in no bead
    bead_sasa, dummy, unassigned = map_sasa(atom_sasa, [[0, 1, 1], [1, 2]])
    assert not dummy.any()
    assert np.allclose(bead_sasa, [[1.0 + 2.0, 2.0 + 4.0], [2.0 + 0.5, 4.0 + 0.5]])
    assert unassigned.tolist() == [3]
    assert np.allclose(bead_sasa.sum(axis=0) + atom_sasa[3], atom_sasa.sum(axis=0))


def test_map_sasa_dummy_takes_no_share():
    from fast_forward.sasa import map_sasa
    atom_sasa = np.array([[1.0], [2.0], [4.0], [8.0]]) # 4 atoms, 1 frame
    # bead 2 is a dummy on all atoms; atom 1 is shared by beads 0 and 1 only
    bead_sasa, dummy, unassigned = map_sasa(atom_sasa, [[0, 1], [1, 2, 3], [0, 1, 2, 3]])
    assert dummy.tolist() == [False, False, True]
    assert np.allclose(bead_sasa, [[1.0 + 1.0], [1.0 + 4.0 + 8.0]])
    assert len(unassigned) == 0


def _run_ff_map_beads(map_file, outdir):
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(map_file),
              '-mols', 'LIG:GSH', '-o', 'mapped.xtc', '-sasa', str(outdir)]
    proc = subprocess.run(ff_map, cwd=outdir, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    beads = {p.name[len("GSH_"):-len("_sasa.dat")]: np.loadtxt(p) for p in outdir.glob("GSH_1_*_sasa.dat")}
    return proc, np.loadtxt(outdir / "GSH_sasa.dat"), beads


def _modified_map(path, edit):
    """GSH.map with every line passed through ``edit`` (returns the new line, or None to drop it)."""
    lines = [edit(line, line.split()) for line in Path(GSH_MAP).read_text().splitlines()]
    path.write_text("\n".join(line for line in lines if line is not None) + "\n")
    return path


@needs_data
def test_ff_map_bead_sasa(tmp_path):
    """
    The SASA of the atoms is split over the beads of the mapping: an atom shared by
    k beads gives 1/k to each, regardless of the mapping weights; virtual sites
    (a bead VS1 made of all atoms of AMD1 and SUL1) are left out; atoms in no bead
    (H16) count for the molecule only, with a warning.
    """
    for name in ("full", "shared", "vs"):
        (tmp_path / name).mkdir()
    _, sasa, full = _run_ff_map_beads(GSH_MAP, tmp_path / "full")
    assert sorted(full) == ["1_AMC1", "1_AMD1", "1_AMD2", "1_CAC1", "1_CAC2", "1_SUL1"]

    # C6 shared by AMD1 and SUL1 with weights 0.3/0.7, H16 in no bead
    def shared(line, tokens):
        if tokens[:2] == ['9', 'C6']:
            return '9\tC6\t0.3 AMD1 0.7 SUL1'
        return None if tokens[:2] == ['36', 'H16'] else line
    proc, shared_sasa, modified = _run_ff_map_beads(_modified_map(tmp_path / "shared.map", shared),
                                                    tmp_path / "shared")
    assert "1 atom(s) of GSH are in no bead" in proc.stdout and "H16" in proc.stdout
    assert np.allclose(shared_sasa, sasa) # the molecule does not depend on the mapping
    gained = modified["1_AMD1"] - full["1_AMD1"]
    assert np.allclose(gained, full["1_SUL1"] - modified["1_SUL1"]) # half of C6 moved, despite 0.3/0.7
    assert np.any(gained > 0)
    for bead in ("1_CAC1", "1_AMC1", "1_CAC2"):
        assert np.allclose(modified[bead], full[bead])
    lost = full["1_AMD2"] - modified["1_AMD2"] # H16
    assert np.all(lost >= -1e-12)
    assert np.allclose(sum(modified.values()) + lost, sasa)

    # virtual site VS1 on all atoms of AMD1 and SUL1: left out, the other beads are unchanged
    def virtual_site(line, tokens):
        if tokens[:1] == ['CAC1'] or (len(tokens) == 3 and tokens[2] in ('AMD1', 'SUL1')):
            return line.rstrip() + ' VS1'
        return line
    proc, vs_sasa, vs = _run_ff_map_beads(_modified_map(tmp_path / "vs.map", virtual_site), tmp_path / "vs")
    assert "No SASA assigned to the dummy bead(s) 1_VS1" in proc.stdout
    assert "in no bead" not in proc.stdout
    assert sorted(vs) == sorted(full) # no file for VS1
    assert np.allclose(vs_sasa, sasa)
    for bead in full:
        assert np.allclose(vs[bead], full[bead])


@needs_data
def test_cg_bead_sasa_skip_dummies():
    """
    ff_assess saves the SASA of every CG bead, virtual sites included, except dummy
    beads (type U, radius 0).
    """
    import vermouth.forcefield
    from vermouth.gmx.itp_read import read_itp
    ff = vermouth.forcefield.ForceField("test")
    lines = Path(GSH_ITP_OUTPUT).read_text().splitlines(True)
    read_itp(lines + ["[ virtual_sitesn ]\n", "4 1 1 2 3\n"], ff) # bead 4 as a virtual site
    block = ff.blocks["GSH"]
    names = [f"{block.nodes[n]['resid']}_{block.nodes[n]['atomname']}" for n in block.nodes]

    u = mda.Universe(str(GSH_ASSESS_TPR), str(GSH_ASSESS_TRAJ))
    atoms = u.select_atoms("moltype GSH")
    types = atoms.types.copy()
    types[5] = "U" # last bead as a dummy
    atoms.types = types
    mol_sasa = MoleculeSASA(atoms, "GSH", cg=True)
    mol_sasa.cg_bead_sasa(block)
    assert mol_sasa.bead_names == names[:5] # the virtual site (bead 4) is kept
    assert mol_sasa.bead_series.shape == (5, u.trajectory.n_frames)
    assert np.allclose(mol_sasa.bead_series.sum(axis=0), mol_sasa.series) # the dummy has no SASA
    assert mol_sasa.sasa.results.atom_sasa is None # the SASA of the atoms is freed
    with pytest.raises(ValueError, match="has 6 beads, but the itp has 5"):
        block.remove_node(list(block.nodes)[-1])
        MoleculeSASA(atoms, "GSH", cg=True).cg_bead_sasa(block)


@needs_data
def test_ff_assess_sasa_without_bead_references(tmp_path):
    """
    A reference without bead files (e.g. from an older ff_map) still assesses the
    molecule; the beads are skipped with a warning.
    """
    ref = tmp_path / "ref"
    shutil.copytree(GSH_ASSESS_REFERENCE, ref)
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(GSH_MAP),
              '-mols', 'LIG:GSH', '-o', 'mapped.xtc', '-sasa', str(ref)]
    proc = subprocess.run(ff_map, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for bead_file in ref.glob("GSH_1_*_sasa*.dat"):
        bead_file.unlink()

    ff_assess = [sys.executable, '-m', 'fast_forward.bin.ff_assess',
                 '-f', str(GSH_ASSESS_TRAJ), '-s', str(GSH_ASSESS_TPR), '-i', str(GSH_ITP_OUTPUT),
                 '-d', str(ref), '-interactions-only', '-sasa', '-plots', str(tmp_path)]
    proc = subprocess.run(ff_assess, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No reference SASA for the bead(s) 1_CAC1, 1_AMC1" in proc.stdout
    report = (tmp_path / "report_sasa.out").read_text()
    assert "GSH" in report and "Bead SASA" not in report
    assert (tmp_path / "sasa_distributions.png").is_file()
    assert not (tmp_path / "sasa_bead_distributions_GSH.png").exists()
