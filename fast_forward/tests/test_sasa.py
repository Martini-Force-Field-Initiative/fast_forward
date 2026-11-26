"""
Tests of the SASA calculation (fast_forward.sasa) and its use in ff_map and ff_assess that need
sassy; they are skipped without it. The tests without sassy are in test_sasa_core.py.
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

from fast_forward.tests.datafiles import (ENV, GSH_AA_TRAJ, GSH_AA_TPR, GSH_MAP, GSH_ASSESS_TRAJ, GSH_ASSESS_TPR,
                                          GSH_ITP_OUTPUT, GSH_ASSESS_REFERENCE, HAVE_EXAMPLES_DATA,
                                           MISSING_EXAMPLES_DATA_REASON)

pytest.importorskip("sassy")
needs_data = pytest.mark.skipif(not HAVE_EXAMPLES_DATA, reason=MISSING_EXAMPLES_DATA_REASON)

from fast_forward.sasa import MoleculeSASA  # noqa: E402


def _bead_sasa(directory, mol="GSH", suffix="sasa.dat"):
    """{bead name: SASA time series (or distribution, suffix 'sasa_distr.dat')} of the real beads, via the bead table"""
    from fast_forward.sasa import read_bead_table
    rows = read_bead_table(Path(directory) / f"{mol}_sasa_beads.dat")
    return {row.label: np.loadtxt(Path(directory) / row.file.replace("sasa_distr.dat", suffix))
            for row in rows if not row.dummy}


SASA_LINE = re.compile(r'^\t(\S+)\s*:\s*([-\d.]+)\s*\(([-\d.]+)\)\s+([-\d.]+) / ([-\d.]+)')


@needs_data
def test_molecule_sasa_aa(tmp_path):
    u = mda.Universe(str(GSH_AA_TPR), str(GSH_AA_TRAJ))
    ag = u.select_atoms("moltype LIG")
    mol_sasa = MoleculeSASA(ag, "GSH", prefix="test_", cg=False)
    assert mol_sasa.sasa.keep_atoms is False # AA: the bead SASA is summed during the run
    with pytest.raises(ValueError, match="needs an atomistic SASA"):
        MoleculeSASA(ag, "GSH", cg=True).set_mapped_beads([], None)
    with pytest.raises(RuntimeError, match="call run"):
        mol_sasa.save(tmp_path)
    mol_sasa.run()
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


def _split(atom_sasa, beads):
    """The SASA of the beads that are not dummies, as MoleculeSASA splits it, and the dummies."""
    from sassy import group_weights
    from fast_forward.sasa import find_dummies
    dummy = find_dummies(beads)
    weights, unassigned = group_weights([b for b, d in zip(beads, dummy) if not d], len(atom_sasa))
    return weights @ atom_sasa, dummy, unassigned


def test_bead_split_shared():
    atom_sasa = np.array([[1.0, 2.0], [4.0, 8.0], [0.5, 0.5], [3.0, 3.0]]) # 4 atoms, 2 frames
    # atom 1 is shared (and listed twice in bead 0), atom 3 is in no bead
    bead_sasa, dummy, unassigned = _split(atom_sasa, [[0, 1, 1], [1, 2]])
    assert not dummy.any()
    assert np.allclose(bead_sasa, [[1.0 + 2.0, 2.0 + 4.0], [2.0 + 0.5, 4.0 + 0.5]])
    assert unassigned.tolist() == [3]
    assert np.allclose(bead_sasa.sum(axis=0) + atom_sasa[3], atom_sasa.sum(axis=0))


def test_bead_split_dummy_takes_no_share():
    atom_sasa = np.array([[1.0], [2.0], [4.0], [8.0]]) # 4 atoms, 1 frame
    # bead 2 is a dummy on all atoms; atom 1 is shared by beads 0 and 1 only
    bead_sasa, dummy, unassigned = _split(atom_sasa, [[0, 1], [1, 2, 3], [0, 1, 2, 3]])
    assert dummy.tolist() == [False, False, True]
    assert np.allclose(bead_sasa, [[1.0 + 1.0], [1.0 + 4.0 + 8.0]])
    assert len(unassigned) == 0


def test_statistics_match_sassy():
    sassy = pytest.importorskip("sassy")
    from fast_forward.sasa import STATISTICS
    assert set(STATISTICS) == set(sassy.STATISTICS)


@needs_data
def test_cg_surface_statistic(tmp_path):
    # an explicit statistic ignores the mapped structure
    u = mda.Universe(str(GSH_ASSESS_TPR), str(GSH_ASSESS_TRAJ))
    mol_sasa = MoleculeSASA(u.select_atoms("moltype GSH"), "GSH", cg=True).run()
    u.atoms[:6].write(str(tmp_path / "GSH_sasa_mapped.pdb"))
    surface = mol_sasa.write_cg_surface(reference_dir=tmp_path, outdir=tmp_path, statistic="min")
    assert surface.rmsd is None
    assert np.isclose(mol_sasa.series[surface.frame], mol_sasa.series.min())


@needs_data
def test_cg_surface_mismatched_reference(tmp_path, capsys):
    u = mda.Universe(str(GSH_ASSESS_TPR), str(GSH_ASSESS_TRAJ))
    mol_sasa = MoleculeSASA(u.select_atoms("moltype GSH"), "GSH", cg=True).run()
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
    # the SASA of every bead of the mapping, by position, adding up to that of the molecule
    beads = ["CAC1", "AMC1", "AMD1", "SUL1", "AMD2", "CAC2"]
    table = (ref / "GSH_sasa_beads.dat").read_text().splitlines()
    assert table[1:] == [f"{k}  {bead}  real  GSH_1_{bead}_sasa_distr.dat" for k, bead in enumerate(beads, 1)]
    ref_beads = _bead_sasa(ref)
    assert list(ref_beads) == beads
    assert np.allclose(sum(ref_beads.values()), np.loadtxt(ref / "GSH_sasa.dat"))
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
    assert set(bead_scores) == {f"1_{bead}" for bead in beads} # named as in the CG itp
    for bead, (h_score, score, cg_mean, ref_mean) in bead_scores.items():
        assert 0.0 <= h_score <= 1.0 and 0.0 <= score <= 1.0
        # the reference mean is that of the bead SASA written by ff_map
        assert np.isclose(ref_mean, ref_beads[bead[2:]].mean(), atol=0.01)
    assert "named differently" not in proc.stdout and "not assessed" not in proc.stdout
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
    cg_beads = _bead_sasa(dists)
    assert list(cg_beads) == beads
    assert np.allclose(sum(cg_beads.values()), np.loadtxt(dists / "GSH_sasa.dat"))


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


@needs_data
def test_ff_assess_sasa_missing_reference(tmp_path):
    ref = tmp_path / "ref"
    shutil.copytree(GSH_ASSESS_REFERENCE, ref) # bonded references only
    ff_assess = [sys.executable, '-m', 'fast_forward.bin.ff_assess',
                 '-f', str(GSH_ASSESS_TRAJ), '-s', str(GSH_ASSESS_TPR), '-i', str(GSH_ITP_OUTPUT),
                 '-d', str(ref), '-interactions-only', '-sasa']
    proc = subprocess.run(ff_assess, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    # a molecule without a reference is skipped with a warning, as a bead without one
    assert proc.returncode == 0, proc.stderr
    assert "No reference SASA for GSH (GSH_sasa_distr.dat" in proc.stdout
    assert "No SASA could be assessed" in proc.stdout


def _run_ff_map_beads(map_file, outdir, extra=()):
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(map_file),
              '-mols', 'LIG:GSH', '-o', 'mapped.xtc', '-sasa', str(outdir), *extra]
    proc = subprocess.run(ff_map, cwd=outdir, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc, np.loadtxt(outdir / "GSH_sasa.dat"), _bead_sasa(outdir)


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
    assert sorted(full) == ["AMC1", "AMD1", "AMD2", "CAC1", "CAC2", "SUL1"]

    # C6 shared by AMD1 and SUL1 with weights 0.3/0.7, H16 in no bead
    def shared(line, tokens):
        if tokens[:2] == ['9', 'C6']:
            return '9\tC6\t0.3 AMD1 0.7 SUL1'
        return None if tokens[:2] == ['36', 'H16'] else line
    proc, shared_sasa, modified = _run_ff_map_beads(_modified_map(tmp_path / "shared.map", shared),
                                                    tmp_path / "shared")
    assert "1 atom(s) of GSH are in no bead" in proc.stdout and "H16" in proc.stdout
    assert np.allclose(shared_sasa, sasa) # the molecule does not depend on the mapping
    gained = modified["AMD1"] - full["AMD1"]
    assert np.allclose(gained, full["SUL1"] - modified["SUL1"]) # half of C6 moved, despite 0.3/0.7
    assert np.any(gained > 0)
    for bead in ("CAC1", "AMC1", "CAC2"):
        assert np.allclose(modified[bead], full[bead])
    lost = full["AMD2"] - modified["AMD2"] # H16
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
    # the bead table lists VS1 as a dummy at its position (the last bead of the mapping)
    table = (tmp_path / "vs" / "GSH_sasa_beads.dat").read_text().splitlines()
    assert table[-1] == "7  VS1  dummy  -"
    assert np.allclose(vs_sasa, sasa)
    for bead in full:
        assert np.allclose(vs[bead], full[bead])

    # -sasa-ignore replaces the dummy detection: CAC1 (position 1) is ignored, VS1 is a real bead,
    # sharing all atoms of AMD1 and SUL1 (half of each to VS1); the atoms of CAC1 go to no bead
    (tmp_path / "ignore").mkdir()
    proc, ignore_sasa, ignored = _run_ff_map_beads(tmp_path / "vs.map", tmp_path / "ignore",
                                                   extra=['-sasa-ignore', '1'])
    assert "No SASA assigned to the bead(s) 1_CAC1 of GSH, ignored with -sasa-ignore" in proc.stdout
    assert "dummy bead(s) 1_VS1" not in proc.stdout
    assert "of GSH are in no bead" in proc.stdout
    table = (tmp_path / "ignore" / "GSH_sasa_beads.dat").read_text().splitlines()
    assert table[1] == "1  CAC1  dummy  -" and table[-1] == "7  VS1  real  GSH_1_VS1_sasa_distr.dat"
    assert sorted(ignored) == sorted(set(full) - {"CAC1"} | {"VS1"})
    assert np.allclose(ignore_sasa, sasa) # the molecule does not depend on the beads
    assert np.allclose(ignored["VS1"], ignored["AMD1"] + ignored["SUL1"])
    assert np.allclose(ignored["AMD1"] + ignored["SUL1"] + ignored["VS1"], full["AMD1"] + full["SUL1"])

    # a position beyond the beads of the molecule is a command line error
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map', '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR),
              '-m', str(GSH_MAP), '-mols', 'LIG:GSH', '-o', 'mapped.xtc', '-sasa', '.', '-sasa-ignore', '2 9']
    proc = subprocess.run(ff_map, cwd=tmp_path / "ignore", timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 2
    assert "GSH has 6 bead(s), so there is no bead at position(s) 9" in proc.stderr


@needs_data
def test_set_cg_beads_skip_dummies():
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
    with pytest.raises(ValueError, match="after run"):
        mol_sasa.set_cg_beads(block)
    mol_sasa.run()
    mol_sasa.set_cg_beads(block)
    assert mol_sasa.bead_names == names[:5] # the virtual site (bead 4) is kept
    assert mol_sasa.bead_series.shape == (5, u.trajectory.n_frames)
    assert np.allclose(mol_sasa.bead_series.sum(axis=0), mol_sasa.series) # the dummy has no SASA
    assert np.array_equal(mol_sasa.bead_series, mol_sasa.sasa.results.atom_sasa[:5]) # one molecule
    assert [bead.dummy for bead in mol_sasa.beads] == [False] * 5 + [True]
    with pytest.raises(ValueError, match="has 6 beads, but the itp has 5"):
        block.remove_node(list(block.nodes)[-1])
        mol_sasa.set_cg_beads(block)


def _gsh_itp(path, edit):
    """GSH.itp with the [ atoms ] lines (tokens) passed through ``edit``."""
    lines, in_atoms = [], False
    for line in Path(GSH_ITP_OUTPUT).read_text().splitlines():
        if line.startswith("["):
            in_atoms = line.split()[1] == "atoms"
        elif in_atoms and line.strip() and not line.startswith(";"):
            line = " ".join(edit(line.split()))
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path


@needs_data
def test_ff_assess_bead_matching(tmp_path):
    """
    ff_assess matches the beads to the reference by their position within the molecule
    (bead table of ff_map), whatever the resids and names of the CG itp; a renamed bead
    or a different number of beads is warned about, and a reference without bead table
    still assesses the molecule.
    """
    ref = tmp_path / "ref"
    shutil.copytree(GSH_ASSESS_REFERENCE, ref)
    ff_map = [sys.executable, '-m', 'fast_forward.bin.ff_map',
              '-f', str(GSH_AA_TRAJ), '-s', str(GSH_AA_TPR), '-m', str(GSH_MAP),
              '-mols', 'LIG:GSH', '-o', 'mapped.xtc', '-sasa', str(ref)]
    proc = subprocess.run(ff_map, cwd=tmp_path, timeout=600, capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    def assess(name, itp=GSH_ITP_OUTPUT, reference=ref):
        run_dir = tmp_path / name
        run_dir.mkdir()
        ff_assess = [sys.executable, '-m', 'fast_forward.bin.ff_assess',
                     '-f', str(GSH_ASSESS_TRAJ), '-s', str(GSH_ASSESS_TPR), '-i', str(itp),
                     '-d', str(reference), '-interactions-only', '-sasa']
        proc = subprocess.run(ff_assess, cwd=run_dir, timeout=600, capture_output=True, text=True, env=ENV)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = (run_dir / "report_sasa.out").read_text()
        assert "GSH" in report # the molecule is always assessed
        bead_part = report.split(" [ Bead SASA of GSH ]")[1] if "Bead SASA" in report else ""
        matches = [SASA_LINE.match(line) for line in bead_part.splitlines()]
        return proc.stdout, [m.group(1) for m in matches if m]

    # PDB-like resids, with a gap: the beads are still found, named as in the itp
    renumbered = _gsh_itp(tmp_path / "renumbered.itp",
                          lambda t: t[:2] + ["101" if int(t[0]) <= 3 else "205"] + t[3:])
    stdout, beads = assess("renumbered", renumbered)
    assert beads == ["101_CAC1", "101_AMC1", "101_AMD1", "205_SUL1", "205_AMD2", "205_CAC2"]
    assert "WARNING" not in stdout

    # a bead named differently from the mapping: warned about, still assessed by position
    renamed = _gsh_itp(tmp_path / "renamed.itp", lambda t: t[:4] + (["CX2"] if t[4] == "CAC2" else t[4:5]) + t[5:])
    stdout, beads = assess("renamed", renamed)
    assert "named differently (6: itp CX2, mapping CAC2)" in stdout
    assert len(beads) == 6 and "1_CX2" in beads

    # a reference with another number of beads: no bead is assessed
    fewer = tmp_path / "ref_fewer"
    shutil.copytree(ref, fewer)
    table = fewer / "GSH_sasa_beads.dat"
    table.write_text("\n".join(table.read_text().splitlines()[:-1]) + "\n")
    stdout, beads = assess("fewer", reference=fewer)
    assert "The CG itp has 6 beads for GSH, but the reference" in stdout and "has 5" in stdout
    assert beads == []

    # a reference without bead table (e.g. from an older ff_map): no bead is assessed
    (ref / "GSH_sasa_beads.dat").unlink()
    stdout, beads = assess("no_table")
    assert "No bead table" in stdout and "Rerun 'ff_map -sasa'" in stdout
    assert beads == []


