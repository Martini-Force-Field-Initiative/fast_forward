"""
Tests of the SASA code that do not need sassy (distributions, -sasa-stat, -mols, the dummy
beads of a mapping, the bead table, the scores and the report), so they also run
without the 'sasa' extra; the SASA calculation itself is tested in test_sasa.py.
"""
import subprocess
import sys

import numpy as np
import pytest

from fast_forward.interaction_distribution import INTERACTIONS, RECOGNISED_INTERACTIONS, distribution
from fast_forward.tests.datafiles import ENV


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
    ([[], []], [False, False]),
    ([], []),
    # a dummy on all atoms: the beads sharing atom 1 are kept
    ([[0, 1], [1, 2, 3], [0, 1, 2, 3]], [False, False, True]),
])
def test_find_dummies(beads, expected):
    from fast_forward.sasa import find_dummies
    assert find_dummies(beads).tolist() == expected


def test_score_sasa_invalid():
    from fast_forward.interaction_distribution import INTERACTIONS
    from fast_forward.sasa_assessment import score_sasa
    bins = INTERACTIONS['sasa']['bins']
    centres = (bins[:-1] + bins[1:]) / 2
    n_bins = len(centres)
    gauss = np.exp(-((np.arange(n_bins) - n_bins / 2) / 5)**2)
    reference = np.column_stack([centres, gauss])
    h_score, score, _ = score_sasa(gauss, reference, 0.7)
    assert h_score == pytest.approx(0) and score == pytest.approx(0)
    # empty or NaN distributions, and a reference without spread (NaN score), are not scored
    assert score_sasa(np.zeros(n_bins), reference, 0.7) is None
    assert score_sasa(np.full(n_bins, np.nan), reference, 0.7) is None
    assert score_sasa(gauss, np.column_stack([centres, np.full(n_bins, np.nan)]), 0.7) is None
    delta = np.zeros(n_bins)
    delta[n_bins // 2] = 1
    with np.errstate(invalid='ignore', divide='ignore'):
        assert score_sasa(delta, np.column_stack([centres, delta]), 0.7) is None


def test_read_bead_table(tmp_path):
    from fast_forward.sasa import read_bead_table, BEAD_TABLE_HEADER, BeadRow
    table = tmp_path / "GSH_sasa_beads.dat"
    table.write_text(f"{BEAD_TABLE_HEADER}\n1  CAC1  real  GSH_1_CAC1_sasa_distr.dat\n\n2  VS1  dummy  -\n")
    assert read_bead_table(table) == [BeadRow(1, "CAC1", False, "GSH_1_CAC1_sasa_distr.dat"),
                                      BeadRow(2, "VS1", True, None)]
    # malformed lines and positions that are not 1, 2, ... are errors
    table.write_text("1  CAC1  maybe  GSH_1_CAC1_sasa_distr.dat\n")
    with pytest.raises(ValueError, match="line 1"):
        read_bead_table(table)
    table.write_text("2  CAC1  real  GSH_1_CAC1_sasa_distr.dat\n")
    with pytest.raises(ValueError, match="positions are not 1"):
        read_bead_table(table)


def test_report_sasa(tmp_path, monkeypatch):
    from fast_forward.report import report_sasa
    monkeypatch.chdir(tmp_path)
    report_sasa({"GSH": 0.1, "LIG": 0.3}, {"GSH": 0.2, "LIG": 0.4}, {"GSH": (6.0, 6.5), "LIG": (3.0, 2.0)},
                {"GSH": {"1_CAC1": (0.05, 0.1, 1.0, 1.1), "1_AMC1": (0.15, 0.3, 2.0, 1.8)}})
    report = (tmp_path / "report_sasa.out").read_text()
    assert "Overall Score : 0.3 ± 0.0707" in report # mean and standard error of the molecule scores
    assert "\tGSH                 : 0.10 (0.20)   6.00 / 6.50" in report
    assert " [ Bead SASA of GSH ]" in report and "Overall Bead Score : 0.2 ± 0.0707" in report
    assert "\t1_AMC1              : 0.15 (0.30)   2.00 / 1.80" in report
    assert "Bead SASA of LIG" not in report
