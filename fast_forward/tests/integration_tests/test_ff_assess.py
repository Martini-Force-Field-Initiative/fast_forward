import re
import shutil
import subprocess
from pathlib import Path

import pytest
import vermouth.forcefield
from vermouth.tests.helper_functions import find_in_path

from fast_forward.itp_parser_sub import read_itp
from fast_forward.tests.datafiles import (GSH_ASSESS_TRAJ, GSH_ASSESS_TPR, GSH_ITP_OUTPUT,
                                          GSH_ASSESS_REFERENCE, HAVE_EXAMPLES_DATA,
                                          MISSING_EXAMPLES_DATA_REASON)

pytestmark = pytest.mark.skipif(not HAVE_EXAMPLES_DATA, reason=MISSING_EXAMPLES_DATA_REASON)

SCORE_LINE = re.compile(r'^\t(\S+)\s*:\s*([-\d.]+)\s*\(([-\d.]+)\)')


@pytest.mark.parametrize('command_list', [['-f', GSH_ASSESS_TRAJ,
                                           '-s', GSH_ASSESS_TPR,
                                           '-i', GSH_ITP_OUTPUT,
                                           '-d', GSH_ASSESS_REFERENCE]])
def test_ff_assess(tmp_path, monkeypatch, command_list):

    monkeypatch.chdir(tmp_path)
    ff_assess = find_in_path(names=('ff_assess', ))

    command = [ff_assess, ] + command_list

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    exit_code = proc.returncode
    if exit_code:
        print(proc.stdout)
        print(proc.stderr)
    assert not exit_code

    interactions_report = tmp_path / 'report_interactions.out'
    distances_report = tmp_path / 'report_distances.out'
    assert interactions_report.exists()
    assert distances_report.exists()

    ff = vermouth.forcefield.ForceField("dummy")
    with open(GSH_ITP_OUTPUT) as itp_file:
        read_itp(itp_file.readlines(), ff)
    _, block = next(iter(ff.blocks.items()))

    expected_atoms = {f"{data['resid']}_{data['atomname']}" for _, data in block.nodes(data=True)}

    expected_groups = {'bonds': set(), 'angles': set(), 'dihedrals': set()}
    for section in ('bonds', 'constraints', 'angles', 'dihedrals'):
        key = 'bonds' if section == 'constraints' else section
        for interaction in block.interactions.get(section, []):
            comment = interaction.meta.get('comment')
            if comment:
                expected_groups[key].add(comment)

    # every interaction group defined in the itp, per interaction type, is
    # scored and reported: nothing was silently skipped or spuriously added.
    found_groups = {'bonds': set(), 'angles': set(), 'dihedrals': set()}
    current_type = None
    for line in interactions_report.read_text().splitlines():
        stripped = line.strip()
        if stripped in found_groups:
            current_type = stripped
            continue
        match = SCORE_LINE.match(line)
        if match and current_type:
            name, h_score, score = match.groups()
            found_groups[current_type].add(name)
            # scores are bounded distances between distributions
            assert 0.0 <= float(h_score) <= 1.0
            assert 0.0 <= float(score) <= 1.0

    for inter_type in found_groups:
        assert found_groups[inter_type] == expected_groups[inter_type]
        assert found_groups[inter_type]

    # the distance score matrix covers every atom defined in the itp
    distance_lines = distances_report.read_text().splitlines()
    marker = next(i for i, l in enumerate(distance_lines) if 'no overlap' in l)
    header_idx = next(i for i in range(marker + 1, len(distance_lines)) if distance_lines[i].strip())
    atom_names = distance_lines[header_idx].split()

    assert set(atom_names) == expected_atoms

    for offset, row_name in enumerate(atom_names, start=1):
        row_tokens = distance_lines[header_idx + offset].split()
        assert row_tokens[0] == row_name
        values = row_tokens[1:]
        assert len(values) == len(atom_names)
        for score in values:
            assert 0.0 <= float(score) <= 1.0


@pytest.mark.parametrize('missing_file, interactions_report_expected', [
    # missing bonds/angles/dihedrals reference: ff_assess fails before it
    # even gets to write report_interactions.out
    ('CAC1_AMC1_bonds_distr.dat', False),
    # missing distances reference: report_interactions.out is written first,
    # then ff_assess fails while building the distance score matrix
    ('1_CAC1_1_AMC1_distances_distr.dat', True),
])
def test_ff_assess_missing_reference(tmp_path, monkeypatch, missing_file, interactions_report_expected):
    """
    ff_assess must fail loudly when a reference distribution is missing,
    rather than silently reporting a 0.00 ("identical") score for a
    comparison that never actually happened.
    """
    monkeypatch.chdir(tmp_path)
    ff_assess = find_in_path(names=('ff_assess', ))

    broken_reference = tmp_path / 'reference'
    broken_reference.mkdir()
    for reference_file in Path(GSH_ASSESS_REFERENCE).glob('*.dat'):
        if reference_file.name != missing_file:
            shutil.copy(reference_file, broken_reference / reference_file.name)

    command = [ff_assess, '-f', GSH_ASSESS_TRAJ, '-s', GSH_ASSESS_TPR,
              '-i', GSH_ITP_OUTPUT, '-d', broken_reference]

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    assert proc.returncode != 0
    assert 'Error:' in proc.stderr
    assert missing_file in proc.stderr
    # a clean, one-line error message, not a raw Python traceback
    assert 'Traceback' not in proc.stderr

    # no misleading scores were written out for the comparison that failed
    interactions_report = tmp_path / 'report_interactions.out'
    distances_report = tmp_path / 'report_distances.out'
    assert interactions_report.exists() == interactions_report_expected
    assert not distances_report.exists()


def test_ff_assess_no_distance_references(tmp_path, monkeypatch):
    """
    -dist-matrix is opt-in for ff_inter, so a reference folder built without
    it has no '*_distances_distr.dat' files at all. ff_assess should treat
    this as an intentional opt-out and skip the distance assessment with a
    clear warning, rather than failing like it does for a partial/broken
    reference folder (see test_ff_assess_missing_reference).
    """
    monkeypatch.chdir(tmp_path)
    ff_assess = find_in_path(names=('ff_assess', ))

    reference_without_distances = tmp_path / 'reference'
    reference_without_distances.mkdir()
    for reference_file in Path(GSH_ASSESS_REFERENCE).glob('*.dat'):
        if 'distances' not in reference_file.name:
            shutil.copy(reference_file, reference_without_distances / reference_file.name)

    command = [ff_assess, '-f', GSH_ASSESS_TRAJ, '-s', GSH_ASSESS_TPR,
              '-i', GSH_ITP_OUTPUT, '-d', reference_without_distances]

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    assert proc.returncode == 0
    assert 'distances_distr.dat' in proc.stdout
    assert '-dist-matrix' in proc.stdout

    # the interaction assessment still runs and is reported; the distance
    # assessment is cleanly skipped rather than failing
    interactions_report = tmp_path / 'report_interactions.out'
    distances_report = tmp_path / 'report_distances.out'
    assert interactions_report.exists()
    assert not distances_report.exists()


def test_ff_assess_interactions_only(tmp_path, monkeypatch):
    """
    '-interactions-only' should skip the distance assessment even when a
    complete reference (including '*_distances_distr.dat' files) is
    available, distinguishing a deliberate opt-out from the automatic skip
    in test_ff_assess_no_distance_references.
    """
    monkeypatch.chdir(tmp_path)
    ff_assess = find_in_path(names=('ff_assess', ))

    command = [ff_assess, '-f', GSH_ASSESS_TRAJ, '-s', GSH_ASSESS_TPR,
              '-i', GSH_ITP_OUTPUT, '-d', GSH_ASSESS_REFERENCE, '-interactions-only']

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    assert proc.returncode == 0
    assert "Skipping distance assessment ('-interactions-only' given)." in proc.stdout

    interactions_report = tmp_path / 'report_interactions.out'
    distances_report = tmp_path / 'report_distances.out'
    assert interactions_report.exists()
    assert not distances_report.exists()
