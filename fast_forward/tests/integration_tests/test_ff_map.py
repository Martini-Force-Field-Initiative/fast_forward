

from fast_forward.tests.datafiles import (GSH_AA_TPR, GSH_AA_TRAJ, GSH_CG_GRO, GSH_CG_TRAJ, GSH_MAP,
                                          GSH_SASA_AA, GSH_SASA_AA_DISTR, HAVE_EXAMPLES_DATA, MISSING_EXAMPLES_DATA_REASON)

import subprocess
import numpy as np
from MDAnalysis import Universe
import shutil
import pytest

pytestmark = pytest.mark.skipif(not HAVE_EXAMPLES_DATA, reason=MISSING_EXAMPLES_DATA_REASON)

@pytest.mark.parametrize('command_list, output_top, output_traj',
                         ((['-f', GSH_AA_TRAJ,
                           '-s', GSH_AA_TPR,
                           '-m', GSH_MAP,
                           '-mols', 'GSH',
                           '-o', 'mapped.xtc'
                            ], GSH_CG_GRO, GSH_CG_TRAJ),
                         ))
def test_ff_map(tmp_path, monkeypatch, command_list, output_top, output_traj):

    monkeypatch.chdir(tmp_path)
    ff_map = shutil.which('ff_map')
    command = [ff_map, ] + command_list

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    exit_code = proc.returncode
    if exit_code:
        print(proc.stdout)
        print(proc.stderr)
        assert not exit_code

    # check we generated some files
    files = list(tmp_path.iterdir())
    assert files
    # expect a .gro and .xtc from this command
    assert len(files) == 2

    reference_universe = Universe(output_top, output_traj)
    new_universe = Universe([i for i in files if '.gro' in i.name][0],
                            [i for i in files if '.xtc' in i.name][0],
                            )

    # assert that the coordinates we map to are the same as in the reference
    for ts, ts0 in zip(reference_universe.trajectory, new_universe.trajectory):
        assert np.all(np.isclose(reference_universe.atoms.positions, new_universe.atoms.positions))



@pytest.mark.parametrize('prefix', ['', 'test_'])
def test_ff_map_sasa(tmp_path, monkeypatch, prefix):
    """
    '-sasa' computes the atomistic SASA of each molecule before mapping and
    writes the time series and its distribution, which serve as the reference
    for 'ff_assess -sasa'.
    """
    monkeypatch.chdir(tmp_path)
    ff_map = shutil.which('ff_map')
    command = [ff_map, '-f', GSH_AA_TRAJ, '-s', GSH_AA_TPR, '-m', GSH_MAP,
               '-mols', 'GSH', '-o', 'mapped.xtc', '-sasa', '-pref', prefix]

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    exit_code = proc.returncode
    if exit_code:
        print(proc.stdout)
        print(proc.stderr)
    assert not exit_code

    # the mapped trajectory is still written alongside the SASA files
    assert {i.name for i in tmp_path.iterdir()} == {'mapped.gro', 'mapped.xtc',
                                                   f'{prefix}GSH_sasa.dat',
                                                   f'{prefix}GSH_sasa_distr.dat'}
    # the atomistic radii are cited
    assert 'van der Waals radii' in proc.stdout

    # the time series and distribution match the stored GSH reference
    sasa = np.loadtxt(tmp_path / f'{prefix}GSH_sasa.dat')
    assert np.allclose(sasa, np.loadtxt(GSH_SASA_AA), rtol=0, atol=5e-3)
    distr = np.loadtxt(tmp_path / f'{prefix}GSH_sasa_distr.dat')
    reference_distr = np.loadtxt(GSH_SASA_AA_DISTR)
    assert np.allclose(distr.T[0], reference_distr.T[0])
    # compare cumulative distributions, so that frames close to a bin edge
    # (bins are 0.01 nm^2 wide) may move into the neighbouring bin
    assert np.allclose(np.cumsum(distr.T[1]) / distr.T[1].sum(),
                       np.cumsum(reference_distr.T[1]) / reference_distr.T[1].sum(),
                       rtol=0, atol=0.02)


def _check_dots_pdb(path, n_structure):
    """
    The dots PDB holds the structure as ATOM records followed by the dots as
    HETATM records with atom and residue name DOT, and no CONECT records.
    """
    lines = path.read_text().splitlines()
    atom_serials = [int(line[6:11]) for line in lines if line.startswith('ATOM')]
    dot_serials = [int(line[6:11]) for line in lines if line.startswith('HETATM')]
    assert atom_serials == list(range(1, n_structure + 1))
    assert dot_serials and dot_serials[0] == n_structure + 1
    assert {line[17:20] for line in lines if line.startswith('HETATM')} == {'DOT'}
    assert {line[12:16].strip() for line in lines if line.startswith('HETATM')} == {'DOT'}
    assert not any(line.startswith('CONECT') for line in lines)


def test_ff_map_sasa_surface(tmp_path, monkeypatch):
    """
    '-sasa-surface' writes the atomistic structure with its surface dots for
    the frame with median SASA, and the mapped CG structure of that same frame.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'sasa').mkdir()
    ff_map = shutil.which('ff_map')
    command = [ff_map, '-f', GSH_AA_TRAJ, '-s', GSH_AA_TPR, '-m', GSH_MAP,
               '-mols', 'GSH', '-o', 'mapped.xtc', '-sasa', 'sasa', '-sasa-surface']

    proc = subprocess.run(command, cwd='.', timeout=120, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    if proc.returncode:
        print(proc.stdout)
        print(proc.stderr)
    assert not proc.returncode

    sasa_dir = tmp_path / 'sasa'
    assert {i.name for i in sasa_dir.iterdir()} == {'GSH_sasa.dat', 'GSH_sasa_distr.dat',
                                                   'GSH_sasa_dots_AA.pdb',
                                                   'GSH_sasa_mapped.pdb'}
    # the selected frame has the SASA closest to the median
    sasa = np.loadtxt(sasa_dir / 'GSH_sasa.dat')
    frame = int(np.argmin(np.abs(sasa - np.median(sasa))))
    assert f'surface of frame {frame} for GSH' in proc.stdout

    # the mapped structure is that frame of the mapped trajectory
    mapped = Universe(str(sasa_dir / 'GSH_sasa_mapped.pdb'))
    cg = Universe('mapped.gro', 'mapped.xtc')
    cg.trajectory[frame]
    assert list(mapped.atoms.names) == list(cg.atoms.names)
    assert np.allclose(mapped.atoms.positions, cg.atoms.positions, atol=1e-2)

    # the atomistic structure in the dots file lies under the mapped one
    _check_dots_pdb(sasa_dir / 'GSH_sasa_dots_AA.pdb', 36)

    # the citations of fast_forward and the atomistic radii are in the remarks
    remarks = ' '.join(line[9:] for line in (sasa_dir / 'GSH_sasa_dots_AA.pdb').read_text().splitlines()
                       if line.startswith('REMARK'))
    for citation in ['10.1021/acs.jcim.6c01107', '10.1021/jp953141+']:
        assert citation in remarks
    assert '10.1021/acs.jcim.6c01107' in (sasa_dir / 'GSH_sasa_mapped.pdb').read_text()
    assert all(len(line) <= 80 for path in sasa_dir.glob('*.pdb') for line in path.read_text().splitlines())
    aa = Universe(str(sasa_dir / 'GSH_sasa_dots_AA.pdb')).select_atoms('record_type ATOM')
    assert aa.n_atoms == 36
    assert np.allclose(aa.positions.mean(axis=0), mapped.atoms.positions.mean(axis=0), atol=2.0)


def test_ff_map_sasa_surface_requires_sasa(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ff_map = shutil.which('ff_map')
    command = [ff_map, '-f', GSH_AA_TRAJ, '-s', GSH_AA_TPR, '-m', GSH_MAP,
               '-mols', 'GSH', '-o', 'mapped.xtc', '-sasa-surface']
    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode != 0
    assert '-sasa-surface requires -sasa' in proc.stderr
