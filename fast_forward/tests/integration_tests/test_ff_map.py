

from fast_forward.tests.datafiles import (GSH_AA_TPR, GSH_AA_TRAJ, GSH_CG_GRO, GSH_CG_TRAJ, GSH_MAP,
                                          HAVE_EXAMPLES_DATA, MISSING_EXAMPLES_DATA_REASON)

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
                           '-mols', 'LIG',
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

