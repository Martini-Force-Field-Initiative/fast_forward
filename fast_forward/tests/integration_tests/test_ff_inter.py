import pytest

from fast_forward.tests.datafiles import (GSH_CG_TPR, GSH_CG_TRAJ, GSH_ITP_INTIIAL, GSH_ITP_OUTPUT, GSH_DISTS,
                                          HAVE_EXAMPLES_DATA, MISSING_EXAMPLES_DATA_REASON)

import subprocess
import numpy as np
from vermouth.tests.helper_functions import find_in_path
from glob import glob
from vermouth.tests.integration_tests.test_integration import compare_itp
from pathlib import Path

pytestmark = pytest.mark.skipif(not HAVE_EXAMPLES_DATA, reason=MISSING_EXAMPLES_DATA_REASON)

@pytest.mark.parametrize('command_list, reference_distributions',
                             ((['-f', GSH_CG_TRAJ,
                                '-s', GSH_CG_TPR,
                                '-i', GSH_ITP_INTIIAL,
                                '-max-dihedral', '10',
                                '-interactions', 'comments',
                                '-dists',], GSH_DISTS),)
                         )
def test_ff_inter(tmp_path, monkeypatch, command_list, reference_distributions):

    monkeypatch.chdir(tmp_path)
    ff_inter = find_in_path(names=('ff_inter', ))

    command = [ff_inter, ] + command_list

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

    reference_dats = glob(str(reference_distributions))
    output_dats = [i for i in files if i.suffix == '.dat']

    # check we have the same set of output distributions as expected
    assert set([Path(i).stem for i in reference_dats]) == set([i.stem for i in output_dats])
    # ensure identical distributions for each
    for f0, f1 in zip(sorted(list(set([Path(i) for i in reference_dats]))),
                      sorted(list(set([i.name for i in output_dats])))):
        data0 = np.loadtxt(f0)
        data1 = np.loadtxt(f1)
        assert np.allclose(data0, data1, atol=1e-3)

    # compare the output itps
    output_itp = [i for i in files if i.suffix == '.itp'][0]
    compare_itp(GSH_ITP_OUTPUT, output_itp)
