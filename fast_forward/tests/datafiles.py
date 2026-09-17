
from pathlib import Path

try:
    import pkg_resources
except ImportError:
    TEST_DATA = Path(__file__).resolve().parent / 'data'
else:
    TEST_DATA = Path(pkg_resources.resource_filename('fast_forward.tests', 'data'))
    del pkg_resources


GSH_AA_TRAJ = TEST_DATA / 'GSH/AA/atomistic.xtc'
GSH_AA_TPR = TEST_DATA / 'GSH/AA/atomistic.tpr'

GSH_CG_TRAJ = TEST_DATA / 'GSH/CG/mapped.xtc'
GSH_CG_GRO = TEST_DATA / 'GSH/CG/mapped.gro'
GSH_CG_TPR = TEST_DATA / 'GSH/interactions/mapped.tpr'

GSH_MAP = TEST_DATA / 'GSH/GSH.map'

GSH_ITP_INTIIAL = TEST_DATA / 'GSH/GSH_initial.itp'
GSH_ITP_OUTPUT = TEST_DATA / 'GSH/interactions/GSH.itp'
GSH_DISTS = TEST_DATA / 'GSH/interactions/*.npy'

GSH_ASSESS_TRAJ = TEST_DATA / 'GSH/assessment/simulated.xtc'
GSH_ASSESS_TPR = TEST_DATA / 'GSH/assessment/simulated.tpr'
GSH_ASSESS_REFERENCE = TEST_DATA / 'GSH/assessment/reference'

del Path