
from pathlib import Path

# the GSH example/test data lives in examples/GSH at the repository root,
# alongside the other worked examples, rather than being packaged with the
# installed library.

EXAMPLES_DATA = Path(__file__).resolve().parents[2] / 'examples' / 'GSH'

HAVE_EXAMPLES_DATA = EXAMPLES_DATA.is_dir()
MISSING_EXAMPLES_DATA_REASON = (
    f"GSH example data not found at {EXAMPLES_DATA}. This data lives in the "
    "examples/ folder of the fast_forward git repository and is not bundled "
    "with an installed copy of the package; run these tests from a source "
    "checkout instead (e.g. `pip install -e .`)."
)

GSH_AA_TRAJ = EXAMPLES_DATA / 'AA/atomistic.xtc'
GSH_AA_TPR = EXAMPLES_DATA / 'AA/atomistic.tpr'

GSH_CG_TRAJ = EXAMPLES_DATA / 'CG/mapped.xtc'
GSH_CG_GRO = EXAMPLES_DATA / 'CG/mapped.gro'
GSH_CG_TPR = EXAMPLES_DATA / 'interactions/mapped.tpr'

GSH_MAP = EXAMPLES_DATA / 'GSH.map'

GSH_ITP_INTIIAL = EXAMPLES_DATA / 'GSH_initial.itp'
GSH_ITP_OUTPUT = EXAMPLES_DATA / 'interactions/GSH.itp'
GSH_DISTS = EXAMPLES_DATA / 'interactions/*.dat'

GSH_ASSESS_TRAJ = EXAMPLES_DATA / 'assessment/simulated.xtc'
GSH_ASSESS_TPR = EXAMPLES_DATA / 'assessment/simulated.tpr'
GSH_ASSESS_REFERENCE = EXAMPLES_DATA / 'assessment/reference'

del Path
