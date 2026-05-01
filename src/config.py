from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "baseline_and_data"
TRAIN_CSV = DATA_DIR / "paint_aging_trainset.csv"
TEST_CSV = DATA_DIR / "paint_aging_testset.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGS_DIR = OUTPUTS_DIR / "figs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSION_CSV = OUTPUTS_DIR / "predict_out.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
TAU_VALUES = (2.0, 7.0, 21.0)

TARGET_COL = "dietaE"
GROUP_COLS = ["sample", "aging_condition"]
BASE_FEATURES = ["aging_time_day", "L0", "a0", "b0"]


def ensure_output_dirs() -> None:
    for d in (OUTPUTS_DIR, METRICS_DIR, FIGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
