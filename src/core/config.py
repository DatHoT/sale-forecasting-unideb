import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREDICTOR_PATH = os.getenv("PREDICTOR_PATH", str(BASE_DIR / "models" / "tft_store"))
DEFAULT_HIST_PATH = os.getenv("HIST_PATH", str(BASE_DIR / "data" / "xgb_prepared_data_v2.pkl"))
DEFAULT_REQ_PATH  = os.getenv("REQ_PATH",  str(BASE_DIR / "data" / "xgb_prepared_data_20250626_20250831.pkl"))
AUX_TRUTH_PATH = os.getenv("REQ_PATH",  str(BASE_DIR / "data" / "xgb_prepared_data_20250626_20250831.pkl"))
PREPATCHED_PAD_PATH = os.getenv("PREPATCHED_PAD_PATH", "data/prepatched_hist.parquet")
MIN_FAKE_HISTORY_WEEKS = int(os.getenv("MIN_FAKE_HISTORY_WEEKS", "12"))
ITEM_MAPPING_PATH = os.getenv("ITEM_MAPPING_PATH", "data/item_mapping.parquet")
FILL_MISSING_SERIES_WITH_ZERO_HISTORY = os.getenv("FILL_MISSING_ZERO", "1") == "1"
MIN_FAKE_HISTORY_WEEKS = int(os.getenv("MIN_FAKE_HISTORY_WEEKS", "12"))
FUTURE_WEEKS = int(os.getenv("FUTURE_WEEKS", "2"))
EXPORT_DIR = os.getenv("EXPORT_DIR", "exports")
DATASET_PATH = str(BASE_DIR / "data" / "xgb_prepared_data_v2.pkl")
KNOWLEDGE_BASE_MAX_DATE = "2025-05-26"

# NEW: context window = match training (12 weeks by default)
CONTEXT_WEEKS = int(os.getenv("CONTEXT_WEEKS", "12"))
#---------------
# --- Two-stage (zero-inflation) toggles ---
ENABLE_TWO_STAGE = True          # if a classifier exists, use it; else skip
TWO_STAGE_PRED_MODE = "gate"     # "gate" -> (p>=P_CUT)*reg; "soft" -> p*reg
P_CUT = 0.25                     # classify nonzero when p >= 0.25
MIN_CLIP = 0.0                   # after combination, floor at 0
CLASSIFIER_PATH = str(BASE_DIR / "binary_classifier" / "classifier.pkl")
# Optional: class weight and threshold are training-time concerns; keep here for provenance
CLASS_IMBALANCE_POS_WEIGHT = 2.0