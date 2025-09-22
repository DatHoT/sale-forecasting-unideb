import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREDICTOR_PATH = os.getenv("PREDICTOR_PATH", str(BASE_DIR / "models" / "tft_store"))
DEFAULT_HIST_PATH = os.getenv("HIST_PATH", str(BASE_DIR / "data" / "xgb_prepared_data_v2.pkl"))
DEFAULT_REQ_PATH  = os.getenv("REQ_PATH",  str(BASE_DIR / "data" / "xgb_prepared_data_20250626_20250831.pkl"))

FILL_MISSING_SERIES_WITH_ZERO_HISTORY = os.getenv("FILL_MISSING_ZERO", "1") == "1"
MIN_FAKE_HISTORY_WEEKS = int(os.getenv("MIN_FAKE_HISTORY_WEEKS", "12"))
FUTURE_WEEKS = int(os.getenv("FUTURE_WEEKS", "2"))
