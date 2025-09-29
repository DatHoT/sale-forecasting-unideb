# Build a fully padded, calendarized, exog-merged history for ALL series in KB.
# Output: Parquet/PKL that the API can slice per request (fast).

import os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

# -------- config --------
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/xgb_prepared_data_v2.pkl",
)  # your KB
DATASET_PATH1 = os.getenv(
    "DATASET_PATH1",
    "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/xgb_prepared_data_20250626_20250831.pkl",
)
DATASET_PATH2 = os.getenv(
    "DATASET_PATH2",
    "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/xgb_prepared_data_20250601_20250625.pkl",
)
OUT_PARQUET = os.getenv(
    "PREPATCHED_PAD_PATH",
    "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/prepatched_hist.parquet",
)  # preferred (smaller)
OUT_PICKLE = os.getenv(
    "PREPATCHED_PAD_PKL",
    "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/prepatched_hist.pkl",
)  # optional

# canonical schema (match service)
ID_COL, TIME_COL, TARGET_COL = "series_id", "week", "sales"
STATIC_COLS = ["group_id", "item_group1", "item_group2"]
CAL_KNOWN = ["month", "weekofyear", "year"]
EXOG_WEEKLY = ["temperature", "num_sunny_days", "holiday_count"]


# -------- helpers --------
def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]


def _to_wmon(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if ts.isna().any():
        bad = s[pd.isna(ts)].head(5).tolist()
        raise ValueError(f"'week' has unparseable values; examples: {bad}")
    t = pd.to_datetime(ts)
    return (t - pd.to_timedelta(t.dt.weekday.astype(int), unit="D")).dt.normalize()

# --- add this helper near the other helpers ---
def _week_key(s: pd.Series) -> pd.Series:
    """
    Robust weekly key using PeriodIndex with W-MON.
    This avoids any nanosecond/offset math differences from timedelta division.
    """
    ss = pd.to_datetime(s, errors="coerce")
    if ss.isna().any():
        bad = s[pd.isna(ss)].head(5).tolist()
        raise ValueError(f"[WEEK_KEY] Unparseable week values; examples: {bad}")
    # Period ordinals are stable integers per W-MON week
    return pd.PeriodIndex(ss, freq="W-MON").astype("int64")

def _calendarize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COL] = _to_wmon(out[TIME_COL])
    t = out[TIME_COL]
    out["month"] = t.dt.month.astype("int16")
    out["weekofyear"] = t.dt.isocalendar().week.astype("int16")
    out["year"] = t.dt.year.astype("int16")
    return out


def _strict_require(df: pd.DataFrame, required_cols: List[str], ctx: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        head_cols = df.columns.tolist()
        raise ValueError(
            f"[{ctx}] Missing required columns: {missing}\n"
            f"Available columns ({len(head_cols)}): {head_cols[:50]}"
        )


def _load_kb(path: str) -> pd.DataFrame:
    # Load raw
    df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_parquet(path)
    df = _dedup(df)

    print(f"[LOAD] KB columns ({len(df.columns)}): {df.columns.tolist()}")

    # --- derive series_id BEFORE strict checks ---
    if "series_id" not in df.columns:
        if {"partner_id", "itemcode"}.issubset(df.columns):
            df["series_id"] = df["partner_id"].astype(str) + "||" + df["itemcode"].astype(str)
            print("[LOAD] Derived 'series_id' from partner_id||itemcode")
        else:
            raise ValueError(
                "[LOAD] Missing 'series_id' and cannot derive it: "
                "required {'partner_id','itemcode'} not both present."
            )

    # --- strict column presence (no aliases, no fuzzy) ---
    _strict_require(df, [ID_COL, TIME_COL, TARGET_COL], "LOAD")
    _strict_require(df, STATIC_COLS, "LOAD")
    _strict_require(df, EXOG_WEEKLY, "LOAD")

    # Types + alignment
    df[ID_COL] = df[ID_COL].astype(str)
    df[TIME_COL] = _to_wmon(df[TIME_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)

    # Normalize statics now: None/"None"/"" -> UNK, keep pandas StringDtype
    for c in STATIC_COLS:
        s = df[c]
        s = s.replace({None: np.nan, "None": np.nan, "none": np.nan, "": np.nan})
        df[c] = s.fillna("UNK").astype("string")

    # Exogs numeric; keep NaN if unparseable (we also add *_is_missing later)
    for ex in EXOG_WEEKLY:
        df[ex] = pd.to_numeric(df[ex], errors="coerce")

    # Calendar feats
    df = _calendarize(df)

    # Trim to canonical order
    keep = [ID_COL, TIME_COL, TARGET_COL] + STATIC_COLS + CAL_KNOWN + EXOG_WEEKLY
    df = df[keep].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Debug: show null ratios and sample of statics/exogs
    nnr = (1 - df[STATIC_COLS + EXOG_WEEKLY].isna().mean()).round(3).to_dict()
    print(f"[LOAD] Non-null ratios (statics+exogs): {nnr}")
    print("[LOAD] Sample after load:\n", df.head())

    return df


def _exog_weekly_table(hist: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in EXOG_WEEKLY if c in hist.columns]
    if cols_present:
        tbl = hist.groupby(TIME_COL, as_index=False)[cols_present].mean()
    else:
        tbl = hist[[TIME_COL]].drop_duplicates().copy()

    # NEW: ensure Monday-normalized and one row per week
    tbl[TIME_COL] = _to_wmon(tbl[TIME_COL])
    tbl = tbl.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="first")

    for ex in EXOG_WEEKLY:
        if ex not in tbl.columns:
            tbl[ex] = 0.0
    return tbl[[TIME_COL] + EXOG_WEEKLY]


# --- replace your _pad_full_contiguous with this version ---
def _pad_full_contiguous(hist: pd.DataFrame, exog_weekly: pd.DataFrame) -> pd.DataFrame:
    """Pad each series from its observed min..max at W-MON; attach exogs; calendarize."""
    parts: List[pd.DataFrame] = []
    for sid, g in hist.groupby(ID_COL, sort=False):
        if g.empty:
            continue
        tmin, tmax = g[TIME_COL].min(), g[TIME_COL].max()

        # inclusive calendar on Mondays
        rng = pd.date_range(tmin, tmax, freq="W-MON")

        gg = g.set_index(TIME_COL).reindex(rng)
        gg.index.name = TIME_COL
        gg[ID_COL] = sid

        # Broadcast statics: use first value already UNK-filled in _load_kb
        for c in STATIC_COLS:
            if c in g.columns:
                if g[c].notna().any():
                    val = g[c].iloc[0]
                else:
                    val = "UNK"
                gg[c] = val

        gg[TARGET_COL] = pd.to_numeric(gg[TARGET_COL], errors="coerce").fillna(0.0)
        parts.append(gg.reset_index())

    panel = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=[ID_COL, TIME_COL, TARGET_COL] + STATIC_COLS)
    )

    # Build robust week keys for merge
    panel[TIME_COL] = _to_wmon(panel[TIME_COL])
    panel["__wk"] = _week_key(panel[TIME_COL])
    
    ex_tbl = exog_weekly.copy()
    ex_tbl[TIME_COL] = _to_wmon(ex_tbl[TIME_COL])
    ex_tbl["__wk"] = _week_key(ex_tbl[TIME_COL])
    
    # --- DEBUG: check key overlap before join ---
    left_keys  = pd.unique(panel["__wk"])
    right_keys = pd.unique(ex_tbl["__wk"])
    inter      = np.intersect1d(left_keys, right_keys)
    print(f"[DEBUG] week-key sizes → panel={left_keys.size}, exog={right_keys.size}, intersect={inter.size}")
    if inter.size == 0:
        print("[DEBUG] panel weeks (head):", panel[TIME_COL].head().dt.strftime("%Y-%m-%d").tolist())
        print("[DEBUG] exog  weeks (head):", ex_tbl[TIME_COL].head().dt.strftime("%Y-%m-%d").tolist())
        print("[DEBUG] panel __wk (head):", panel['__wk'].head().tolist())
        print("[DEBUG] exog  __wk (head):", ex_tbl['__wk'].head().tolist())
        raise RuntimeError("No overlapping weekly keys between panel and exog table. Check normalization or date range.")
    
    # --- DEBUG: peek the first matching keys and their exog values ---
    peek_keys = inter[:5]
    print("[DEBUG] exog peek rows:", ex_tbl.loc[ex_tbl["__wk"].isin(peek_keys), ["__wk"] + EXOG_WEEKLY].head().to_dict(orient="records"))
    
    # Drop any existing exog columns coming from the reindexed left side to avoid overlap
    to_drop = [c for c in EXOG_WEEKLY if c in panel.columns]
    if to_drop:
        panel = panel.drop(columns=to_drop)
    
    # Join exogs using the integer week key as index (more robust than merge in rare edge cases)
    ex_right = ex_tbl.set_index("__wk")[EXOG_WEEKLY]
    panel = panel.join(ex_right, on="__wk")
    panel = panel.drop(columns="__wk")


    # Types + explicit missing flags
    for ex in EXOG_WEEKLY:
        if ex not in panel.columns:
            panel[ex] = np.nan
        panel[ex] = pd.to_numeric(panel[ex], errors="coerce")
        panel[f"{ex}_is_missing"] = panel[ex].isna().astype("int8")

    # Recompute calendar fields from normalized week
    panel = _calendarize(panel)

    # Final ordering (exclude *_is_missing from the main table)
    order = [ID_COL, TIME_COL, TARGET_COL] + STATIC_COLS + CAL_KNOWN + EXOG_WEEKLY
    panel = panel[order].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Ensure statics are UNK (not None) before save
    for c in STATIC_COLS:
        panel[c] = panel[c].astype("string").fillna("UNK")

    # Debug
    nnr_panel = (1 - panel[STATIC_COLS + EXOG_WEEKLY].isna().mean()).round(3).to_dict()
    print(f"[PAD] Non-null ratios (statics+exogs): {nnr_panel}")
    print("[PAD] Sample after pad+merge:\n", panel.head())

    return panel

# -------- run --------
if __name__ == "__main__":
    print(f"[PREPATCH] Loading KB from {DATASET_PATH} ...", flush=True)
    kb0 = _load_kb(DATASET_PATH)
    print(f"[PREPATCH] KB0 rows={len(kb0):,} | series={kb0[ID_COL].nunique():,}", flush=True)
    
    print(f"[PREPATCH] Loading KB from {DATASET_PATH1} ...", flush=True)
    kb1 = _load_kb(DATASET_PATH1)
    print(f"[PREPATCH] KB1 rows={len(kb1):,} | series={kb1[ID_COL].nunique():,}", flush=True)

    print(f"[PREPATCH] Loading KB from {DATASET_PATH2} ...", flush=True)
    kb2 = _load_kb(DATASET_PATH2)
    print(f"[PREPATCH] KB2 rows={len(kb2):,} | series={kb2[ID_COL].nunique():,}", flush=True)
    
    # concatenate
    kb = pd.concat([kb0, kb1, kb2], ignore_index=True).drop_duplicates([ID_COL, TIME_COL])
    print(f"[PREPATCH] Combined KB rows={len(kb):,} | series={kb[ID_COL].nunique():,}", flush=True)

    print("[PREPATCH] Building weekly exog table ...", flush=True)
    ex_tbl = _exog_weekly_table(kb)
    print(f"[PREPATCH] Exog table weeks={len(ex_tbl):,}", flush=True)

    print("[PREPATCH] Padding full history per series ...", flush=True)
    padded = _pad_full_contiguous(kb, ex_tbl)
    print(
        f"[PREPATCH] Padded rows={len(padded):,} | series={padded[ID_COL].nunique():,}",
        flush=True,
    )

    # Prefer Parquet (size/speed); optionally also pickle for quick local loads
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

    # Safety: make sure statics have UNK (no None) right before writing
    for c in STATIC_COLS:
        padded[c] = padded[c].astype("string").fillna("UNK")

    padded.to_parquet(OUT_PARQUET, index=False)
    print(f"[PREPATCH] Saved → {OUT_PARQUET}", flush=True)

    if OUT_PICKLE:
        padded.to_pickle(OUT_PICKLE)
        print(f"[PREPATCH] Saved → {OUT_PICKLE}", flush=True)

    print("[PREPATCH] Done.", flush=True)
