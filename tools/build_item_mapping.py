# tools/build_item_mapping.py
# Build a de-duplicated mapping: item_group1, item_group2, itemcode
# Sources:
#   - data/xgb_prepared_data_20250626_20250831.pkl  (NEWER / higher priority)
#   - data/xgb_prepared_data_v2.pkl                  (OLDER / lower  priority)
#
# Output:
#   - data/item_mapping.parquet

import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- config ----------------
IN_NEW  = os.getenv("IN_NEW",  "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/xgb_prepared_data_20250626_20250831.pkl")
IN_OLD  = os.getenv("IN_OLD",  "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/xgb_prepared_data_v2.pkl")
OUT_PAR = os.getenv("OUT_PAR", "/home/jovyan/work/deployments/sale_forcasting_externaleffects_service/data/item_mapping.parquet")

ITEMCODE_COL  = "itemcode"
GROUP1_COL    = "item_group1"
GROUP2_COL    = "item_group2"
SERIES_COL    = "series_id"      # fallback to parse "customer||itemcode" if itemcode missing

# ---------------- helpers ----------------
def _load_any(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)

def _to_str_preserve(x) -> str:
    """Best-effort stringify without trailing '.0'. Leading zeros are only preserved if the input was string-like."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # remove trailing .0 for floats like 40223.0
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _normalize_cols(df: pd.DataFrame, source_priority: int) -> pd.DataFrame:
    """Select & normalize columns; derive itemcode from series_id if needed. Attach priority for conflict resolution."""
    cols_present = set(df.columns.str.lower() if isinstance(df.columns, pd.Index) else df.columns)
    # case-insensitive access
    def _ci(name):  # case-insensitive getter
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return None

    ic = _ci(ITEMCODE_COL)
    g1 = _ci(GROUP1_COL)
    g2 = _ci(GROUP2_COL)
    sid = _ci(SERIES_COL)

    out = pd.DataFrame()

    # itemcode: from column or parsed from series_id "customer||item"
    if ic is not None:
        out[ITEMCODE_COL] = df[ic].map(_to_str_preserve)
    elif sid is not None:
        # parse "cust||item"
        out[ITEMCODE_COL] = df[sid].astype(str).str.split("||").str[-1].map(_to_str_preserve)
    else:
        # no way to get itemcode
        out[ITEMCODE_COL] = np.nan

    # groups (may be missing)
    out[GROUP1_COL] = df[g1].astype(str).str.strip() if g1 is not None else "UNK"
    out[GROUP2_COL] = df[g2].astype(str).str.strip() if g2 is not None else "UNK"

    # normalize UNK / NA
    for c in (GROUP1_COL, GROUP2_COL):
        out[c] = out[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        out[c] = out[c].fillna("UNK")

    out[ITEMCODE_COL] = out[ITEMCODE_COL].fillna("").astype(str).str.strip()
    out = out[out[ITEMCODE_COL] != ""]  # require itemcode
    out["priority"] = int(source_priority)
    return out[[GROUP1_COL, GROUP2_COL, ITEMCODE_COL, "priority"]]

def _coalesce(prefer_new_then_better: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve duplicates per itemcode:
      1) Prefer higher priority (newer file)
      2) Prefer rows with more known info (less 'UNK')
      3) If still tied, keep the first occurrence
    """
    def _pick_best(g: pd.DataFrame) -> pd.Series:
        # score: higher is better
        known_score = (g[GROUP1_COL].ne("UNK").astype(int)
                     + g[GROUP2_COL].ne("UNK").astype(int))
        # sort by: priority desc, known_score desc
        idx = (g.assign(_score=known_score)
                 .sort_values(["priority", "_score"], ascending=[False, False])
                 .index[0])
        return g.loc[idx, [GROUP1_COL, GROUP2_COL, ITEMCODE_COL]]

    return (prefer_new_then_better
            .groupby(ITEMCODE_COL, as_index=False, group_keys=False)
            .apply(_pick_best)
            .reset_index(drop=True))

# ---------------- run ----------------
if __name__ == "__main__":
    print(f"[MAP] Loading NEW: {IN_NEW}")
    df_new = _load_any(IN_NEW)
    print(f"[MAP] NEW rows={len(df_new):,}")

    print(f"[MAP] Loading OLD: {IN_OLD}")
    df_old = _load_any(IN_OLD)
    print(f"[MAP] OLD rows={len(df_old):,}")

    # normalize & tag priority (NEW > OLD)
    m_new = _normalize_cols(df_new, source_priority=2)
    m_old = _normalize_cols(df_old, source_priority=1)

    merged = pd.concat([m_new, m_old], ignore_index=True)
    print(f"[MAP] Combined rows (pre-dedup)={len(merged):,}")

    mapping = _coalesce(merged)
    mapping = mapping[[GROUP1_COL, GROUP2_COL, ITEMCODE_COL]].sort_values([GROUP1_COL, GROUP2_COL, ITEMCODE_COL]).reset_index(drop=True)

    # basic stats
    n_items = mapping[ITEMCODE_COL].nunique()
    n_g1 = mapping[GROUP1_COL].nunique()
    n_g2 = mapping[GROUP2_COL].nunique()
    print(f"[MAP] Unique items={n_items:,} | groups1={n_g1:,} | groups2={n_g2:,}")

    # save parquet
    Path(OUT_PAR).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(OUT_PAR, index=False)
    print(f"[MAP] Saved mapping → {OUT_PAR}")
