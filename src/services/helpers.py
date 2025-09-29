from __future__ import annotations
import time, warnings, math
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src.core.config import DATASET_PATH, KNOWLEDGE_BASE_MAX_DATE, CONTEXT_WEEKS

# ------------------- CONSTANTS -------------------
ID_COL, TIME_COL, TARGET_COL = "series_id", "week", "sales"
STATIC_COLS   = ["group_id", "item_group1", "item_group2"]
CAL_KNOWN     = ["month", "weekofyear", "year"]
EXOG_WEEKLY   = ["temperature", "num_sunny_days", "holiday_count"]  # optional in provided history
KNOWN_FUT_CAL = CAL_KNOWN

# Quiet warnings in service
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------- small utils -------------------
def _read_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    return df

USE_TQDM = False
def tqdmit(it, total=None, desc=None, unit=None, leave=False):
    return it  # no progress bars

class phase:
    def __init__(self, name: str): self.name, self.t0 = name, None
    def __enter__(self):
        self.t0 = time.time()
        print(f"[PHASE] {self.name} …", flush=True)
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - (self.t0 or time.time())
        print(f"[PHASE] {self.name} ✓ {dt:,.2f}s", flush=True)

def _align_monday(ts) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("W-MON").start_time

def _clean_num(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

def _clean_overall(d: dict) -> dict:
    return {k: _clean_num(v) for k, v in d.items()}

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

def ensure_sales(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    fam = [c for c in df.columns if c == target_col or c.startswith(target_col)]
    if not fam:
        raise ValueError("No 'sales' or 'sales*' column found.")
    base = target_col if target_col in fam else sorted(fam)[0]
    if base != target_col:
        df[target_col] = df[base]
    for c in fam:
        if c != target_col:
            df[target_col] = df[target_col].where(df[target_col].notna(), df[c])
            df.drop(columns=c, inplace=True, errors="ignore")
    return df

def to_wmon(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if ts.isna().any():
        bad = s[pd.isna(ts)].head(5).tolist()
        raise ValueError(f"'week' has unparseable values; examples: {bad}")
    return ts.dt.to_period("W-MON").dt.to_timestamp("W-MON")

def calendarize(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = to_wmon(out[time_col])
    t = out[time_col]
    out["month"]      = t.dt.month.astype("int16")
    out["weekofyear"] = t.dt.isocalendar().week.astype("int16")
    out["year"]       = t.dt.year.astype("int16")
    return out

def exog_weekly_table(hist: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in EXOG_WEEKLY if c in hist.columns]
    if cols_present:
        tbl = hist.groupby(TIME_COL, as_index=False)[cols_present].mean()
    else:
        tbl = hist[[TIME_COL]].drop_duplicates().copy()
    for ex in EXOG_WEEKLY:
        if ex not in tbl.columns:
            tbl[ex] = 0.0
    return tbl[[TIME_COL] + EXOG_WEEKLY]

# ------------- KEY FIX: bounded padding (window_lb..window_ub) -------------
def pad_until_base(hist: pd.DataFrame, bases: pd.DataFrame, exog_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Pad each series' ACTUAL history weekly-contiguously up to its own base (inclusive).
    This is equivalent to full-history padding followed by '<= base' cut, but much faster.
    """
    # Map series_id -> base timestamp for quick lookup
    base_map = dict(bases[[ID_COL, "base"]].itertuples(index=False, name=None))

    merged = hist.merge(bases[[ID_COL]].drop_duplicates(), on=ID_COL, how="inner")
    parts: List[pd.DataFrame] = []
    n_series = merged[ID_COL].nunique()

    for sid, g in tqdmit(merged.groupby(ID_COL, sort=False),
                         total=n_series, desc="Pad contiguous history (≤ base)", unit="series", leave=False):
        if g.empty:
            continue
        b = base_map.get(sid, None)
        if b is None:
            continue

        # Only pad up to base (equivalent to pad full then cut ≤ base)
        tmin = g[TIME_COL].min()
        tmax = min(g[TIME_COL].max(), b)
        if tmin > tmax:
            continue

        rng = pd.date_range(tmin, tmax, freq="W-MON")
        gg = g.set_index(TIME_COL).reindex(rng)
        gg.index.name = TIME_COL
        gg[ID_COL] = sid

        # statics: ffill/bfill; default to UNK if fully missing
        for c in STATIC_COLS:
            if c in g.columns:
                gg[c] = g[c].ffill().bfill().iloc[0] if gg[c].isna().all() else g[c].ffill().bfill()

        gg[TARGET_COL] = pd.to_numeric(gg[TARGET_COL], errors="coerce").fillna(0.0)
        parts.append(gg.reset_index())

    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[ID_COL, TIME_COL, TARGET_COL] + STATIC_COLS
    )

    # Attach exogs by week; default to 0.0
    panel = panel.merge(exog_weekly, on=TIME_COL, how="left")
    for ex in EXOG_WEEKLY:
        if ex not in panel.columns:
            panel[ex] = 0.0
        else:
            panel[ex] = pd.to_numeric(panel[ex], errors="coerce").fillna(0.0)

    panel = calendarize(panel, TIME_COL)
    order = [ID_COL, TIME_COL, TARGET_COL] \
            + [c for c in STATIC_COLS if c in panel.columns] \
            + CAL_KNOWN + EXOG_WEEKLY

    return panel[order].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

def synthetic_zero_history(bases: pd.DataFrame, min_weeks: int) -> pd.DataFrame:
    rows = []
    total = len(bases)
    for _, r in tqdmit(bases.iterrows(), total=total, desc="Synthesize zero histories", unit="series", leave=False):
        sid, b = r[ID_COL], r["base"]
        start = b - pd.to_timedelta(max(min_weeks, 1) - 1, unit="W")
        weeks = pd.date_range(start, b, freq="W-MON")
        for wk in weeks:
            rows.append((sid, wk, 0.0))
    df = pd.DataFrame(rows, columns=[ID_COL, TIME_COL, TARGET_COL])
    df = calendarize(df, TIME_COL)
    for c in STATIC_COLS:
        df[c] = "UNK"
    return df

# helpers.py
def choose_per_series_bases(req: pd.DataFrame, Hmodel: int) -> pd.DataFrame:
    # ensure weekly-Monday timestamps no matter what the caller passed
    req[TIME_COL] = to_wmon(req[TIME_COL])
    df = req.copy()
    df[TIME_COL] = to_wmon(df[TIME_COL])

    g = df.groupby(ID_COL)[TIME_COL]
    info = pd.DataFrame({
        ID_COL: g.count().index,
        "rmin": g.min().values,
        "rmax": g.max().values,
    })
    info["span_h"] = ((info["rmax"] - info["rmin"]).dt.days // 7) + 1

    too_wide = info[info["span_h"] > Hmodel]
    if not too_wide.empty:
        raise ValueError(
            "Some series request a span wider than model prediction_length.\n" +
            "\n".join(f" - {row[ID_COL]}: span={row['span_h']} > H={Hmodel}" for _, row in too_wide.iterrows())
        )

    info["base"] = info["rmax"] - pd.to_timedelta(Hmodel, unit="W")
    return info[[ID_COL, "base", "rmin", "rmax", "span_h"]]

def preds_to_long(pred, id_col: str = ID_COL, time_col: str = TIME_COL) -> pd.DataFrame:
    try:
        df = pred.to_pandas() if hasattr(pred, "to_pandas") else (pred.to_pd() if hasattr(pred, "to_pd") else pred)
    except Exception:
        df = pred
    if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex):
        val = "mean" if "mean" in df.columns else ("0.5" if "0.5" in df.columns else df.columns[0])
        fr = df[[val]].rename(columns={val: "y_pred"}).reset_index()
        a, b = fr.columns[:2].tolist()
        if np.issubdtype(fr[a].dtype, np.datetime64):
            fr.rename(columns={a: time_col, b: id_col}, inplace=True)
        else:
            fr.rename(columns={a: id_col, b: time_col}, inplace=True)
        fr[time_col] = pd.to_datetime(fr[time_col]).dt.to_period("W-MON").dt.to_timestamp("W-MON")
        return fr[[id_col, time_col, "y_pred"]].sort_values([id_col, time_col]).reset_index(drop=True)
    long = df.stack().rename("y_pred").to_frame().reset_index()
    a, b = long.columns[:2].tolist()
    if np.issubdtype(long[a].dtype, np.datetime64):
        long.rename(columns={long.columns[0]: time_col, long.columns[1]: id_col}, inplace=True)
    else:
        long.rename(columns={long.columns[0]: id_col, long.columns[1]: time_col}, inplace=True)
    long[time_col] = pd.to_datetime(long[time_col]).dt.to_period("W-MON").dt.to_timestamp("W-MON")
    return long[[id_col, time_col, "y_pred"]].sort_values([id_col, time_col]).reset_index(drop=True)

def extend_to_base_with_predictions(
    hist_padded: pd.DataFrame,
    bases: pd.DataFrame,
    predictor: TimeSeriesPredictor,
    static_feats_ag: Optional[pd.DataFrame],
    *,
    H_model: int,
) -> pd.DataFrame:
    hist_cols = [TARGET_COL] + CAL_KNOWN + EXOG_WEEKLY

    lasts = hist_padded.groupby(ID_COL, observed=True)[TIME_COL].max().rename("last").reset_index()
    need = lasts.merge(bases[[ID_COL, "base"]], on=ID_COL, how="left")
    need = need[need["last"] < need["base"]]
    if need.empty:
        print("[ALIGN] No series need roll-forward to base.")
        return hist_padded

    round_ix = 0
    while True:
        round_ix += 1
        lasts = hist_padded.groupby(ID_COL, observed=True)[TIME_COL].max().rename("last").reset_index()
        need = lasts.merge(bases[[ID_COL, "base"]], on=ID_COL, how="left")
        need = need[need["last"] < need["base"]]
        if need.empty:
            break

        print(f"[RF] round {round_ix}: short series={need.shape[0]:,}", flush=True)
        ids = need[ID_COL].tolist()

        ctx_df = hist_padded[hist_padded[ID_COL].isin(ids)][[ID_COL, TIME_COL] + hist_cols].copy()
        sidx = None
        if (static_feats_ag is not None) and (not static_feats_ag.empty):
            ssub = static_feats_ag[static_feats_ag[ID_COL].isin(ids)].copy(deep=True)
            sidx = ssub if not ssub.empty else None

        ts_ctx = TimeSeriesDataFrame.from_data_frame(
            ctx_df, id_column=ID_COL, timestamp_column=TIME_COL, static_features_df=sidx
        )

        fut_grid = predictor.make_future_data_frame(data=ts_ctx)
        fut_grid = fut_grid.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})[[ID_COL, TIME_COL]]
        fut_grid[TIME_COL] = to_wmon(fut_grid[TIME_COL])
        fut_grid = calendarize(fut_grid, TIME_COL)
        kf_ts = TimeSeriesDataFrame.from_data_frame(
            fut_grid[[ID_COL, TIME_COL] + KNOWN_FUT_CAL],
            id_column=ID_COL, timestamp_column=TIME_COL
        )

        fcst = predictor.predict(data=ts_ctx, known_covariates=kf_ts)
        preds = preds_to_long(fcst).sort_values([ID_COL, TIME_COL])

        need_small = need[[ID_COL, "last", "base"]].set_index(ID_COL)
        new_rows = []

        have_statics = [c for c in STATIC_COLS if c in hist_padded.columns]
        last_stat = (hist_padded[[ID_COL] + have_statics]
                     .drop_duplicates([ID_COL], keep="last")
                     .set_index(ID_COL)) if have_statics else None

        for sid, gpred in tqdmit(preds.groupby(ID_COL, sort=False), total=need.shape[0],
                                 desc=f"RF round {round_ix}", unit="series", leave=False):
            if sid not in need_small.index:
                continue
            last = need_small.at[sid, "last"]
            base = need_small.at[sid, "base"]
            gap_weeks = int(((base - last).days) // 7)
            take = min(H_model, max(gap_weeks, 0))
            if take <= 0:
                continue

            step_rows = gpred.head(take)
            new = pd.DataFrame({
                ID_COL: sid,
                TIME_COL: step_rows[TIME_COL].values,
                TARGET_COL: step_rows["y_pred"].astype(float).values,
            })

            for c in STATIC_COLS:
                if (last_stat is not None) and (c in last_stat.columns) and (sid in last_stat.index):
                    new[c] = last_stat.at[sid, c]
                else:
                    new[c] = "UNK"

            new = calendarize(new, TIME_COL)
            for ex in EXOG_WEEKLY:
                if ex not in new.columns:
                    new[ex] = 0.0
                else:
                    new[ex] = pd.to_numeric(new[ex], errors="coerce").fillna(0.0)

            new_rows.append(new)

        if not new_rows:
            break

        add_all = pd.concat(new_rows, ignore_index=True)
        hist_padded = (pd.concat([hist_padded, add_all], ignore_index=True)
                         .sort_values([ID_COL, TIME_COL])
                         .reset_index(drop=True))
        print(f"[ALIGN] Batched roll-forward round {round_ix}: added {len(add_all):,} rows for {need.shape[0]:,} series.", flush=True)

    print("[ALIGN] Extended all short series to base using batched model predictions.")
    return hist_padded

def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    err  = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# ------------------- KB helpers -------------------
def get_context_from_kb(*, customer_id: str | None = None, group_id: str | None = None) -> pd.DataFrame:
    """Return KB context (features + target) filtered by customer/group if available."""
    df = _RAW_DF
    if customer_id and "customer_id" in df.columns:
        df = df[df["customer_id"].astype(str) == str(customer_id)]
    if group_id and "group_id" in df.columns:
        df = df[df["group_id"].astype(str) == str(group_id)]
    return df.copy()

def attach_truth_from_kb(preds: pd.DataFrame) -> pd.DataFrame:
    """Attach y_true from KB only (safe for evaluation)."""
    return preds.merge(_TRUTH_DF, on=[ID_COL, TIME_COL], how="left")

def _overlay_optional_history(kb_df: pd.DataFrame, hist_opt: pd.DataFrame | None) -> pd.DataFrame:
    """Merge optional user history onto KB: user rows override on (series_id, week)."""
    if hist_opt is None or hist_opt.empty:
        return kb_df
    h = dedup(hist_opt.copy())
    if ID_COL not in h.columns and {"customer_id","item_id"}.issubset(h.columns):
        h[ID_COL] = h["customer_id"].astype(str) + "||" + h["item_id"].astype(str)
    need = {ID_COL, TIME_COL, TARGET_COL}
    missing = need - set(h.columns)
    if missing:
        raise ValueError(f"history missing {missing}")
    h[ID_COL] = h[ID_COL].astype(str)
    h[TIME_COL] = to_wmon(h[TIME_COL])
    h[TARGET_COL] = pd.to_numeric(h[TARGET_COL], errors="coerce").fillna(0.0)
    for c in STATIC_COLS:
        if c in h.columns: h[c] = h[c].astype(str).fillna("UNK")
    for ex in EXOG_WEEKLY:
        if ex in h.columns: h[ex] = pd.to_numeric(h[ex], errors="coerce")
    h = calendarize(h, TIME_COL)

    # union columns, prefer user rows (keep="last")
    cols = list(kb_df.columns.union(h.columns))
    kb = kb_df.reindex(columns=cols)
    h2 = h.reindex(columns=cols)
    merged = pd.concat([kb, h2], ignore_index=True)
    merged = (merged
              .sort_values([ID_COL, TIME_COL])
              .drop_duplicates([ID_COL, TIME_COL], keep="last")
              .reset_index(drop=True))
    return merged

def _compute_bases_sliding(series_ids, hist_all, weeks_req, H):
    """
    Sliding-window base per series:
      - If rmax within H weeks after kb_last(series), base = kb_last(series)
      - Else base = kb_last + H * floor((diff-1)/H)
      - If series has no kb_last (not in KB): base = rmax - H
    """
    rmax = pd.to_datetime(weeks_req.max())
    sid_df = pd.DataFrame({ID_COL: pd.Index(series_ids, dtype=str)})

    # last known week per series from KB
    kb_last = (hist_all.groupby(ID_COL, observed=True)[TIME_COL]
                        .max()
                        .rename("kb_last")
                        .reset_index())

    bases = sid_df.merge(kb_last, on=ID_COL, how="left")
    # default base for unknown series
    default_base = rmax - pd.to_timedelta(H, unit="W")

    # diff in weeks from kb_last to rmax (NaT -> large)
    diff_weeks = ((rmax - bases["kb_last"]).dt.days // 7).astype("Int64")

    # start with default everywhere
    bases["base"] = default_base

    # where kb_last exists:
    have_hist = bases["kb_last"].notna()

    # within horizon: base = kb_last
    within = have_hist & (diff_weeks <= H)
    bases.loc[within, "base"] = bases.loc[within, "kb_last"]

    # beyond horizon: slide in H-chunks: base = kb_last + H * floor((diff-1)/H)
    beyond = have_hist & (diff_weeks > H)
    k = ((diff_weeks[beyond] - 1) // H).astype(int)
    bases.loc[beyond, "base"] = bases.loc[beyond, "kb_last"] + pd.to_timedelta(k * H, unit="W")

    return bases[[ID_COL, "base"]]

# ------------------- Load KB once -------------------
_RAW_DF = _read_dataset(DATASET_PATH)
_RAW_DF = dedup(_RAW_DF)

# Normalize columns
if any((c == TARGET_COL or c.startswith(TARGET_COL)) for c in _RAW_DF.columns):
    _RAW_DF = ensure_sales(_RAW_DF, TARGET_COL)

if ID_COL not in _RAW_DF.columns and {"partner_id","itemcode"}.issubset(_RAW_DF.columns):
    _RAW_DF[ID_COL] = _RAW_DF["partner_id"].astype(str) + "||" + _RAW_DF["itemcode"].astype(str)

_RAW_DF[ID_COL] = _RAW_DF[ID_COL].astype(str)
_RAW_DF[TIME_COL] = to_wmon(_RAW_DF[TIME_COL])
_RAW_DF[TARGET_COL] = pd.to_numeric(_RAW_DF[TARGET_COL], errors="coerce").fillna(0.0)

for c in STATIC_COLS:
    if c in _RAW_DF.columns:
        _RAW_DF[c] = _RAW_DF[c].astype(str).fillna("UNK")
for ex in EXOG_WEEKLY:
    if ex in _RAW_DF.columns:
        _RAW_DF[ex] = pd.to_numeric(_RAW_DF[ex], errors="coerce")

_RAW_DF = calendarize(_RAW_DF, TIME_COL)

_TRUTH_DF = (_RAW_DF[[ID_COL, TIME_COL, TARGET_COL]]
             .rename(columns={TARGET_COL: "y_true"})
             .drop_duplicates([ID_COL, TIME_COL]))


