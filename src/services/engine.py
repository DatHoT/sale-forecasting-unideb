from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pandas as pd
from typing import Optional
from .helpers import (
    ID_COL, TIME_COL, TARGET_COL, STATIC_COLS, CAL_KNOWN, EXOG_WEEKLY, KNOWN_FUT_CAL,
    phase, dedup, ensure_sales, to_wmon, calendarize, exog_weekly_table, pad_until_base,
    synthetic_zero_history, choose_per_series_bases, extend_to_base_with_predictions,
    preds_to_long, metrics_np, _clean_overall, _overlay_optional_history, attach_truth_from_kb, 
    get_context_from_kb,_align_monday, _compute_bases_sliding
)
from src.core.config import DATASET_PATH, KNOWLEDGE_BASE_MAX_DATE, DEFAULT_PREDICTOR_PATH, CONTEXT_WEEKS, PREPATCHED_PAD_PATH, MIN_FAKE_HISTORY_WEEKS, AUX_TRUTH_PATH
_PREPATCHED = None  # global cache
_AUX_TRUTH_CACHE = None

def _load_aux_truth_once() -> pd.DataFrame:
    """
    Load auxiliary ground-truth (REQ) once and normalize to:
        [series_id, week(W-MON), y_true: float]
    Handles duplicate columns, multiple sales* columns, and builds series_id if needed.
    Returns empty DataFrame if path missing or on parse error (with a warning).
    """
    import pandas as pd
    from pathlib import Path

    global _AUX_TRUTH_CACHE
    if _AUX_TRUTH_CACHE is not None:
        return _AUX_TRUTH_CACHE

    if not AUX_TRUTH_PATH or not Path(AUX_TRUTH_PATH).exists():
        print("[AUX] AUX_TRUTH_PATH not set or file missing; skipping.", flush=True)
        _AUX_TRUTH_CACHE = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])
        return _AUX_TRUTH_CACHE

    path = str(AUX_TRUTH_PATH)
    try:
        if path.endswith(".pkl"):
            df = pd.read_pickle(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f"[AUX] Failed to read {path}: {e}", flush=True)
        _AUX_TRUTH_CACHE = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])
        return _AUX_TRUTH_CACHE

    # 1) De-duplicate columns (first occurrence wins)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    # 2) Normalize ID column: prefer existing series_id; else build from partner_id+itemcode
    if ID_COL not in df.columns:
        if {"partner_id", "itemcode"}.issubset(df.columns):
            df[ID_COL] = df["partner_id"].astype(str) + "||" + df["itemcode"].astype(str)
        else:
            print(f"[AUX] Missing {ID_COL} and (partner_id,itemcode). Columns: {df.columns.tolist()[:20]} ...", flush=True)
            _AUX_TRUTH_CACHE = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])
            return _AUX_TRUTH_CACHE

    # 3) Normalize week column name
    if TIME_COL not in df.columns:
        # common alternates we might see
        for alt in ["timestamp", "date", "week_start", "ts", "time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: TIME_COL})
                break
    if TIME_COL not in df.columns:
        print("[AUX] No usable week/timestamp column found.", flush=True)
        _AUX_TRUTH_CACHE = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])
        return _AUX_TRUTH_CACHE

    # 4) Coalesce sales* family into a single 'sales'
    sales_family = [c for c in df.columns if str(c).lower() == "sales" or str(c).lower().startswith("sales")]
    if not sales_family:
        print("[AUX] No 'sales' or 'sales*' columns found.", flush=True)
        _AUX_TRUTH_CACHE = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])
        return _AUX_TRUTH_CACHE

    # build a single 1-D vector by taking first non-null across the family, left-to-right
    sales_series = None
    for c in sales_family:
        col = pd.to_numeric(df[c], errors="coerce")
        sales_series = col if sales_series is None else sales_series.where(sales_series.notna(), col)
    df["__sales_coalesced"] = sales_series

    # 5) Keep only needed columns and normalize types
    out = df[[ID_COL, TIME_COL, "__sales_coalesced"]].copy()
    out[ID_COL] = out[ID_COL].astype(str)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce").dt.to_period("W-MON").dt.to_timestamp("W-MON")
    out["y_true"] = pd.to_numeric(out["__sales_coalesced"], errors="coerce")
    out = out.drop(columns=["__sales_coalesced"]).dropna(subset=[TIME_COL])

    # 6) Deduplicate keys; last wins
    out = out.drop_duplicates([ID_COL, TIME_COL], keep="last").reset_index(drop=True)

    if not out.empty:
        print(f"[AUX] loaded aux truth rows={len(out):,} | span=({out[TIME_COL].min().date()}..{out[TIME_COL].max().date()})",
              flush=True)
    else:
        print("[AUX] aux truth loaded but empty after normalization.", flush=True)

    _AUX_TRUTH_CACHE = out
    return _AUX_TRUTH_CACHE
    
def _load_prepatched_once() -> pd.DataFrame:
    global _PREPATCHED
    if _PREPATCHED is None:
        print(f"[BOOT] Loading prepatched history from {PREPATCHED_PAD_PATH} ...", flush=True)
        _PREPATCHED = pd.read_parquet(PREPATCHED_PAD_PATH) if PREPATCHED_PAD_PATH.endswith(".parquet") \
                     else pd.read_pickle(PREPATCHED_PAD_PATH)
        # optional: ensure types are right
        _PREPATCHED[TIME_COL] = pd.to_datetime(_PREPATCHED[TIME_COL])
        _PREPATCHED[ID_COL] = _PREPATCHED[ID_COL].astype(str)
        _PREPATCHED[TARGET_COL] = pd.to_numeric(_PREPATCHED[TARGET_COL], errors="coerce").fillna(0.0)
        # columns should already include statics, calendar, exogs
        print(f"[BOOT] Prepatched rows={len(_PREPATCHED):,} | series={_PREPATCHED[ID_COL].nunique():,}", flush=True)
    return _PREPATCHED

def run_report(
    *,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    predictor_path: Optional[str] = None,
    customer_id: Optional[str] = None,
    group_id: Optional[str] = None,
    history_df: Optional[pd.DataFrame] = None,  # optional manual history to extend KB
):
    """
    Forecast over [interval_start..interval_end] with CONTEXT_WEEKS of encoder context before start.
    Uses KB by default, overlays optional user history. Returns (summary_dict, preds_df). No files written.
    """

    def _resolve_predictor_path(pth: Optional[str]) -> str:
        path = pth or DEFAULT_PREDICTOR_PATH
        if not path:
            raise ValueError("No predictor_path provided and DEFAULT_PREDICTOR_PATH is empty.")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Predictor folder not found at '{path}'. "
                "It must be the full AutoGluon predictor directory (predictor.pkl, learner.pkl, models/, ...)."
            )
        return path

    # ---------- 0) Request → weekly grid, sanity ----------
    start_wmon = _align_monday(interval_start)  # Monday-aligned
    end_wmon   = _align_monday(interval_end)

    # Normalize ordering (handle reversed dates gracefully)
    if end_wmon < start_wmon:
        print(f"[WARN] interval_end {end_wmon.date()} precedes interval_start {start_wmon.date()}; swapping.", flush=True)
        start_wmon, end_wmon = end_wmon, start_wmon

    # Build aligned weekly index (inclusive)
    weeks_req = pd.date_range(start_wmon, end_wmon, freq="W-MON")
    if len(weeks_req) == 0:
        # Defensive fallback: at least the start week
        weeks_req = pd.DatetimeIndex([start_wmon])

    if len(weeks_req) > 12:
        raise ValueError(f"Interval length {len(weeks_req)} exceeds max 12 weeks.")

    print(
        f"[INFO] weeks_req={len(weeks_req)} | start_aligned={start_wmon.date()} | "
        f"end_aligned={end_wmon.date()} | CONTEXT_WEEKS={CONTEXT_WEEKS}",
        flush=True,
    )
    # ---------- 1) Load model ----------
    with phase("Load model"):
        predictor = TimeSeriesPredictor.load(_resolve_predictor_path(predictor_path))
        H = int(getattr(predictor, "prediction_length", 12))

    # ---------- 2) Build context from KB (+ optional history overlay) ----------
    with phase("Assemble context from KB (+ optional history)"):
        kb_ctx = get_context_from_kb(customer_id=customer_id, group_id=group_id)
    
        # --- STRICT CUSTOMER FILTER (if provided) ---
        if customer_id:
            cid = str(customer_id)
            if "customer_id" in kb_ctx.columns:
                kb_ctx = kb_ctx[kb_ctx["customer_id"].astype(str) == cid]
            else:
                # fall back to series_id prefix: "<customer_id>||<item_id>"
                kb_ctx = kb_ctx[kb_ctx[ID_COL].astype(str).str.startswith(cid + "||", na=False)]
    
        keep = [ID_COL, TIME_COL, TARGET_COL] \
               + [c for c in STATIC_COLS if c in kb_ctx.columns] \
               + CAL_KNOWN + [c for c in EXOG_WEEKLY if c in kb_ctx.columns]
        kb_ctx = kb_ctx[keep].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    
        # Overlay any provided history for these series only
        hist_all = _overlay_optional_history(kb_ctx, history_df)
    
        # Guardrail: if customer is specified, ensure no foreign series remain
        if customer_id:
            assert hist_all[ID_COL].astype(str).str.startswith(str(customer_id) + "||", na=False).all(), \
                "Foreign series leaked into hist_all despite customer filter."

    # ---------- 3) HARD TRIM to request window + encoder context ----------
    lower_bound = start_wmon - pd.to_timedelta(max(CONTEXT_WEEKS, 1), unit="W")
    upper_bound = end_wmon
    hist_narrow = hist_all[(hist_all[TIME_COL] >= lower_bound) & (hist_all[TIME_COL] <= upper_bound)].copy()

    # ---- Choose the smallest necessary series set ----
    if history_df is not None and not history_df.empty:
        # Client provided explicit history → forecast only these series
        h = history_df.copy()
        if ID_COL not in h.columns and {"customer_id","item_id"}.issubset(h.columns):
            h[ID_COL] = h["customer_id"].astype(str) + "||" + h["item_id"].astype(str)
        series_ids = sorted(h[ID_COL].astype(str).dropna().unique().tolist())
    elif not hist_narrow.empty:
        # Use series that actually appear in the 12w context window
        series_ids = sorted(hist_narrow[ID_COL].dropna().unique().tolist())
    else:
        # Last resort: any series present in KB scope (but prune to those with data near the window)
        # Keep series having at least one observation on or before upper_bound and not too far in the past
        cutoff = lower_bound - pd.to_timedelta(4, unit="W")  # 4-week grace before context window
        has_recent = hist_all[(hist_all[TIME_COL] <= upper_bound) & (hist_all[TIME_COL] >= cutoff)]
        if not has_recent.empty:
            series_ids = sorted(has_recent[ID_COL].dropna().unique().tolist())
        else:
            # fallback to all in scope (should be rare)
            series_ids = sorted(hist_all[ID_COL].dropna().unique().tolist())
            if not series_ids:
                raise ValueError("No series available in KB for the requested customer/interval.")

    if not series_ids:
        raise ValueError("No series available in KB/history for the requested scope (after filters).")

    # Build the requested (series, week) grid
    req = pd.DataFrame([(sid, wk) for sid in series_ids for wk in weeks_req], columns=[ID_COL, TIME_COL])


    # ---------- 4) Clamp per-series span ≤ H; compute bases ----------
    # bases = choose_per_series_bases(req, H)

    # # OPTIONAL: defensive re-trim around bases
    # base_min, base_max = bases["base"].min(), bases["base"].max()
    # lb2 = min(lower_bound, base_min)
    # ub2 = max(upper_bound, base_max)
    # hist_narrow = hist_narrow[(hist_narrow[TIME_COL] >= lb2) & (hist_narrow[TIME_COL] <= ub2)].copy()

    # ---------- 4) Compute per-series base using SLIDING WINDOWS ----------
    with phase("Compute bases (sliding windows)"):
        bases = _compute_bases_sliding(series_ids, hist_all, weeks_req, H)
        # simple sanity
        assert not bases["base"].isna().any(), "Found NaT base; sliding windows failed."

    dbg = bases.merge(
        hist_all.groupby(ID_COL, observed=True)[TIME_COL].max().rename("kb_last").reset_index(),
        on=ID_COL, how="left"
    )
    rmax = pd.to_datetime(weeks_req.max())
    dbg["diff_w"] = ((rmax - dbg["kb_last"]).dt.days // 7)
    print("[DBG] bases (head):")
    print(dbg.head(5).to_string(index=False))


    # ---------- 5) Use prepatched history & cut to base ----------
    with phase("Pad history & cut to base"):
        pre = _load_prepatched_once()
    
        # Slice prepatched frame ONLY to the filtered series_ids
        keep_ids = pd.Index(series_ids, dtype=str)
        hist_padded = pre[pre[ID_COL].isin(keep_ids)].copy()
    
        # Safety: if customer filter was used, ensure slice is clean
        if customer_id:
            bad = hist_padded.loc[~hist_padded[ID_COL].astype(str).str.startswith(str(customer_id) + "||", na=False), ID_COL].unique()
            if len(bad) > 0:
                raise ValueError(f"Customer filter violation: found foreign series_ids in padded slice, e.g. {bad[:5]}")

        # If client provided 'history' overrides, update the padded frame for those (series, week)
        if history_df is not None and not history_df.empty:
            hh = history_df.copy()
            if ID_COL not in hh.columns and {"customer_id","item_id"}.issubset(hh.columns):
                hh[ID_COL] = hh["customer_id"].astype(str) + "||" + hh["item_id"].astype(str)
            hh = hh[[ID_COL, TIME_COL, TARGET_COL]].copy()
            hh[TIME_COL] = pd.to_datetime(hh[TIME_COL]).dt.to_period("W-MON").dt.to_timestamp("W-MON")
            hh[TARGET_COL] = pd.to_numeric(hh[TARGET_COL], errors="coerce").fillna(0.0)
    
            # index update (fast, no chained assignment warnings)
            hist_padded.set_index([ID_COL, TIME_COL], inplace=True)
            hh.set_index([ID_COL, TIME_COL], inplace=True)
            # overwrite sales where provided
            hist_padded.loc[hh.index, TARGET_COL] = hh[TARGET_COL]
            hist_padded.reset_index(inplace=True)
    
        # CUT TO <= base (request-specific)
        hist_padded = hist_padded.merge(bases[[ID_COL, "base"]], on=ID_COL, how="inner")
        hist_padded = hist_padded[hist_padded[TIME_COL] <= hist_padded["base"]].drop(columns=["base"])
    
        # Past-only real history for fallback means (<= base), using original KB scope
        hist_cut = hist_all.merge(bases[[ID_COL, "base"]], on=ID_COL, how="inner")
        hist_cut = hist_cut[hist_cut[TIME_COL] <= hist_cut["base"]].drop(columns=["base"], errors="ignore")
    
        # Synthesize zero history ONLY for series truly missing from KB (rare if KB is complete)
        have_in_hist = req[ID_COL].isin(hist_padded[ID_COL].unique())
        missing_series = req.loc[~have_in_hist, [ID_COL]].drop_duplicates()
        if not missing_series.empty:
            # create minimal zero history ending at base for those series
            rows = []
            for sid, b in bases[bases[ID_COL].isin(missing_series[ID_COL])][[ID_COL, "base"]].itertuples(index=False):
                start = b - pd.to_timedelta(MIN_FAKE_HISTORY_WEEKS - 1, unit="W")
                for wk in pd.date_range(start, b, freq="W-MON"):
                    rows.append((sid, wk, 0.0))
            synth = pd.DataFrame(rows, columns=[ID_COL, TIME_COL, TARGET_COL])
            # attach statics/exogs defaults
            for c in STATIC_COLS: synth[c] = "UNK"
            for ex in EXOG_WEEKLY: synth[ex] = 0.0
            synth = calendarize(synth, TIME_COL)
            hist_padded = pd.concat([hist_padded, synth], ignore_index=True)
    
        hist_padded = hist_padded.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
        print(f"[INFO] padded rows (sliced)={len(hist_padded):,} | series={hist_padded[ID_COL].nunique():,}")

    # ---------- 6) Static features (if available) ----------
    if all(c in hist_all.columns for c in STATIC_COLS):
        static_df = hist_all[[ID_COL] + STATIC_COLS].drop_duplicates([ID_COL])
    else:
        static_df = pd.DataFrame(columns=[ID_COL] + STATIC_COLS)
    static_feats_ag = static_df[[ID_COL] + [c for c in STATIC_COLS if c in static_df.columns]] \
        .drop_duplicates([ID_COL]) if not static_df.empty else None

    # ---------- 7) Roll-forward short series to base (batched with model) ----------
    with phase("Roll-forward to base (batched)"):
        hist_padded = extend_to_base_with_predictions(
            hist_padded, bases, predictor, static_feats_ag, H_model=H
        )

    # ---------- 8) Build context (past-only target + past exog) & known-future (calendar only) ----------
    
    # HARDEN DTYPES before building TimeSeriesDataFrame
    # Target must be numeric; EXOG must be numeric; statics should be strings; time must be datetime64[ns]
    hist_padded[TIME_COL] = to_wmon(hist_padded[TIME_COL])

    # numeric columns
    num_cols = [TARGET_COL] + [c for c in EXOG_WEEKLY if c in hist_padded.columns]
    for c in num_cols:
        hist_padded[c] = pd.to_numeric(hist_padded[c], errors="coerce")

    # target: fill NaNs with 0.0 (past-only)
    hist_padded[TARGET_COL] = hist_padded[TARGET_COL].fillna(0.0)

    # static features to string (if present)
    for c in STATIC_COLS:
        if c in hist_padded.columns:
            hist_padded[c] = hist_padded[c].astype(str).fillna("UNK")

    # sanity assert (helps catch regressions fast)
    if not np.issubdtype(hist_padded[TARGET_COL].dtype, np.number):
        dt = str(hist_padded[TARGET_COL].dtype)
        sample = hist_padded[TARGET_COL].dropna().head(3).tolist()
        raise TypeError(f"{TARGET_COL} dtype must be numeric, got {dt}; samples={sample}")
    hist_cols = [TARGET_COL] + CAL_KNOWN + EXOG_WEEKLY
    ts_context = TimeSeriesDataFrame.from_data_frame(
        hist_padded[[ID_COL, TIME_COL] + hist_cols],
        id_column=ID_COL, timestamp_column=TIME_COL,
        static_features_df=(static_feats_ag.copy(deep=True) if static_feats_ag is not None else None)
    )

    # NEW: guard to avoid AG resampling crash on empty context
    if getattr(ts_context, "num_items", 0) == 0 or len(ts_context) == 0:
        raise ValueError("Empty ts_context after padding; cannot build future grid. Check upstream fallbacks.")

    fut_grid = predictor.make_future_data_frame(data=ts_context)
    fut_grid = fut_grid.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})[[ID_COL, TIME_COL]]
    fut_grid[TIME_COL] = to_wmon(fut_grid[TIME_COL])
    fut_grid = calendarize(fut_grid, TIME_COL)
    kf_ts = TimeSeriesDataFrame.from_data_frame(
        fut_grid[[ID_COL, TIME_COL] + KNOWN_FUT_CAL],
        id_column=ID_COL, timestamp_column=TIME_COL
    )

    # ---------- 9) Predict ----------
    with phase("Predict"):
        forecasts = predictor.predict(data=ts_context, known_covariates=kf_ts)
        preds_long = preds_to_long(forecasts)
        preds_req = req.merge(preds_long, on=[ID_COL, TIME_COL], how="left")

    # ---------- 10) Fallback for NaNs ----------
    nan_mask = preds_req["y_pred"].isna()
    if nan_mask.any():
        item_means = hist_cut.groupby(ID_COL, observed=True)[TARGET_COL].mean()
        # assign in one shot, then coerce numeric
        filled = preds_req.loc[nan_mask, ID_COL].map(item_means)
        preds_req.loc[nan_mask, "y_pred"] = filled.astype(float)
    # coerce & fill remaining NaN -> 0.0
    preds_req["y_pred"] = pd.to_numeric(preds_req["y_pred"], errors="coerce").fillna(0.0)


    # ---------- 11) Attach truth for reporting (two streams) ----------
    def _as_wmon(dt_series: pd.Series) -> pd.Series:
        return pd.to_datetime(dt_series, errors="coerce").dt.to_period("W-MON").dt.to_timestamp("W-MON")
    
    # Normalize prediction keys
    preds_req = preds_req.copy()
    preds_req[ID_COL] = preds_req[ID_COL].astype(str)
    preds_req[TIME_COL] = _as_wmon(preds_req[TIME_COL])
    
    # Build per-series safe window (for hist truth)
    bases_local = bases[[ID_COL, "base"]].copy()
    bases_local["base"] = _as_wmon(bases_local["base"])
    # clamp to end of requested interval (no need—end_wmon is known, but keep it explicit)
    bases_local["truth_max"] = bases_local["base"].where(bases_local["base"] <= end_wmon, end_wmon)
    
    # (A) Hist truth <= base (leak-safe; will drive metrics)
    hist_truth = hist_all[[ID_COL, TIME_COL, TARGET_COL]].copy()
    hist_truth[ID_COL] = hist_truth[ID_COL].astype(str)
    hist_truth[TIME_COL] = _as_wmon(hist_truth[TIME_COL])
    hist_truth = hist_truth.merge(bases_local[[ID_COL, "truth_max"]], on=ID_COL, how="inner")
    hist_truth = hist_truth[hist_truth[TIME_COL] <= hist_truth["truth_max"]]
    hist_truth = (hist_truth
                  .rename(columns={TARGET_COL: "y_true_hist"})
                  .drop(columns=["truth_max"])
                  .drop_duplicates([ID_COL, TIME_COL], keep="last"))
    
    preds_req = preds_req.merge(hist_truth, on=[ID_COL, TIME_COL], how="left")
    preds_req["y_true_hist"] = pd.to_numeric(preds_req["y_true_hist"], errors="coerce")
    
    matched_hist = int(preds_req["y_true_hist"].notna().sum())
    total_rows = len(preds_req)
    print(f"[YTRUE] hist (<=base) matched: {matched_hist}/{total_rows} "
          f"({100.0*matched_hist/max(total_rows,1):.1f}%)", flush=True)
    
    # (B) AUX truth for prediction weeks ONLY (no base cap). Safe for reporting; NOT used for metrics.
    try:
        aux_truth = _load_aux_truth_once()
    except Exception as e:
        print(f"[YTRUE] AUX load failed: {e}", flush=True)
        aux_truth = pd.DataFrame(columns=[ID_COL, TIME_COL, "y_true"])

    if not aux_truth.empty:
        aux = aux_truth.copy()
        aux[ID_COL] = aux[ID_COL].astype(str)
        aux[TIME_COL] = _as_wmon(aux[TIME_COL])

        # 🔧 Use the actual forecast weeks present in preds_req (NOT weeks_req)
        requested_weeks_actual = preds_req[TIME_COL].drop_duplicates()
        aux = aux[aux[TIME_COL].isin(requested_weeks_actual)]

        aux = aux.drop_duplicates([ID_COL, TIME_COL], keep="last")

        preds_req = preds_req.merge(
            aux.rename(columns={"y_true": "y_true_aux"}),
            on=[ID_COL, TIME_COL],
            how="left",
        )
        preds_req["y_true_aux"] = pd.to_numeric(preds_req["y_true_aux"], errors="coerce")

        added_aux = int(preds_req["y_true_aux"].notna().sum())
        print(f"[YTRUE] AUX (prediction weeks) available rows: {added_aux}", flush=True)
    else:
        preds_req["y_true_aux"] = pd.NA
        print("[YTRUE] AUX empty; skipped.", flush=True)

    # Final y_true for Excel/reporting = coalesce(hist, aux)
    preds_req["y_true"] = preds_req["y_true_hist"].where(
        preds_req["y_true_hist"].notna(), preds_req["y_true_aux"]
    )

    # Ensure numeric types
    preds_req["y_pred"] = pd.to_numeric(preds_req["y_pred"], errors="coerce")
    preds_req["y_true"] = pd.to_numeric(preds_req["y_true"], errors="coerce")
    preds_req["y_true"] = preds_req["y_true"].fillna(0.0)


    # ---------- 12) Metrics ----------
    # We'll compute:
    #  - overall_req: using y_true (coalesced hist<=base OR AUX on forecast weeks)
    #  - per-individual (series) using y_true if available; else empty
    #  - per-group (customer group) using y_true if available; else empty
    #  - NEW: per item_group1 / item_group2 using y_true
    
    # Attach statics for item groups if available in KB
    sid_stat_cols = [ID_COL] + [c for c in STATIC_COLS if c in hist_all.columns]
    if len(sid_stat_cols) > 1:
        sid_stat = hist_all[sid_stat_cols].drop_duplicates([ID_COL], keep="last")
        preds_req = preds_req.merge(sid_stat, on=ID_COL, how="left")
    else:
        # ensure the columns exist for downstream grouping
        for c in STATIC_COLS:
            if c not in preds_req.columns:
                preds_req[c] = "UNK"
    
    # Coalesced-eval rows (prefer AUX if hist is missing)
    eval_rows_req = preds_req.dropna(subset=["y_true"]).copy()
    
    # Also keep the strict hist-only eval if you want to inspect leak-free metrics
    eval_rows_hist = preds_req.dropna(subset=["y_true_hist"]).copy()
    
    # Overall (prefer coalesced y_true so it's non-empty when AUX exists)
    if eval_rows_req.empty:
        overall = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    else:
        overall = metrics_np(
            eval_rows_req["y_true"].to_numpy(float),
            eval_rows_req["y_pred"].to_numpy(float)
        )
    
    # Per-individual (series) — use coalesced y_true
    if eval_rows_req.empty:
        metrics_per_individual = pd.DataFrame(columns=[ID_COL, "MAE", "RMSE", "MAPE"])
    else:
        rows = []
        for sid, g in eval_rows_req.groupby(ID_COL, observed=True):
            m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
            m[ID_COL] = sid
            rows.append(m)
        metrics_per_individual = (
            pd.DataFrame(rows, columns=[ID_COL, "MAE", "RMSE", "MAPE"])
            .sort_values(ID_COL)
        )
    
    # By customer group_id (if present in KB statics)
    if "group_id" in preds_req.columns and not eval_rows_req.empty:
        rows_g = []
        eg = eval_rows_req.copy()

        # If eval_rows_req doesn’t have group_id yet, bring it from preds_req
        if "group_id" not in eg.columns:
            eg = eg.merge(
                preds_req[[ID_COL, "group_id"]].drop_duplicates([ID_COL]),
                on=ID_COL, how="left"
            )

        # If a merge created suffixes, coalesce back into a single 'group_id'
        if "group_id_x" in eg.columns or "group_id_y" in eg.columns:
            gx = eg["group_id_x"] if "group_id_x" in eg.columns else pd.Series(pd.NA, index=eg.index)
            gy = eg["group_id_y"] if "group_id_y" in eg.columns else pd.Series(pd.NA, index=eg.index)
            eg["group_id"] = gx.where(gx.notna(), gy)
            eg.drop(columns=[c for c in ["group_id_x", "group_id_y"] if c in eg.columns],
                    inplace=True)

        for grp, g in eg.groupby("group_id", dropna=False, observed=True):
            m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
            m["group_id"] = grp
            rows_g.append(m)

        metrics_by_group = (
            pd.DataFrame(rows_g, columns=["group_id", "MAE", "RMSE", "MAPE"])
              .sort_values("group_id")
        )
    else:
        metrics_by_group = pd.DataFrame(columns=["group_id", "MAE", "RMSE", "MAPE"])
    
    # NEW: metrics by item_group1
    if "item_group1" in preds_req.columns and not eval_rows_req.empty:
        rows_g1 = []
        eg1 = eval_rows_req.copy()
        if "item_group1" not in eg1.columns:
            eg1 = eg1.merge(
                preds_req[[ID_COL, "item_group1"]].drop_duplicates([ID_COL]),
                on=ID_COL, how="left"
            )
        if "item_group1_x" in eg1.columns or "item_group1_y" in eg1.columns:
            gx = eg1["item_group1_x"] if "item_group1_x" in eg1.columns else pd.Series(pd.NA, index=eg1.index)
            gy = eg1["item_group1_y"] if "item_group1_y" in eg1.columns else pd.Series(pd.NA, index=eg1.index)
            eg1["item_group1"] = gx.where(gx.notna(), gy)
            eg1.drop(columns=[c for c in ["item_group1_x", "item_group1_y"] if c in eg1.columns],
                     inplace=True)
        for ig1, g in eg1.groupby("item_group1", dropna=False, observed=True):
            m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
            m["item_group1"] = ig1 if pd.notna(ig1) else "UNK"
            rows_g1.append(m)
        metrics_by_item_group1 = (
            pd.DataFrame(rows_g1, columns=["item_group1", "MAE", "RMSE", "MAPE"])
              .sort_values("item_group1")
        )
    else:
        metrics_by_item_group1 = pd.DataFrame(columns=["item_group1", "MAE", "RMSE", "MAPE"])
    
    # NEW: metrics by item_group2  (robust to suffixes and missing labels)
    if not eval_rows_req.empty:
        eg2 = eval_rows_req.copy()

        # Attach item_group2 if eval_rows_req doesn't already have it
        if "item_group2" not in eg2.columns and "item_group2" in preds_req.columns:
            eg2 = eg2.merge(
                preds_req[[ID_COL, "item_group2"]].drop_duplicates([ID_COL]),
                on=ID_COL, how="left"
            )

        # If a merge created suffixes, coalesce to a single 'item_group2'
        if "item_group2_x" in eg2.columns or "item_group2_y" in eg2.columns:
            gx = eg2["item_group2_x"] if "item_group2_x" in eg2.columns else pd.Series(pd.NA, index=eg2.index)
            gy = eg2["item_group2_y"] if "item_group2_y" in eg2.columns else pd.Series(pd.NA, index=eg2.index)
            eg2["item_group2"] = gx.where(gx.notna(), gy)
            eg2.drop(columns=[c for c in ["item_group2_x", "item_group2_y"] if c in eg2.columns], inplace=True)

        # Only compute if we truly have a single 'item_group2' column now
        if "item_group2" in eg2.columns:
            rows_g2 = []
            for ig2, g in eg2.groupby("item_group2", dropna=False, observed=True):
                m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
                m["item_group2"] = ig2 if pd.notna(ig2) else "UNK"
                rows_g2.append(m)
            metrics_by_item_group2 = (
                pd.DataFrame(rows_g2, columns=["item_group2", "MAE", "RMSE", "MAPE"])
                  .sort_values("item_group2")
            )
        else:
            # Label truly not available anywhere → return empty table
            metrics_by_item_group2 = pd.DataFrame(columns=["item_group2", "MAE", "RMSE", "MAPE"])
    else:
        metrics_by_item_group2 = pd.DataFrame(columns=["item_group2", "MAE", "RMSE", "MAPE"])

    # ---------- 12b) Metrics on PREDICTION WINDOW (y_true zero-filled) ----------
    # We compute MAE, RMSE, sMAPE for the requested prediction rows (weeks_req),
    # using y_true==0.0 where actuals are not yet available (per user request).

    def _agg_metrics_uppercase(df: pd.DataFrame) -> pd.Series:
        y = df["y_true"].to_numpy(dtype=float)
        yhat = df["y_pred"].to_numpy(dtype=float)
        if y.size == 0:
            return pd.Series({"MAE": np.nan, "RMSE": np.nan, "SMAPE": np.nan})
        err = yhat - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        eps = 1e-8
        smape = float(100.0 * np.mean(2.0 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps)))
        return pd.Series({"MAE": mae, "RMSE": rmse, "SMAPE": smape})

    # Per-series metrics over prediction rows
    pred_metrics_per_individual = (
        preds_req.groupby(ID_COL, observed=True)
                 .apply(_agg_metrics_uppercase)
                 .reset_index()
                 .rename(columns={ID_COL: "series_id"})
                 .sort_values("series_id")
                 .reset_index(drop=True)
    )

    # Per-group metrics over prediction rows – use group_id already attached to preds_req
    if "group_id" in preds_req.columns:
        pred_metrics_by_group = (
            preds_req.groupby("group_id", dropna=False, observed=True)
                     .apply(_agg_metrics_uppercase)
                     .reset_index()
                     .sort_values("group_id")
                     .reset_index(drop=True)
        )
    else:
        pred_metrics_by_group = pd.DataFrame(columns=["group_id", "MAE", "RMSE", "SMAPE"])

    # OVERWRITE the hist-only per-series/group in-memory tables that go into summary,
    # but KEEP 'overall' as the (leak-safe) historical overall you already computed.
    metrics_per_individual = pred_metrics_per_individual
    metrics_by_group = pred_metrics_by_group

            
    # ---------- 13) Package outputs & return ----------
    # 13a) predictions list (string weeks) – legacy field; now include y_true (0 if unknown)
    preds_list = preds_req[[ID_COL, TIME_COL, "y_pred", "y_true"]].copy()
    preds_list.rename(columns={ID_COL: "series_id"}, inplace=True)
    
    # Convert week to ISO string; if TIME_COL already 'week', don't drop it
    preds_list["week"] = pd.to_datetime(preds_list[TIME_COL]).dt.date.astype(str)
    if TIME_COL != "week":
        preds_list = preds_list.drop(columns=[TIME_COL])
    
    preds_list["y_pred"] = pd.to_numeric(preds_list["y_pred"], errors="coerce").astype(float)
    preds_list["y_true"] = pd.to_numeric(preds_list["y_true"], errors="coerce").fillna(0.0).astype(float)
    
    predictions_records = preds_list[["series_id", "week", "y_pred", "y_true"]].to_dict(orient="records")

    # 13b) summary dict
    summary = {
        "rows_pred": int(len(preds_req)),
        "series_pred": int(preds_req[ID_COL].nunique()),
        "overall": _clean_overall(overall),  # NaN/Inf -> None
        "predictions": preds_req.to_dict(orient="records"),
        "metrics_per_individual": metrics_per_individual.to_dict(orient="records"),
        "metrics_by_group": metrics_by_group.to_dict(orient="records"),
        "metrics_by_item_group1": metrics_by_item_group1.to_dict(orient="records"),
        "metrics_by_item_group2": metrics_by_item_group2.to_dict(orient="records"),
    }

    # 13c) DataFrame for API callers (CSV/Excel export)
    preds_df = (
        preds_req[[ID_COL, TIME_COL, "y_pred", "y_true"]]
        .rename(columns={ID_COL: "series_id", TIME_COL: "week"})
        .copy()
    )
    preds_df["week"]  = pd.to_datetime(preds_df["week"]).dt.date.astype(str)
    preds_df["y_pred"] = pd.to_numeric(preds_df["y_pred"], errors="coerce")
    preds_df["y_true"] = pd.to_numeric(preds_df["y_true"], errors="coerce").fillna(0.0)
    
    preds_df = preds_df.sort_values(["series_id", "week"]).reset_index(drop=True)


    return summary, preds_df






