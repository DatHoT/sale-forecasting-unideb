from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

from .helpers import (
    ID_COL, TIME_COL, TARGET_COL, STATIC_COLS, CAL_KNOWN, EXOG_WEEKLY, KNOWN_FUT_CAL,
    phase, dedup, ensure_sales, to_wmon, calendarize, exog_weekly_table, pad_until_base,
    synthetic_zero_history, choose_per_series_bases, extend_to_base_with_predictions,
    preds_to_long, metrics_np, save_artifacts
)

def run_report(
    hist_df: pd.DataFrame,
    req_df: pd.DataFrame,
    predictor_path: str,
    out_dir: Path,
    *,
    future_weeks: int = 2,
    fill_missing_zero: bool = True,
    min_fake_history_weeks: int = 12,
):
    stem = "req"

    # Normalize HIST & REQ (from JSON input, not files)
    with phase("Normalize HIST & REQ"):
        hist = dedup(hist_df.copy())
        if any((c == TARGET_COL or c.startswith(TARGET_COL)) for c in hist.columns):
            hist = ensure_sales(hist, TARGET_COL)
        # series_id may come from JSON or be already set by router
        if ID_COL not in hist.columns:
            raise ValueError("history must include 'series_id' per row (router should have built it).")

        need = {ID_COL, TIME_COL, TARGET_COL}
        if not need.issubset(hist.columns):
            raise ValueError(f"history missing {need}. Got: {hist.columns.tolist()[:20]} ...")

        hist[ID_COL]   = hist[ID_COL].astype(str)
        hist[TIME_COL] = to_wmon(hist[TIME_COL])
        hist[TARGET_COL] = pd.to_numeric(hist[TARGET_COL], errors="coerce").fillna(0.0)
        for c in STATIC_COLS:
            if c in hist.columns: hist[c] = hist[c].astype(str).fillna("UNK")
        for ex in EXOG_WEEKLY:
            if ex in hist.columns: hist[ex] = pd.to_numeric(hist[ex], errors="coerce")
        hist = calendarize(hist, TIME_COL)
        keep = [ID_COL, TIME_COL, TARGET_COL] + [c for c in STATIC_COLS if c in hist.columns] + CAL_KNOWN + [c for c in EXOG_WEEKLY if c in hist.columns]
        hist = hist[keep].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

        req = dedup(req_df.copy())
        req[ID_COL] = req[ID_COL].astype(str)
        req[TIME_COL] = to_wmon(req[TIME_COL])
        req = req.drop_duplicates().sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Load predictor & clamp REQ span ≤ H (also extend by up to future_weeks per series)
    with phase("Load predictor"):
        predictor = TimeSeriesPredictor.load(predictor_path)
        H = int(getattr(predictor, "prediction_length", 12))
        print(f"[MODEL] Loaded: {predictor_path} | prediction_length={H}")

        req_span = req.groupby(ID_COL)[TIME_COL].agg(['min', 'max']).reset_index()
        req_span['span_h'] = ((req_span['max'] - req_span['min']).dt.days // 7) + 1
        req_span['addable'] = (H - req_span['span_h']).clip(lower=0)
        fut_rows = []
        for _, r in req_span.iterrows():
            sid, rmax = r[ID_COL], r['max']
            kmax = int(min(future_weeks, r['addable']))
            for k in range(1, kmax + 1):
                fut_rows.append((sid, rmax + pd.to_timedelta(k, unit="W")))
        if fut_rows:
            req_extra = pd.DataFrame(fut_rows, columns=[ID_COL, TIME_COL])
            req = (pd.concat([req, req_extra], ignore_index=True)
                     .drop_duplicates()
                     .sort_values([ID_COL, TIME_COL])
                     .reset_index(drop=True))

    bases = choose_per_series_bases(req, H)

    # Split requested series into present / missing in history
    have_in_hist = req[ID_COL].isin(hist[ID_COL].unique())
    missing_series = req.loc[~have_in_hist, [ID_COL]].drop_duplicates()
    if not missing_series.empty:
        print(f"[INFO] {len(missing_series):,} requested series not in history — "
              f"{'synthesizing zeros' if fill_missing_zero else 'dropping them'}.")

    req = req[have_in_hist | fill_missing_zero].reset_index(drop=True)

    # Build exog & pad histories
    with phase("Build exog & pad history"):
        ex_tbl = exog_weekly_table(hist)
        hist_padded = pad_until_base(hist, bases, ex_tbl)
        if fill_missing_zero and not missing_series.empty:
            bases_missing = bases[bases[ID_COL].isin(missing_series[ID_COL])]
            synth = synthetic_zero_history(bases_missing, min_fake_history_weeks)
            for ex in EXOG_WEEKLY:
                if ex not in synth.columns:
                    synth[ex] = 0.0
                synth[ex] = pd.to_numeric(synth[ex], errors="coerce").fillna(0.0)
            hist_padded = pd.concat([hist_padded, synth], ignore_index=True)

        hist_padded = hist_padded.merge(bases[[ID_COL, "base"]], on=ID_COL, how="left")
        hist_padded = hist_padded[hist_padded[TIME_COL] <= hist_padded["base"]].drop(columns=["base"])

        hist_cut = hist.merge(bases[[ID_COL, "base"]], on=ID_COL, how="inner")
        hist_cut = hist_cut[hist_cut[TIME_COL] <= hist_cut["base"]].drop(columns=["base"], errors="ignore")

        keep_ids = req[ID_COL].drop_duplicates()
        hist_padded = hist_padded[hist_padded[ID_COL].isin(keep_ids)]
        print(f"[INFO] padded rows={len(hist_padded):,} | series={hist_padded[ID_COL].nunique():,}")

    # Static features (if included in JSON)
    if all(c in hist.columns for c in STATIC_COLS):
        stat_real = hist[[ID_COL] + STATIC_COLS].drop_duplicates([ID_COL])
    else:
        stat_real = pd.DataFrame(columns=[ID_COL] + STATIC_COLS)
    stat_missing = pd.DataFrame({ID_COL: missing_series[ID_COL].unique()}) if not missing_series.empty else pd.DataFrame({ID_COL: []})
    for c in STATIC_COLS: stat_missing[c] = "UNK"
    static_df = pd.concat([stat_real, stat_missing], ignore_index=True).drop_duplicates([ID_COL])
    static_df = static_df[static_df[ID_COL].isin(keep_ids)]
    static_feats_ag = static_df[[ID_COL] + [c for c in STATIC_COLS if c in static_df.columns]].drop_duplicates([ID_COL]) if not static_df.empty else None

    # Roll-forward to base (batched)
    with phase("Roll-forward to base (batched)"):
        hist_padded = hist_padded.merge(bases[[ID_COL, "base"]], on=ID_COL, how="left")
        hist_padded = hist_padded[hist_padded[TIME_COL] <= hist_padded["base"]]
        hist_padded = extend_to_base_with_predictions(
            hist_padded.drop(columns=["base"], errors="ignore"),
            bases,
            predictor,
            static_feats_ag,
            H_model=H,
        )

    # Context & future grid
    hist_cols = [TARGET_COL] + CAL_KNOWN + EXOG_WEEKLY
    ts_context = TimeSeriesDataFrame.from_data_frame(
        hist_padded[[ID_COL, TIME_COL] + hist_cols],
        id_column=ID_COL,
        timestamp_column=TIME_COL,
        static_features_df=(static_feats_ag.copy(deep=True) if static_feats_ag is not None else None)
    )

    with phase("Build future grid"):
        fut_grid = predictor.make_future_data_frame(data=ts_context)
        fut_grid = fut_grid.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})[[ID_COL, TIME_COL]]
        fut_grid[TIME_COL] = to_wmon(fut_grid[TIME_COL])
        fut_grid = calendarize(fut_grid, TIME_COL)
        kf_ts = TimeSeriesDataFrame.from_data_frame(
            fut_grid[[ID_COL, TIME_COL] + KNOWN_FUT_CAL],
            id_column=ID_COL, timestamp_column=TIME_COL
        )

    # Predict
    with phase("Predict"):
        forecasts = predictor.predict(data=ts_context, known_covariates=kf_ts)
        preds_long = preds_to_long(forecasts)
        print(f"[INFO] preds rows={len(preds_long):,} | series={preds_long[ID_COL].nunique():,}")

    # Align to requested grid & fallbacks
    preds_req = req.merge(preds_long, on=[ID_COL, TIME_COL], how="left")
    truth_from_hist = (
        hist[[ID_COL, TIME_COL, TARGET_COL]]
        .rename(columns={TARGET_COL: "y_true"})
        .drop_duplicates([ID_COL, TIME_COL])
    )
    preds_req = preds_req.merge(truth_from_hist, on=[ID_COL, TIME_COL], how="left")

    nan_mask = preds_req["y_pred"].isna()
    n_nan = int(nan_mask.sum())
    print(f"[FALLBACK] NaN forecast rows: {n_nan:,}")
    if n_nan > 0:
        item_means = hist_cut.groupby(ID_COL, observed=True)[TARGET_COL].mean()
        preds_req.loc[nan_mask, "y_pred"] = preds_req.loc[nan_mask, ID_COL].map(item_means)
        preds_req["y_pred"].fillna(0.0, inplace=True)

    # Metrics (only where y_true exists in provided history)
    eval_rows = preds_req.dropna(subset=["y_true"]).copy()
    if eval_rows.empty:
        overall = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        metrics_per_individual = pd.DataFrame(columns=[ID_COL, "MAE", "RMSE", "MAPE"])
        metrics_by_group = pd.DataFrame(columns=["group_id", "MAE", "RMSE", "MAPE"])
    else:
        overall = metrics_np(eval_rows["y_true"].to_numpy(float), eval_rows["y_pred"].to_numpy(float))
        rows = []
        for sid, g in eval_rows.groupby(ID_COL, observed=True):
            m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
            m[ID_COL] = sid
            rows.append(m)
        metrics_per_individual = pd.DataFrame(rows, columns=[ID_COL, "MAE", "RMSE", "MAPE"]).sort_values(ID_COL)

        if "group_id" in hist.columns:
            eg = eval_rows.merge(hist[[ID_COL, "group_id"]].drop_duplicates([ID_COL]), on=ID_COL, how="left")
            rows_g = []
            for grp, g in eg.groupby("group_id", dropna=False, observed=True):
                m = metrics_np(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
                m["group_id"] = grp
                rows_g.append(m)
            metrics_by_group = pd.DataFrame(rows_g, columns=["group_id", "MAE", "RMSE", "MAPE"]).sort_by("group_id") if rows_g else pd.DataFrame(columns=["group_id", "MAE", "RMSE", "MAPE"])
        else:
            metrics_by_group = pd.DataFrame(columns=["group_id", "MAE", "RMSE", "MAPE"])

    metrics_overall = pd.DataFrame(
        [{"metric": "MAE", "value": overall["MAE"]},
         {"metric": "RMSE", "value": overall["RMSE"]},
         {"metric": "MAPE", "value": overall["MAPE"]}]
    )

    artifacts = save_artifacts(
        preds_req,
        metrics_per_individual,
        metrics_by_group,
        metrics_overall,
        out_dir=out_dir,
        stem=stem,
    )

    return {
        "rows_pred": int(len(preds_req)),
        "series_pred": int(preds_req[ID_COL].nunique()),
        "overall": overall,
        "artifacts": artifacts,
    }, preds_req
