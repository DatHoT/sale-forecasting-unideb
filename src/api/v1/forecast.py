from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
import pandas as pd
import uuid
from src.models.schemas import (
    PredictPairsRequest, ForecastByCustomerRequest, ForecastByGroupRequest, JobResponse
)
from src.core.config import FUTURE_WEEKS, ITEM_MAPPING_PATH, EXPORT_DIR
from src.services.engine import run_report
from src.services.helpers import ID_COL, TIME_COL
from src.core.config import ITEM_MAPPING_PATH
from datetime import datetime
router = APIRouter()

def _attach_groups_with_debug(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """
    Attach item_group1/2 with robust matching and debug:
      1) exact normalized join
      2) if match rate < 5%, try zero-padded numeric join to dominant width in mapping
    """
    mp = _load_item_map_once().copy()

    # Prepare join keys (exact normalized)
    df = df.copy()
    df["itemcode"] = _norm_itemcode_series(df["itemcode"])
    mp["itemcode"] = _norm_itemcode_series(mp["itemcode"])

    merged = df.merge(mp, on="itemcode", how="left")

    # Debug: match rate & sample of misses
    miss = merged["item_group1"].isna()
    match_rate = 100.0 * (~miss).mean()
    n_total = len(merged)
    n_match = int((~miss).sum())
    print(f"[DBG] mapping match rate ({where} exact): {match_rate:.1f}% ({n_match}/{n_total})", flush=True)

    if n_total and match_rate < 5.0:
        # Fallback: numeric-only + zero-padding to dominant width in mapping
        width = (mp["itemcode"].str.len().mode().iat[0]
                 if not mp["itemcode"].empty else 0)

        def _digits(s: pd.Series) -> pd.Series:
            return s.str.replace(r"\D", "", regex=True)

        dfx = merged.copy()
        mpx = mp.copy()

        dfx["__knum"] = _digits(dfx["itemcode"])
        mpx["__knum"] = _digits(mpx["itemcode"])

        if width > 0:
            dfx["__knum"] = dfx["__knum"].str.zfill(width)
            mpx["__knum"] = mpx["__knum"].str.zfill(width)

        fallback = dfx.merge(mpx[["__knum","item_group1","item_group2"]],
                             on="__knum", how="left", suffixes=("","_fb"))

        # Prefer exact, fill remaining with fallback
        for col in ("item_group1", "item_group2"):
            merged[col] = merged[col].where(merged[col].notna(), fallback[f"{col}_fb"])

        miss2 = merged["item_group1"].isna()
        match_rate2 = 100.0 * (~miss2).mean()
        n_match2 = int((~miss2).sum())
        print(f"[DBG] mapping match rate ({where} padded): {match_rate2:.1f}% ({n_match2}/{n_total}), width={width}", flush=True)

        # Show a few misses to eyeball why join fails
        if miss2.any():
            sample_miss = merged.loc[miss2, "itemcode"].head(10).tolist()
            print(f"[DBG] sample missing itemcodes ({where}): {sample_miss}", flush=True)

    # Final fill UNK
    merged["item_group1"] = merged["item_group1"].fillna("UNK")
    merged["item_group2"] = merged["item_group2"].fillna("UNK")
    return merged

_ITEM_MAP = None
def _load_item_map_once() -> pd.DataFrame:
    """Load and normalize the mapping once."""
    global _ITEM_MAP
    if _ITEM_MAP is None:
        mp = pd.read_parquet(ITEM_MAPPING_PATH)

        # Canonical columns
        ren = {}
        for want in ("item_group1", "item_group2", "itemcode"):
            if want not in mp.columns:
                for c in mp.columns:
                    if c.lower() == want:
                        ren[c] = want
                        break
        if ren:
            mp = mp.rename(columns=ren)
        missing = {"item_group1","item_group2","itemcode"} - set(mp.columns)
        if missing:
            raise RuntimeError(f"Item mapping missing required columns: {missing}")

        # Normalize
        mp["itemcode"] = _norm_itemcode_series(mp["itemcode"])
        mp["item_group1"] = (mp["item_group1"].astype(str).str.strip()
                             .replace({"nan": None, "None": None}).fillna("UNK"))
        mp["item_group2"] = (mp["item_group2"].astype(str).str.strip()
                             .replace({"nan": None, "None": None}).fillna("UNK"))

        # Drop dups (last wins)
        mp = mp.drop_duplicates(subset=["itemcode"], keep="last").reset_index(drop=True)

        _ITEM_MAP = mp
        print(f"[MAP] loaded mapping rows={len(_ITEM_MAP):,} | sample={_ITEM_MAP['itemcode'].head(3).tolist()}", flush=True)
    return _ITEM_MAP

def _norm_itemcode_series(s: pd.Series) -> pd.Series:
    """Normalize itemcode strings: strip spaces, tabs, NBSP; drop a trailing .0"""
    s = s.astype(str)
    s = s.str.replace("\u00A0", " ", regex=False)  # NBSP -> space
    s = s.str.replace("\t", "", regex=False)
    s = s.str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)     # 54021.0 -> 54021
    return s

def _check_interval_weeks(start_date, end_date, max_weeks=12):
    # Normalize ordering
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    # Align both to W-MON and compute inclusive week count
    s = pd.to_datetime(start_date).to_period("W-MON").start_time
    e = pd.to_datetime(end_date).to_period("W-MON").start_time
    delta_weeks = int(((e - s).days // 7) + 1)
    if delta_weeks < 1:
        delta_weeks = 1
    if delta_weeks > max_weeks:
        raise HTTPException(status_code=400, detail=f"Interval length {delta_weeks} weeks exceeds max {max_weeks}.")

def _history_to_df(history_rows):
    if not history_rows:
        return None
    df = pd.DataFrame([h.model_dump() for h in history_rows])
    if "series_id" not in df.columns:
        if {"customer_id","item_id"}.issubset(df.columns):
            df["series_id"] = df["customer_id"].astype(str) + "||" + df["item_id"].astype(str)
        else:
            raise HTTPException(400, "Each history row must include series_id or (customer_id and item_id).")
    if "week" not in df.columns:
        raise HTTPException(400, "Each history row must include 'week' (ISO date).")
    return df

@router.post("/predict/pairs", response_model=JobResponse)
def predict_pairs(payload: PredictPairsRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)
    job_id = uuid.uuid4().hex

    summary, preds_df = run_report(
        interval_start=payload.interval.start_date,
        interval_end=payload.interval.end_date,
        customer_id=None,
        group_id=None,
    )

    if preds_df is None or preds_df.empty:
        return JobResponse(
            job_id=job_id,
            rows_pred=0,
            series_pred=0,
            overall=summary.get("overall", {}),
            predictions=[],
            metrics_per_individual=summary.get("metrics_per_individual", []),
            metrics_by_group=summary.get("metrics_by_group", []),
            predictions_by_item_group1=[],
            predictions_by_item_group2=[],
        )

    # df = preds_df.loc[:, ["series_id", "week", "y_pred"]].copy()
    cols = ["series_id", "week", "y_pred"] + (["y_true"] if "y_true" in preds_df.columns else [])
    df = preds_df.loc[:, cols].copy()
    
    df["series_id"] = df["series_id"].astype(str).str.strip()
    parts = df["series_id"].astype(str).str.split(r"\|\|", n=1, expand=True)
    if parts.shape[1] == 2:
        df["customer_id"] = parts[0].str.strip()
        df["itemcode"] = parts[1].fillna("").astype(str).str.strip()
    else:
        df["customer_id"] = ""
        df["itemcode"] = df["series_id"]

    df = _attach_groups_with_debug(df, where="pairs")


    df["week"] = pd.to_datetime(df["week"]).dt.date.astype(str)

    by_series = (df.groupby(["series_id","week"], as_index=False)["y_pred"].sum()
                   .sort_values(["series_id","week"]).reset_index(drop=True))
    by_g1 = (df.groupby(["customer_id","item_group1","week"], as_index=False)["y_pred"].sum()
               .sort_values(["customer_id","week","item_group1"]).reset_index(drop=True))
    by_g2 = (df.groupby(["customer_id","item_group2","week"], as_index=False)["y_pred"].sum()
               .sort_values(["customer_id","week","item_group2"]).reset_index(drop=True))

    predictions_records = [
        {"series_id": r.series_id, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_series.itertuples(index=False)
    ]
    predictions_by_item_group1 = [
        {"customer_id": r.customer_id, "item_group1": r.item_group1, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_g1.itertuples(index=False)
    ]
    predictions_by_item_group2 = [
        {"customer_id": r.customer_id, "item_group2": r.item_group2, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_g2.itertuples(index=False)
    ]

    rows_pred = len(predictions_records)
    series_pred = int(df["series_id"].nunique())

    # ---------- NEW: optional Excel export, path goes into summary ----------
    if getattr(payload, "export_excel", False):
        import os
        from datetime import datetime
        os.makedirs(EXPORT_DIR, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        excel_path = os.path.join(EXPORT_DIR, f"predict_pairs_{ts}_{job_id}.xlsx")
    
        # 1) Build a series-level frame with y_true for the same (series_id, week)
        preds_for_excel = preds_df.copy()
        preds_for_excel["week"] = pd.to_datetime(preds_for_excel["week"]).dt.date.astype(str)

        # Ensure y_true exists in df (if engine didn’t return it, keep as NA)
        if "y_true" not in df.columns:
            df["y_true"] = pd.NA

        # 2) Merge y_true onto df; coalesce any suffixes back to 'y_true'
        df_excel = df.merge(
            preds_for_excel[["series_id", "week", "y_true"]],
            on=["series_id", "week"],
            how="left",
            suffixes=("", "_from_preds"),
        )
        if "y_true_from_preds" in df_excel.columns:
            # prefer df.y_true if present, else take merged
            df_excel["y_true"] = df_excel["y_true"].where(
                df_excel["y_true"].notna(), df_excel["y_true_from_preds"]
            )
            df_excel.drop(columns=["y_true_from_preds"], inplace=True)

        # 3) Guarantee numeric y_pred / y_true for aggregation
        for c in ("y_pred", "y_true"):
            if c not in df_excel.columns:
                df_excel[c] = 0.0
            df_excel[c] = pd.to_numeric(df_excel[c], errors="coerce").fillna(0.0)

        # Group ONLY by item_group1 + week
        by_g1_xl = (
            df_excel
            .groupby(["item_group1", "week"], as_index=False)[["y_pred", "y_true"]]
            .sum()
            .sort_values(["week", "item_group1"])
            .reset_index(drop=True)
        )
        
        # Group ONLY by item_group2 + week
        by_g2_xl = (
            df_excel
            .groupby(["item_group2", "week"], as_index=False)[["y_pred", "y_true"]]
            .sum()
            .sort_values(["week", "item_group2"])
            .reset_index(drop=True)
        )

    
        # 3) Metrics tables from summary (if present)
        mpi = summary.get("metrics_per_individual", [])
        mbg = summary.get("metrics_by_group", [])
        mbig1 = summary.get("metrics_by_item_group1", [])
        mbig2 = summary.get("metrics_by_item_group2", [])
        ovl = summary.get("overall", {})
    
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as w:
            # Series-level predictions with truth
            cols_series = ["series_id", "week", "y_pred", "y_true", "y_true_hist", "y_true_aux"]
            present = [c for c in cols_series if c in preds_for_excel.columns]
            preds_for_excel[present].to_excel(w, sheet_name="predictions_by_series", index=False)
    
            # Grouped predictions with truth
            by_g1_xl.to_excel(w, sheet_name="predictions_by_item_group1", index=False)
            by_g2_xl.to_excel(w, sheet_name="predictions_by_item_group2", index=False)
    
            # Metrics sheets
            pd.DataFrame([ovl]).to_excel(w, sheet_name="metrics_overall", index=False)
            if mpi:
                pd.DataFrame(mpi).to_excel(w, sheet_name="metrics_per_series", index=False)
            if mbg:
                pd.DataFrame(mbg).to_excel(w, sheet_name="metrics_by_group", index=False)
            if mbig1:
                pd.DataFrame(mbig1).to_excel(w, sheet_name="metrics_by_item_group1", index=False)
            if mbig2:
                pd.DataFrame(mbig2).to_excel(w, sheet_name="metrics_by_item_group2", index=False)
    
            # META
            meta = pd.DataFrame([{
                "interval_start": payload.interval.start_date,
                "interval_end": payload.interval.end_date,
                "rows_pred": rows_pred,
                "series_pred": series_pred,
                "job_id": job_id,
            }])
            meta.to_excel(w, sheet_name="META", index=False)
    
            readme = pd.DataFrame({
                "note": [
                    "Workbook generated by /api/v1/predict/pairs.",
                    "predictions_by_series includes y_pred and y_true (coalesced from hist<=base or AUX).",
                    "Grouped sheets aggregate both y_pred and y_true.",
                    "metrics_overall uses coalesced y_true.",
                ]
            })
            readme.to_excel(w, sheet_name="README", index=False)
    
        # Put the artifact path into summary (no schema change)
        if isinstance(summary, dict):
            summary = dict(summary)
            summary["export_path"] = excel_path



    return JobResponse(
        job_id=job_id,
        rows_pred=rows_pred,
        series_pred=series_pred,
        overall=summary.get("overall", {}),
        predictions=predictions_records,
        metrics_per_individual=summary.get("metrics_per_individual", []),
        metrics_by_group=summary.get("metrics_by_group", []),
        predictions_by_item_group1=predictions_by_item_group1,
        predictions_by_item_group2=predictions_by_item_group2,
    )

@router.post("/forecast/customer", response_model=JobResponse)
def forecast_customer(payload: ForecastByCustomerRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)
    job_id = uuid.uuid4().hex  # single job_id

    # 1) Run the engine (already filtered by customer_id in engine)
    summary, preds_df = run_report(
        interval_start=payload.interval.start_date,
        interval_end=payload.interval.end_date,
        customer_id=payload.customer_id,
        group_id=None,
    )

    # Guard: if engine returned nothing, short-circuit with empty result
    if preds_df is None or preds_df.empty:
        return JobResponse(
            job_id=job_id,
            rows_pred=0,
            series_pred=0,
            overall=summary.get("overall", {}),
            predictions=[],
            metrics_per_individual=summary.get("metrics_per_individual", []),
            metrics_by_group=summary.get("metrics_by_group", []),
            predictions_by_item_group1=[],
            predictions_by_item_group2=[],
        )

    # 2) Core columns
    # df = preds_df.loc[:, ["series_id", "week", "y_pred"]].copy()
    cols = ["series_id", "week", "y_pred"] + (["y_true"] if "y_true" in preds_df.columns else [])
    df = preds_df.loc[:, cols].copy()
    
    df["series_id"] = df["series_id"].astype(str).str.strip()
    req_cust = str(payload.customer_id).strip()

    # 3) Parse customer_id, itemcode from series_id "customer||item"
    # Robust split: handle missing delimiter gracefully
    parts = df["series_id"].astype(str).str.split(r"\|\|", n=1, expand=True)
    if parts.shape[1] == 2:
        df["customer_id"] = parts[0].str.strip()
        df["itemcode"] = parts[1].fillna("").astype(str).str.strip()
    else:
        # No delimiter? fallback: customer_id unknown, itemcode whole string
        df["customer_id"] = ""
        df["itemcode"] = df["series_id"]

    # 4) Strict filter (normalized)
    # before = len(df)
    # df = df[df["customer_id"].str.strip().str.casefold() == req_cust.casefold()].copy()
    # after = len(df)
    # print(f"[DBG] customer filter kept {after}/{before} rows for '{req_cust}'", flush=True)

    # # If engine already filtered by customer, this should be a no-op; if it drops to 0,
    # # it means series_id formatting differs (e.g., extra spaces, different delimiter).
    # if df.empty:
    #     return JobResponse(
    #         job_id=job_id,
    #         rows_pred=0,
    #         series_pred=0,
    #         overall=summary.get("overall", {}),
    #         predictions=[],
    #         metrics_per_individual=summary.get("metrics_per_individual", []),
    #         metrics_by_group=summary.get("metrics_by_group", []),
    #         predictions_by_item_group1=[],
    #         predictions_by_item_group2=[],
    #     )

    # 5) Attach item mapping (left-join; won’t drop rows)
    mp = _load_item_map_once()
    mp = mp.copy()
    mp["itemcode"] = mp["itemcode"].astype(str).str.strip()
    # normalize on both sides the same way
    df["itemcode"] = _norm_itemcode_series(df["itemcode"])
    df = _attach_groups_with_debug(df, where="pairs")


    # 6) Normalize week string for JSON
    df["week"] = pd.to_datetime(df["week"]).dt.date.astype(str)

    # A) per-series (what your schema expects in `predictions`)
    by_item = (df.groupby(["series_id", "week"], as_index=False)["y_pred"]
                 .sum()
                 .sort_values(["series_id", "week"])
                 .reset_index(drop=True))
    predictions_records = [
        {"series_id": r.series_id, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_item.itertuples(index=False)
    ]

    # B) grouped views
    by_g1 = (df.groupby(["customer_id", "item_group1", "week"], as_index=False)["y_pred"]
               .sum()
               .sort_values(["customer_id", "week", "item_group1"])
               .reset_index(drop=True))
    by_g2 = (df.groupby(["customer_id", "item_group2", "week"], as_index=False)["y_pred"]
               .sum()
               .sort_values(["customer_id", "week", "item_group2"])
               .reset_index(drop=True))
    predictions_by_item_group1 = [
        {"customer_id": r.customer_id, "item_group1": r.item_group1, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_g1.itertuples(index=False)
    ]
    predictions_by_item_group2 = [
        {"customer_id": r.customer_id, "item_group2": r.item_group2, "week": r.week, "y_pred": float(r.y_pred)}
        for r in by_g2.itertuples(index=False)
    ]

    # 7) Top-level counters to satisfy JobResponse
    rows_pred = len(predictions_records)
    series_pred = int(df["series_id"].nunique())

    return JobResponse(
        job_id=job_id,
        rows_pred=rows_pred,
        series_pred=series_pred,
        overall=summary.get("overall", {}),
        predictions=predictions_records,
        metrics_per_individual=summary.get("metrics_per_individual", []),
        metrics_by_group=summary.get("metrics_by_group", []),
        predictions_by_item_group1=predictions_by_item_group1,
        predictions_by_item_group2=predictions_by_item_group2,
    )

@router.post("/forecast/group", response_model=JobResponse)
def forecast_group(payload: ForecastByGroupRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)
    hist_df = _history_to_df(payload.history)
    job_id = uuid.uuid4().hex[:8]

    summary, _ = run_report(
        interval_start=pd.to_datetime(payload.interval.start_date),
        interval_end=pd.to_datetime(payload.interval.end_date),
        predictor_path=payload.predictor_path,
        group_id=payload.group_id,
        history_df=hist_df,
    )
    return JobResponse(
        job_id=job_id,
        rows_pred=summary["rows_pred"],
        series_pred=summary["series_pred"],
        overall=summary["overall"],
        predictions=summary["predictions"],
        metrics_per_individual=summary["metrics_per_individual"],
        metrics_by_group=summary["metrics_by_group"],
    )
