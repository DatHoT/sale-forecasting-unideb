from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import uuid

from src.models.schemas import (
    PredictPairsRequest, ForecastByCustomerRequest, ForecastByGroupRequest, JobResponse
)
from src.core.config import (
    DEFAULT_OUTPUT_DIR, DEFAULT_PREDICTOR_PATH,
    FILL_MISSING_SERIES_WITH_ZERO_HISTORY, MIN_FAKE_HISTORY_WEEKS, FUTURE_WEEKS
)
from src.services.forecasting import run_report, ID_COL, TIME_COL

router = APIRouter()

def _check_interval_weeks(start_date, end_date, max_weeks=12):
    delta_weeks = (end_date - start_date).days // 7 + 1
    if delta_weeks > max_weeks:
        raise HTTPException(status_code=400, detail=f"Interval length {delta_weeks} weeks exceeds max {max_weeks}.")

def _history_to_df(history_rows):
    # Convert list[HistoryRow] → pandas.DataFrame
    df = pd.DataFrame([h.model_dump() for h in history_rows])
    # Normalize series_id if missing but (customer_id,item_id) present
    if "series_id" not in df.columns:
        df["series_id"] = None
    need_sid = df["series_id"].isna()
    if need_sid.any():
        if {"customer_id", "item_id"}.issubset(df.columns):
            df.loc[need_sid, "series_id"] = (
                df.loc[need_sid, "customer_id"].astype(str) + "||" + df.loc[need_sid, "item_id"].astype(str)
            )
        else:
            raise HTTPException(400, "Each history row must include series_id or (customer_id and item_id).")
    # Rename week column to match service constant
    if "week" not in df.columns:
        raise HTTPException(400, "Each history row must include 'week' (ISO date).")
    return df

def _build_req_from_interval(series_ids, start_date, end_date):
    weeks = pd.date_range(start_date, end_date, freq="W-MON")
    return pd.DataFrame([(sid, wk) for sid in series_ids for wk in weeks], columns=[ID_COL, TIME_COL])

@router.post("/predict/pairs", response_model=JobResponse)
def predict_pairs(payload: PredictPairsRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)

    hist_df = _history_to_df(payload.history)
    # Use all series present in provided history
    series_ids = hist_df["series_id"].dropna().unique().tolist()
    if not series_ids:
        raise HTTPException(400, "No series_id found in history.")
    req_df = _build_req_from_interval(series_ids, payload.interval.start_date, payload.interval.end_date)

    job_id = uuid.uuid4().hex[:8]
    out_dir = Path(DEFAULT_OUTPUT_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = payload.predictor_path or DEFAULT_PREDICTOR_PATH

    summary, _ = run_report(
        hist_df=hist_df,
        req_df=req_df,
        predictor_path=predictor_path,
        out_dir=out_dir,
        future_weeks=FUTURE_WEEKS,
        fill_missing_zero=FILL_MISSING_SERIES_WITH_ZERO_HISTORY,
        min_fake_history_weeks=MIN_FAKE_HISTORY_WEEKS,
    )

    return JobResponse(
        job_id=job_id,
        rows_pred=summary["rows_pred"],
        series_pred=summary["series_pred"],
        outputs=summary["artifacts"],
        overall=summary["overall"],
    )

@router.post("/forecast/customer", response_model=JobResponse)
def forecast_customer(payload: ForecastByCustomerRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)

    hist_df = _history_to_df(payload.history)
    if "customer_id" not in hist_df.columns:
        raise HTTPException(400, "History rows must include 'customer_id' for this endpoint.")
    mask = hist_df["customer_id"].astype(str) == str(payload.customer_id)
    series_ids = hist_df.loc[mask, "series_id"].dropna().unique().tolist()
    if not series_ids:
        raise HTTPException(404, f"No series found for customer_id={payload.customer_id} in provided history.")
    req_df = _build_req_from_interval(series_ids, payload.interval.start_date, payload.interval.end_date)

    job_id = uuid.uuid4().hex[:8]
    out_dir = Path(DEFAULT_OUTPUT_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = payload.predictor_path or DEFAULT_PREDICTOR_PATH

    summary, _ = run_report(
        hist_df=hist_df,
        req_df=req_df,
        predictor_path=predictor_path,
        out_dir=out_dir,
        future_weeks=FUTURE_WEEKS,
        fill_missing_zero=FILL_MISSING_SERIES_WITH_ZERO_HISTORY,
        min_fake_history_weeks=MIN_FAKE_HISTORY_WEEKS,
    )

    return JobResponse(
        job_id=job_id,
        rows_pred=summary["rows_pred"],
        series_pred=summary["series_pred"],
        outputs=summary["artifacts"],
        overall=summary["overall"],
    )

@router.post("/forecast/group", response_model=JobResponse)
def forecast_group(payload: ForecastByGroupRequest):
    _check_interval_weeks(payload.interval.start_date, payload.interval.end_date, 12)

    hist_df = _history_to_df(payload.history)
    if "group_id" not in hist_df.columns:
        raise HTTPException(400, "History rows must include 'group_id' for this endpoint.")
    mask = hist_df["group_id"].astype(str) == str(payload.group_id)
    series_ids = hist_df.loc[mask, "series_id"].dropna().unique().tolist()
    if not series_ids:
        raise HTTPException(404, f"No series found for group_id={payload.group_id} in provided history.")
    req_df = _build_req_from_interval(series_ids, payload.interval.start_date, payload.interval.end_date)

    job_id = uuid.uuid4().hex[:8]
    out_dir = Path(DEFAULT_OUTPUT_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = payload.predictor_path or DEFAULT_PREDICTOR_PATH

    summary, _ = run_report(
        hist_df=hist_df,
        req_df=req_df,
        predictor_path=predictor_path,
        out_dir=out_dir,
        future_weeks=FUTURE_WEEKS,
        fill_missing_zero=FILL_MISSING_SERIES_WITH_ZERO_HISTORY,
        min_fake_history_weeks=MIN_FAKE_HISTORY_WEEKS,
    )

    return JobResponse(
        job_id=job_id,
        rows_pred=summary["rows_pred"],
        series_pred=summary["series_pred"],
        outputs=summary["artifacts"],
        overall=summary["overall"],
    )
