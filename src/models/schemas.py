from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List, Dict

# ---- Input primitives ----

class Interval(BaseModel):
    start_date: date
    end_date: date

class HistoryRow(BaseModel):
    # One weekly record; either provide `series_id` or both (customer_id, item_id)
    week: date
    sales: float = Field(ge=0)
    series_id: Optional[str] = None

    # Optional identifiers (help us build series_id if missing and filter by customer/group)
    customer_id: Optional[str] = None
    item_id: Optional[str] = None
    group_id: Optional[str] = None
    item_group1: Optional[str] = None
    item_group2: Optional[str] = None

    # Optional weekly exogenous (if you trained with these, they’ll be used from history)
    temperature: Optional[float] = None
    num_sunny_days: Optional[float] = None
    holiday_count: Optional[float] = None

# ---- Requests ----

class PredictPairsRequest(BaseModel):
    interval: Interval
    predictor_path: Optional[str] = None
    # The complete 12w pre-history (and older is fine) for all series you want forecasted
    history: List[HistoryRow]

class ForecastByCustomerRequest(BaseModel):
    interval: Interval
    customer_id: str
    history: List[HistoryRow]
    predictor_path: Optional[str] = None

class ForecastByGroupRequest(BaseModel):
    interval: Interval
    group_id: str
    history: List[HistoryRow]
    predictor_path: Optional[str] = None

# ---- Response ----

class JobResponse(BaseModel):
    job_id: str
    rows_pred: int
    series_pred: int
    outputs: Dict[str, str]  # {"preds_csv": "...", "metrics_xlsx": "..."}
    overall: Dict[str, float]
