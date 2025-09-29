from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List, Dict, Any

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
    history: Optional[List[HistoryRow]] = None  # <-- optional
    export_excel: Optional[bool] = False  # default off

class ForecastByCustomerRequest(BaseModel):
    interval: Interval
    customer_id: str
    predictor_path: Optional[str] = None
    history: Optional[List[HistoryRow]] = None  # <-- optional

class ForecastByGroupRequest(BaseModel):
    interval: Interval
    group_id: str
    predictor_path: Optional[str] = None
    history: Optional[List[HistoryRow]] = None  # <-- optional

class CustomerGroup1Row(BaseModel):
    customer_id: str
    item_group1: str
    week: str
    y_pred: float

class CustomerGroup2Row(BaseModel):
    customer_id: str
    item_group2: str
    week: str
    y_pred: float

# ---- Response ----

class JobResponse(BaseModel):
    job_id: str
    rows_pred: int
    series_pred: int
    overall: Dict[str, Optional[float]]
    predictions: List[dict]  # or your existing PredictionRow model

    metrics_per_individual: List[dict] = []  # keep your concrete type if you had one
    metrics_by_group: List[dict] = []

    # NEW (optional) aggregated views
    predictions_by_item_group1: Optional[List[CustomerGroup1Row]] = None
    predictions_by_item_group2: Optional[List[CustomerGroup2Row]] = None