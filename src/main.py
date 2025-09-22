from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.v1 import forecast, health
from src.core.config import DEFAULT_OUTPUT_DIR

app = FastAPI(title="Sales Forecasting Service", version="1.0")

# Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")

# Serve artifacts
app.mount("/outputs", StaticFiles(directory=str(DEFAULT_OUTPUT_DIR), html=True), name="outputs")
