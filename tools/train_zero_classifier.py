# tools/train_zero_classifier.py
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from src.core.config import PREPATCHED_PAD_PATH

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | zero_cls | %(message)s"
)
log = logging.getLogger("zero_cls")

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_recall_curve, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

# LightGBM
import lightgbm as lgb

import time
# CatBoost (optional)
try:
    from catboost import CatBoostClassifier, Pool as CatPool
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    log.warning("CatBoost not available; will skip CatBoost base model.")

OUT_DIR = Path("models/zero_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Utilities
# ==============================

def _dstr(x) -> str:
    """Format numpy.datetime64 / pandas.Timestamp / datetime to YYYY-MM-DD string safely."""
    try:
        return pd.Timestamp(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)

class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        log.info(f"[{self.name}] start")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        log.info(f"[{self.name}] done in {dt:.2f}s")

def _ensure_datetime_week(df: pd.DataFrame, col: str = "week") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

def _split_series_id(df: pd.DataFrame) -> pd.DataFrame:
    # series_id format: partner_id||itemcode
    s = df["series_id"].astype(str).str.split("||", n=1, expand=True)
    df["partner_id"] = s[0].astype("category")
    df["itemcode"] = s[1].astype("category")
    return df

def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    # weekofyear already present (1..53). Normalize to 1..52
    woy = df["weekofyear"].clip(lower=1, upper=52)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    return df

def _group_shift(df: pd.DataFrame, by: List[str], col: str, shift: int) -> pd.Series:
    return df.groupby(by, observed=True)[col].shift(shift)

def _group_roll_sum(df: pd.DataFrame, by: List[str], col: str, window: int) -> pd.Series:
    return (
        df.groupby(by, observed=True)[col]
          .apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
          .reset_index(level=by, drop=True)
    )

def _time_since_last_nonzero(df: pd.DataFrame, by: List[str], y_col: str) -> pd.Series:
    # Compute distance (in weeks) since last y==1, per series
    def per_group(s: pd.Series) -> pd.Series:
        # s is binary series aligned to time. Shift to avoid peeking at current week.
        shifted = s.shift(1).fillna(0).to_numpy()
        out = np.zeros_like(shifted, dtype=np.int32)
        last = -1
        for i, v in enumerate(shifted):
            if last == -1:
                out[i] = 10_000  # big number for "never seen"
            else:
                out[i] = i - last
            if v > 0.5:
                last = i
        return pd.Series(out, index=s.index)
    return df.groupby(by, observed=True)[y_col].apply(per_group).reset_index(level=by, drop=True)

def _expanding_rate(df: pd.DataFrame, by: List[str], y_col: str) -> pd.Series:
    # Expanding mean of past activity, shifted to avoid leakage
    return (
        df.groupby(by, observed=True)[y_col]
          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
          .reset_index(level=by, drop=True)
    )

def _make_binary_label(df: pd.DataFrame) -> pd.Series:
    return (df["sales"].fillna(0) > 0).astype(np.uint8)

def _infer_cutoff_for_validation(df: pd.DataFrame, val_weeks: int = 12) -> pd.Timestamp:
    max_w = df["week"].max()
    cutoff = (max_w - pd.to_timedelta(val_weeks, unit="W")).normalize()
    # Align to Monday for W-MON; keep it simple: choose the exact date in data
    # Use the latest date that is >= cutoff and < max_w
    candidates = df["week"].sort_values().unique()
    cutoff = candidates[max(0, len(candidates) - val_weeks - 1)]
    return pd.Timestamp(cutoff)

@dataclass
class SplitCfg:
    val_weeks: int = 12
    cv_folds: int = 3  # rolling origins, left to right

@dataclass
class TrainCfg:
    lgbm_params: Dict
    cat_params: Optional[Dict]
    use_catboost: bool
    random_state: int = 7

DEFAULT_TRAIN_CFG = TrainCfg(
    lgbm_params=dict(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=128,
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        n_jobs=-1,
    ),
    cat_params=dict(
        loss_function="Logloss",
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        iterations=3000,
        random_seed=7,
        verbose=False,
        od_type="Iter",
        od_wait=200,
    ) if HAS_CATBOOST else None,
    use_catboost=HAS_CATBOOST,
)

# ==============================
# Feature engineering
# ==============================

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    lg = logging.getLogger("zero_cls")

    with StepTimer("features.copy"):
        df = raw.copy()

    with StepTimer("features.ensure_datetime_week"):
        df = _ensure_datetime_week(df, "week")

    with StepTimer("features.split_series_id"):
        df = _split_series_id(df)

    with StepTimer("features.seasonality"):
        df = _add_seasonal_features(df)

    with StepTimer("features.label"):
        df["y"] = _make_binary_label(df)
        lg.info(f"[label] positives={int(df['y'].sum())} / n={len(df)} (rate={df['y'].mean():.4f})")

    with StepTimer("features.sort"):
        df = df.sort_values(["series_id", "week"]).reset_index(drop=True)

    with StepTimer("features.binary_lags"):
        for k in range(1, 13):
            df[f"y_lag_{k}"] = _group_shift(df, ["series_id"], "y", k).fillna(0).astype(np.uint8)
        lg.info("[binary_lags] added y_lag_1..12")

    with StepTimer("features.y_rollsum"):
        for w in (4, 8, 12):
            # keep float/NaN; models handle missing; we drop early series rows later anyway
            df[f"y_rollsum_{w}"] = _group_roll_sum(df, ["series_id"], "y", w)
        lg.info("[y_rollsum] added y_rollsum_4/8/12")

    with StepTimer("features.recency"):
        df["weeks_since_last_nonzero"] = _time_since_last_nonzero(df, ["series_id"], "y")

    with StepTimer("features.sales_lags"):
        for k in (1, 2, 4, 8, 12):
            df[f"sales_lag_{k}"] = _group_shift(df, ["series_id"], "sales", k).fillna(0)
        lg.info("[sales_lags] added sales_lag_1/2/4/8/12")

    with StepTimer("features.weather_lags_ma"):
        for col in ["temperature", "num_sunny_days", "holiday_count"]:
            df[f"{col}_lag_1"] = _group_shift(df, ["series_id"], col, 1)
            df[f"{col}_ma_4"] = (
                df.groupby("series_id", observed=True)[col]
                  .apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
                  .reset_index(level="series_id", drop=True)
            )
        lg.info("[weather] added *_lag_1 and *_ma_4 for temperature/num_sunny_days/holiday_count")

    with StepTimer("features.expanding_rates"):
        # Always time-order within the grouping key to avoid leakage across series of the same partner/group
        df["exp_rate_series"] = (
            df.groupby("series_id", observed=True)["y"]
              .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
              .reset_index(level="series_id", drop=True)
        )
    
        # Partner-level
        df["_orig_idx"] = np.arange(len(df))
        tmp = df.sort_values(["partner_id", "week"]).copy()
        tmp["exp_rate_partner"] = (
            tmp.groupby("partner_id", observed=True)["y"]
               .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
               .reset_index(level="partner_id", drop=True)
        )
        df.loc[tmp["_orig_idx"].values, "exp_rate_partner"] = tmp["exp_rate_partner"].values
        del tmp
    
        # Itemcode-level
        tmp = df.sort_values(["itemcode", "week"]).copy()
        tmp["exp_rate_itemcode"] = (
            tmp.groupby("itemcode", observed=True)["y"]
               .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
               .reset_index(level="itemcode", drop=True)
        )
        df.loc[tmp["_orig_idx"].values, "exp_rate_itemcode"] = tmp["exp_rate_itemcode"].values
        del tmp
    
        # Group-level
        tmp = df.sort_values(["group_id", "week"]).copy()
        tmp["exp_rate_group"] = (
            tmp.groupby("group_id", observed=True)["y"]
               .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
               .reset_index(level="group_id", drop=True)
        )
        df.loc[tmp["_orig_idx"].values, "exp_rate_group"] = tmp["exp_rate_group"].values
        del tmp
        df = df.drop(columns=["_orig_idx"])

    with StepTimer("features.astype_cats"):
        for c in ["series_id", "partner_id", "itemcode", "group_id", "item_group1", "item_group2", "month", "year"]:
            df[c] = df[c].astype("category")

    # Drop first 12 rows per series (lags need context). This naturally removes most engineered NaNs.
    with StepTimer("features.drop_early_12"):
        df["min_idx_in_series"] = df.groupby("series_id", observed=True).cumcount()
        before = len(df)
        df = df[df["min_idx_in_series"] >= 12].drop(columns=["min_idx_in_series"])
        lg.info(f"[drop_early_12] kept {len(df)}/{before} rows")

    # Report any remaining NaNs per engineered column & show unique affected weeks (for audit only)
    with StepTimer("features.nan_audit_after_drop"):
        eng_cols = [c for c in df.columns if c not in ["series_id","partner_id","itemcode","group_id",
                                                        "item_group1","item_group2","month","year","week","sales","y"]]
        nan_cols = [c for c in eng_cols if df[c].isna().any()]
        if nan_cols:
            lg.warning(f"[nan_audit] columns with NaN after drop: {nan_cols}")
            # unique weeks containing NaN across any numeric feature
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            bad_mask = ~np.isfinite(df[num_cols].to_numpy()).all(axis=1)
            uw = sorted(pd.to_datetime(df.loc[bad_mask, "week"]).unique())
            if uw:
                lg.warning(f"[nan_audit] unique weeks with non-finite values ({len(uw)}): " +
                           ", ".join(_dstr(w) for w in uw[:50]) + (" …" if len(uw) > 50 else ""))
        else:
            lg.info("[nan_audit] no NaN in engineered columns after drop")

    return df

# ==============================
# Splitting
# ==============================

def make_splits(df: pd.DataFrame, cfg: SplitCfg) -> Tuple[pd.Timestamp, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Returns:
      - cutoff timestamp for final validation start
      - list of (train_end, val_end) tuples for rolling CV folds (val_length = cfg.val_weeks)
    """
    cutoff = _infer_cutoff_for_validation(df, cfg.val_weeks)
    # Build folds ending before cutoff
    weeks = np.sort(df["week"].unique())
    val_len = cfg.val_weeks
    folds = []
    # Create 3 folds right before cutoff, each shifted by val_len
    # Example: [..., t-36:t-24], [..., t-24:t-12], final val is [t-12:t)
    end_idx = np.where(weeks == cutoff)[0][0]  # validation start index
    for i in range(cfg.cv_folds, 0, -1):
        val_start = weeks[end_idx - i * val_len]
        val_end = weeks[end_idx - (i - 1) * val_len]
        folds.append((val_start, val_end))
    return cutoff, folds

def slice_by_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["week"] >= start) & (df["week"] < end)]

# ==============================
# Model builders
# ==============================

def build_lgbm(params: Dict, scale_pos_weight: float) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(**params)
    m.set_params(scale_pos_weight=scale_pos_weight)
    return m

def build_catboost(params: Dict, class_weights: Tuple[float, float]) -> CatBoostClassifier:
    m = CatBoostClassifier(**params)
    m.set_params(class_weights=class_weights)
    return m

# ==============================
# Training & Stacking
# ==============================

def train_and_evaluate():
    # 1) Load raw
    hist_path = Path(PREPATCHED_PAD_PATH)
    log.info(f"Loading history from {hist_path}")
    raw = pd.read_parquet(hist_path)
    # raw = pd.read_parquet(hist_path).merge(pd.read_parquet(hist_path)[["series_id"]].drop_duplicates().sample(frac=0.001, random_state=7), on="series_id")

    # 2) Build features
    log.info("Building features…")
    df = build_features(raw)
    scale_pos_weight = float((1 - df["y"].mean()) / max(1e-6, df["y"].mean()))

    # Columns
    label_col = "y"
    time_col = "week"
    cat_cols = ["series_id", "partner_id", "itemcode", "group_id", "item_group1", "item_group2", "month", "year"]
    # num_cols = [c for c in df.columns if c not in cat_cols + [label_col, time_col]]
    num_cols = [c for c in df.columns if c not in cat_cols + [label_col, time_col, "sales"]]


    # 3) Splits
    split_cfg = SplitCfg(val_weeks=12, cv_folds=3)
    cutoff, folds = make_splits(df, split_cfg)
    log.info(f"Validation (final) starts at: {cutoff}  (last {split_cfg.val_weeks} weeks)")

    # Train base models on CV folds to get OOF predictions
    oof_meta = []
    oof_y = []

    # Prepare encoders/pipelines for LightGBM (handle categoricals via one-hot or native category indices)
    # LightGBM can handle pandas categorical directly. We’ll pass raw matrices.
    # We still one-hot low-cardinality cols to add interaction capacity.
    # For simplicity and speed, we won’t one-hot here; LGBM handles categoricals.

    # Class imbalance
    pos_rate = df[label_col].mean()
    scale_pos_weight = (1 - pos_rate) / max(1e-6, pos_rate)
    log.info(f"Class balance: pos_rate={pos_rate:.4f} → scale_pos_weight≈{scale_pos_weight:.2f}")
    # Initialize containers for stacking
    oof_meta, oof_y = [], []
    base_preds = {"lgbm": [], "cat": []}
    base_models = {"lgbm": [], "cat": []}

    # 3a) Cross-validated OOF for stacking
    for i, (val_start, val_end) in enumerate(folds, 1):
        train_df = df[df[time_col] < val_start]
        valid_df = slice_by_range(df, val_start, val_end)

        X_tr = train_df[cat_cols + num_cols]
        y_tr = train_df[label_col].values
        X_va = valid_df[cat_cols + num_cols]
        y_va = valid_df[label_col].values

        # LightGBM (categoricals)
        lgbm = build_lgbm(DEFAULT_TRAIN_CFG.lgbm_params, scale_pos_weight)
        lgbm.set_params(
            categorical_feature=cat_cols,
        )
        log.info(f"[Fold {i}] Train LightGBM …")
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=100)]
        )
        p_lgbm = lgbm.predict_proba(X_va)[:, 1]
        base_preds["lgbm"].append(p_lgbm)
        base_models["lgbm"].append(lgbm)

        # CatBoost (optional)
        if DEFAULT_TRAIN_CFG.use_catboost:
            pos_w = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
            neg_w = 1.0
            cb = build_catboost(DEFAULT_TRAIN_CFG.cat_params, class_weights=(neg_w, pos_w))
            # CatBoost needs Pool to supply categorical feature indices
            cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
            log.info(f"[Fold {i}] Train CatBoost …")
            cb.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_idx,
                verbose=False
            )
            p_cat = cb.predict_proba(X_va)[:, 1]
            base_preds["cat"].append(p_cat)
            base_models["cat"].append(cb)

        # Collect for meta
        stack_cols = [p_lgbm]
        if DEFAULT_TRAIN_CFG.use_catboost:
            stack_cols.append(p_cat)
        P = np.vstack(stack_cols).T  # shape (n_val, n_models)
        oof_meta.append(P)
        oof_y.append(y_va)

        # Fold metrics
        ap = average_precision_score(y_va, P.mean(axis=1))
        auc = roc_auc_score(y_va, P.mean(axis=1))
        log.info(f"[Fold {i}] mean-prob PR-AUC={ap:.4f} ROC-AUC={auc:.4f}")

    oof_meta = np.vstack(oof_meta)
    oof_y = np.concatenate(oof_y)

    # 3b) Meta-learner (logistic regression)
    log.info("Fit meta-learner on OOF probabilities …")
    meta = LogisticRegression(max_iter=1000)
    meta.fit(oof_meta, oof_y)

    # 4) Train base models on all data before final validation
    train_all = df[df[time_col] < cutoff]
    valid_final = df[df[time_col] >= cutoff]

    X_tr_all = train_all[cat_cols + num_cols]
    y_tr_all = train_all[label_col].values
    X_va_fin = valid_final[cat_cols + num_cols]
    y_va_fin = valid_final[label_col].values

    # Refit LightGBM
    with StepTimer("final.lgbm.fit"):
        lgbm_final = build_lgbm(DEFAULT_TRAIN_CFG.lgbm_params, scale_pos_weight)
        lgbm_final.fit(
            X_tr_all, y_tr_all,
            eval_set=[(X_va_fin, y_va_fin)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=100)]
        )
        if hasattr(lgbm_final, "best_score_"):
            try:
                auc_best = list(lgbm_final.best_score_["valid_0"].values())[0]
                log.info(f"[final.lgbm] best_iter={lgbm_final.best_iteration_} best_valid_auc={auc_best:.4f}")
            except Exception:
                pass
    
    cb_final = None
    if DEFAULT_TRAIN_CFG.use_catboost:
        with StepTimer("final.cat.fit"):
            pos_w = (len(y_tr_all) - y_tr_all.sum()) / max(1, y_tr_all.sum())
            cb_final = build_catboost(DEFAULT_TRAIN_CFG.cat_params, class_weights=(1.0, float(pos_w)))
            cat_idx = [X_tr_all.columns.get_loc(c) for c in cat_cols]
            cb_final.set_params(verbose=200)
            cb_final.fit(
                X_tr_all, y_tr_all,
                eval_set=(X_va_fin, y_va_fin),
                cat_features=cat_idx,
            )

    # 5) Stacked predictions on final validation
    with StepTimer("final.stack_predict"):
        p_lgbm_fin = lgbm_final.predict_proba(X_va_fin)[:, 1]
        stack_fin = [p_lgbm_fin]
        if DEFAULT_TRAIN_CFG.use_catboost and cb_final is not None:
            p_cat_fin = cb_final.predict_proba(X_va_fin)[:, 1]
            stack_fin.append(p_cat_fin)
        P_fin = np.vstack(stack_fin).T
        p_meta_fin = meta.predict_proba(P_fin)[:, 1]
    
    with StepTimer("final.threshold_tune"):
        prec, rec, thr = precision_recall_curve(y_va_fin, p_meta_fin)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        best_idx = np.nanargmax(f1s)
        best_thr = 0.5 if best_idx >= len(thr) else thr[best_idx]
        best_f1 = f1s[best_idx]
        ap = average_precision_score(y_va_fin, p_meta_fin)
        auc = roc_auc_score(y_va_fin, p_meta_fin)
        log.info(f"[FINAL] PR-AUC={ap:.4f} ROC-AUC={auc:.4f}  F1*={best_f1:.4f} @ thr={best_thr:.4f}")
        y_pred = (p_meta_fin >= best_thr).astype(int)
        log.info("[FINAL] Classification report:\n" + classification_report(y_va_fin, y_pred, digits=3))
    
    with StepTimer("persist.artifacts"):
        import joblib
        artifacts = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "cutoff": cutoff,
            "val_weeks": 12,
            "lgbm": lgbm_final,
            "catboost": cb_final if DEFAULT_TRAIN_CFG.use_catboost else None,
            "meta": meta,
            "best_threshold": float(best_thr),
            "train_cfg": asdict(DEFAULT_TRAIN_CFG),
        }
        out_path = OUT_DIR / "zero_classifier_stack.joblib"
        joblib.dump(artifacts, out_path)
        log.info(f"Saved model artifacts → {out_path}")
    
    with StepTimer("persist.validation_csv"):
        val_preds_df = valid_final[["series_id", "week", "sales"]].copy()
        val_preds_df["y_true"] = y_va_fin
        val_preds_df["p_lgbm"] = p_lgbm_fin
        if DEFAULT_TRAIN_CFG.use_catboost and cb_final is not None:
            val_preds_df["p_cat"] = p_cat_fin
        val_preds_df["p_meta"] = p_meta_fin
        val_preds_df["y_pred"] = y_pred
        csv_path = OUT_DIR / "validation_predictions.csv"
        val_preds_df.to_csv(csv_path, index=False)
        log.info(f"Saved validation predictions → {csv_path}")

if __name__ == "__main__":
    train_and_evaluate()
