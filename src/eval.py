from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import CV_FOLDS, RANDOM_STATE, TARGET_COL
from .constraints import apply_all
from .features import pigment_category


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def score_100(rmse_value: float, rmse_worst: float = 10.0) -> float:
    return max(0.0, 100.0 * (1.0 - rmse_value / rmse_worst))


@dataclass
class CVResult:
    model_name: str
    scheme: str
    rmse: float
    score_100: float
    mae: float
    fit_seconds: float
    n_folds: int
    per_fold_rmse: list[float]


ModelFactory = Callable[[], "object"]  # returns a fresh BaseModel


def _fit_predict(model_factory: ModelFactory, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                 *, apply_constraints: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model = model_factory()
    model.fit(train_df)
    preds = model.predict(valid_df)
    y_true = valid_df[TARGET_COL].to_numpy()
    if apply_constraints:
        tm = train_df[["sample", "aging_condition", "aging_time_day"]].reset_index(drop=True)
        y_tr = train_df[TARGET_COL].to_numpy()
        preds = apply_all(tm, y_tr, valid_df[["sample", "aging_condition", "aging_time_day"]], preds)
    return y_true, preds


def random_kfold_cv(
    model_factory: ModelFactory,
    df_train: pd.DataFrame,
    n_splits: int = CV_FOLDS,
    seed: int = RANDOM_STATE,
    *,
    model_name: str = "model",
    apply_constraints: bool = True,
) -> CVResult:
    df = df_train.reset_index(drop=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold = []
    all_true, all_pred = [], []
    t0 = time.perf_counter()
    for tr_idx, va_idx in kf.split(df):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        yt, yp = _fit_predict(model_factory, tr, va, apply_constraints=apply_constraints)
        per_fold.append(rmse(yt, yp))
        all_true.append(yt); all_pred.append(yp)
    elapsed = time.perf_counter() - t0
    yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
    r = rmse(yt, yp)
    return CVResult(model_name, "random_kfold", r, score_100(r), mae(yt, yp), elapsed, n_splits, per_fold)


def time_holdout_cv(
    model_factory: ModelFactory,
    df_train: pd.DataFrame,
    *,
    model_name: str = "model",
    apply_constraints: bool = True,
) -> CVResult:
    """Hold out the maximum aging_time_day per (sample, aging_condition) group
    (excluding t=0). Train once on the rest, predict the held-out rows.
    """
    df = df_train.reset_index(drop=True).copy()
    df["__key"] = df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)

    holdout_idx = []
    for key, sub in df.groupby("__key"):
        sub_nonzero = sub[sub["aging_time_day"] > 0]
        if len(sub_nonzero) < 2:
            continue
        # pick the row with maximum aging_time_day
        pick = sub_nonzero.loc[sub_nonzero["aging_time_day"].idxmax()]
        holdout_idx.append(pick.name)
    holdout_idx = sorted(set(holdout_idx))

    va = df.loc[holdout_idx].drop(columns="__key").reset_index(drop=True)
    tr = df.drop(index=holdout_idx).drop(columns="__key").reset_index(drop=True)
    t0 = time.perf_counter()
    yt, yp = _fit_predict(model_factory, tr, va, apply_constraints=apply_constraints)
    elapsed = time.perf_counter() - t0
    r = rmse(yt, yp)
    return CVResult(model_name, "time_holdout", r, score_100(r), mae(yt, yp), elapsed, 1, [r])


def per_group_error(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = meta.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["pigment_category"] = df["sample"].map(pigment_category)

    def _agg(sub: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(sub),
            "rmse": rmse(sub["y_true"], sub["y_pred"]),
            "mae": mae(sub["y_true"], sub["y_pred"]),
        })

    out = df.groupby(["pigment_category", "aging_condition"]).apply(_agg).reset_index()
    return out


def results_to_df(results: list[CVResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "cv_scheme": r.scheme,
            "rmse": r.rmse,
            "score_100": r.score_100,
            "mae": r.mae,
            "fit_seconds": r.fit_seconds,
            "n_folds": r.n_folds,
        })
    return pd.DataFrame(rows)
