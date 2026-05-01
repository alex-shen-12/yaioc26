from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..config import TARGET_COL
from ..features import pigment_category
from .base import BaseModel


def _fit_group_nnls(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit y = a*log(t+1) + b*t with a,b >= 0 via NNLS. Always uses (t, y) as given."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.log1p(t), t])
    coef, _ = nnls(A, y)
    return coef  # shape (2,)


class CurveFitModel(BaseModel):
    """Per-group curve fit: dietaE = a * log(t+1) + b * t with a,b>=0 (NNLS).
    Empirical-Bayes shrinkage toward (pigment_category, condition) prior.
    """

    name = "curve"

    def __init__(self, shrink_k: float = 3.0, b_cap_quantile: float = 0.95):
        self.shrink_k = shrink_k
        self.b_cap_quantile = b_cap_quantile
        self.group_coef_: dict[str, np.ndarray] = {}
        self.group_n_: dict[str, int] = {}
        self.category_prior_: dict[tuple[str, str], np.ndarray] = {}
        self.global_prior_: np.ndarray = np.array([0.0, 0.0])

    def _group_key(self, sample: str, cond: str) -> str:
        return f"{sample}__{cond}"

    def _cat_key(self, sample: str, cond: str) -> tuple[str, str]:
        return (pigment_category(sample), cond)

    def fit(self, df_train: pd.DataFrame) -> "CurveFitModel":
        df = df_train.copy()
        df["__gkey"] = df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)
        df["__ckey"] = df["sample"].map(pigment_category).astype(str) + "__" + df["aging_condition"].astype(str)

        raw_coefs: dict[str, np.ndarray] = {}
        ns: dict[str, int] = {}
        for key, sub in df.groupby("__gkey"):
            t = sub["aging_time_day"].to_numpy(dtype=float)
            y = sub[TARGET_COL].to_numpy(dtype=float)
            # ensure t=0 anchor exists (add if missing)
            if not np.any(t == 0):
                t = np.concatenate([[0.0], t])
                y = np.concatenate([[0.0], y])
            ns[key] = int(np.sum(sub["aging_time_day"].to_numpy() > 0))  # count non-zero points
            if ns[key] >= 1:
                coef = _fit_group_nnls(t, y)
            else:
                coef = np.array([0.0, 0.0])
            raw_coefs[key] = coef

        # Build category-conditioned prior (mean of coefs across groups in same category+cond)
        cat_coefs: dict[tuple[str, str], list[np.ndarray]] = {}
        for key, coef in raw_coefs.items():
            sample, cond = key.split("__", 1)
            ck = (pigment_category(sample), cond)
            cat_coefs.setdefault(ck, []).append(coef)
        category_prior = {ck: np.mean(np.vstack(cs), axis=0) for ck, cs in cat_coefs.items()}
        global_prior = np.mean(np.vstack(list(raw_coefs.values())), axis=0) if raw_coefs else np.array([0.0, 0.0])

        # Cap b at quantile across well-populated groups (protect extrapolation)
        bs = np.array([c[1] for k, c in raw_coefs.items() if ns[k] >= 3])
        b_cap = float(np.quantile(bs, self.b_cap_quantile)) if len(bs) > 0 else np.inf

        # Shrink
        shrunk: dict[str, np.ndarray] = {}
        for key, coef in raw_coefs.items():
            sample, cond = key.split("__", 1)
            ck = (pigment_category(sample), cond)
            prior = category_prior.get(ck, global_prior)
            n = ns[key]
            w = n / (n + self.shrink_k)
            c = w * coef + (1.0 - w) * prior
            # clamp b
            c[1] = min(c[1], b_cap)
            c = np.maximum(c, 0.0)
            shrunk[key] = c

        self.group_coef_ = shrunk
        self.group_n_ = ns
        self.category_prior_ = category_prior
        self.global_prior_ = global_prior
        return self

    def _predict_one(self, sample: str, cond: str, t: float) -> float:
        key = self._group_key(sample, cond)
        coef = self.group_coef_.get(key)
        if coef is None:
            ck = self._cat_key(sample, cond)
            coef = self.category_prior_.get(ck, self.global_prior_)
        a, b = coef
        y = a * np.log1p(t) + b * t
        return float(max(0.0, y))

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df_pred), dtype=float)
        for i, row in enumerate(df_pred.itertuples(index=False)):
            sample = getattr(row, "sample")
            cond = getattr(row, "aging_condition")
            t = float(getattr(row, "aging_time_day"))
            preds[i] = self._predict_one(sample, cond, t)
        return preds
