from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..config import TARGET_COL
from ..features import build_feature_frame
from .base import BaseModel


class RidgeCVModel(BaseModel):
    name = "ridge"

    def __init__(self):
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None
        self.scaler = StandardScaler(with_mean=False)
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 25))

    def fit(self, df_train: pd.DataFrame) -> "RidgeCVModel":
        X, self.levels_ = build_feature_frame(df_train, feature_set="linear")
        self.feature_columns_ = list(X.columns)
        Xs = self.scaler.fit_transform(X.to_numpy())
        y = df_train[TARGET_COL].to_numpy()
        self.model.fit(Xs, y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="linear")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        Xs = self.scaler.transform(X.to_numpy())
        return self.model.predict(Xs)


class PolyModel(BaseModel):
    name = "poly"

    def __init__(self, degree: int = 2):
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None
        # Use a compact numeric-only subset for poly; stack one-hots afterward.
        self.numeric_cols = [
            "aging_time_day", "log1p_t", "sqrt_t", "L0_norm", "C0", "km_feature", "cond_is_uv"
        ]
        self.pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    def fit(self, df_train: pd.DataFrame) -> "PolyModel":
        X, self.levels_ = build_feature_frame(df_train, feature_set="linear")
        self.feature_columns_ = list(X.columns)
        Xn = X[self.numeric_cols].to_numpy()
        y = df_train[TARGET_COL].to_numpy()
        self.pipe.fit(Xn, y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="linear")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        Xn = X[self.numeric_cols].to_numpy()
        return self.pipe.predict(Xn)
