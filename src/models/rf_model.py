from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import RANDOM_STATE, TARGET_COL
from ..features import build_feature_frame
from .base import BaseModel


class RFModel(BaseModel):
    name = "rf"

    def __init__(
        self,
        n_estimators: int = 800,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        max_features: str | float = "sqrt",
        random_state: int = RANDOM_STATE,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None

    def fit(self, df_train: pd.DataFrame) -> "RFModel":
        X, self.levels_ = build_feature_frame(df_train, feature_set="rf")
        self.feature_columns_ = list(X.columns)
        y = df_train[TARGET_COL].to_numpy()
        self.rf.fit(X.to_numpy(), y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        assert self.levels_ is not None
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="rf")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        return self.rf.predict(X.to_numpy())


class RFBaselineModel(BaseModel):
    """Replicates the organizer's baseline (only 4 numeric features, no cats)."""

    name = "rf_baseline"

    def __init__(self, random_state: int = RANDOM_STATE):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[["aging_time_day", "L0", "a0", "b0"]].to_numpy()

    def fit(self, df_train: pd.DataFrame) -> "RFBaselineModel":
        X = self._x(df_train)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        y = df_train[TARGET_COL].to_numpy()
        self.rf.fit(X, y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        X = self._x(df_pred)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.rf.predict(X)
