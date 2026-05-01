from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RANDOM_STATE, TARGET_COL
from ..features import build_feature_frame
from .base import BaseModel


class XGBModel(BaseModel):
    name = "xgb"

    def __init__(
        self,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 3.0,
        random_state: int = RANDOM_STATE,
    ):
        try:
            import xgboost as xgb  # noqa
        except ImportError as e:
            raise ImportError("xgboost not installed. pip install xgboost") from e
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None
        self.model_ = None

    def fit(self, df_train: pd.DataFrame) -> "XGBModel":
        import xgboost as xgb
        X, self.levels_ = build_feature_frame(df_train, feature_set="gbm")
        self.feature_columns_ = list(X.columns)
        y = df_train[TARGET_COL].to_numpy()
        n = X.shape[1]
        mono = [0] * n
        mono[0] = 1
        self.model_ = xgb.XGBRegressor(
            **self.params,
            monotone_constraints=tuple(mono),
        )
        self.model_.fit(X.to_numpy(), y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="gbm")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        return self.model_.predict(X.to_numpy())


class LGBMModel(BaseModel):
    name = "lgbm"

    def __init__(
        self,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 15,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_samples: int = 3,
        random_state: int = RANDOM_STATE,
    ):
        try:
            import lightgbm as lgb  # noqa
        except ImportError as e:
            raise ImportError("lightgbm not installed. pip install lightgbm") from e
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None
        self.model_ = None

    def fit(self, df_train: pd.DataFrame) -> "LGBMModel":
        import lightgbm as lgb
        X, self.levels_ = build_feature_frame(df_train, feature_set="gbm")
        self.feature_columns_ = list(X.columns)
        y = df_train[TARGET_COL].to_numpy()
        n = X.shape[1]
        mono = [0] * n
        mono[0] = 1
        self.model_ = lgb.LGBMRegressor(
            **self.params,
            monotone_constraints=mono,
            monotone_constraints_method="intermediate",
        )
        self.model_.fit(X.to_numpy(), y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="gbm")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        return self.model_.predict(X.to_numpy())
