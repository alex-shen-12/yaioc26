from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, TARGET_COL
from ..features import build_feature_frame
from .base import BaseModel


class GPRModel(BaseModel):
    name = "gpr"

    def __init__(self, random_state: int = RANDOM_STATE):
        self.scaler = StandardScaler()
        self.levels_: dict | None = None
        self.feature_columns_: list[str] | None = None
        self.model_: GaussianProcessRegressor | None = None
        self.random_state = random_state

    def fit(self, df_train: pd.DataFrame) -> "GPRModel":
        X, self.levels_ = build_feature_frame(df_train, feature_set="gpr")
        self.feature_columns_ = list(X.columns)
        Xs = self.scaler.fit_transform(X.to_numpy())
        y = df_train[TARGET_COL].to_numpy()
        n_dim = Xs.shape[1]
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=np.ones(n_dim) * 1.0, length_scale_bounds=(1e-1, 1e2)
        ) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 10.0))
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=self.random_state,
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X, _ = build_feature_frame(df_pred, fit_levels=self.levels_, feature_set="gpr")
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)
        Xs = self.scaler.transform(X.to_numpy())
        return self.model_.predict(Xs)
