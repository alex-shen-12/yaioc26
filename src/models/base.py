from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaseModel:
    """Common interface. Concrete models accept the full training DataFrame
    (with sample, aging_condition, aging_time_day, L0, a0, b0, dietaE) and
    a prediction DataFrame (same columns minus dietaE).
    """

    name: str = "base"

    def fit(self, df_train: pd.DataFrame) -> "BaseModel":
        raise NotImplementedError

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        return joblib.load(path)
