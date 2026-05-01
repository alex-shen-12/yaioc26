from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUP_COLS, TARGET_COL, TEST_CSV, TRAIN_CSV


def load_train(path: Path = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    return df


def load_test(path: Path = TEST_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(axis=1, how="all")
    return df


def add_group_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_key"] = df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)
    return df


def train_group_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    df = add_group_key(df_train)
    agg = df.groupby("group_key").agg(
        sample=("sample", "first"),
        aging_condition=("aging_condition", "first"),
        n_points=("aging_time_day", "size"),
        t_min=("aging_time_day", "min"),
        t_max=("aging_time_day", "max"),
        y_min=(TARGET_COL, "min"),
        y_max=(TARGET_COL, "max"),
    ).reset_index()
    return agg.sort_values(["aging_condition", "sample"]).reset_index(drop=True)


def check_test_coverage(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    tr = add_group_key(df_train)
    te = add_group_key(df_test)
    tr_keys = set(tr["group_key"].unique())
    rows = []
    for key, sub in te.groupby("group_key"):
        tr_sub = tr[tr["group_key"] == key]
        t_max_train = int(tr_sub["aging_time_day"].max()) if len(tr_sub) else -1
        t_test_max = int(sub["aging_time_day"].max())
        rows.append({
            "group_key": key,
            "in_train": key in tr_keys,
            "n_train_rows": len(tr_sub),
            "t_max_train": t_max_train,
            "t_test_list": sorted(sub["aging_time_day"].unique().tolist()),
            "extrapolation_gap": t_test_max - t_max_train if t_max_train > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("n_train_rows").reset_index(drop=True)


def assert_t0_is_zero(df_train: pd.DataFrame) -> None:
    zero_rows = df_train[df_train["aging_time_day"] == 0]
    assert len(zero_rows) > 0, "No t=0 rows found in training data"
    assert np.allclose(zero_rows[TARGET_COL].to_numpy(), 0.0), "Some t=0 rows have dietaE != 0"


def write_submission(preds: np.ndarray, out_path: Path) -> None:
    preds = np.asarray(preds).reshape(-1)
    pd.DataFrame({TARGET_COL: preds}).to_csv(out_path, index=False)
