from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import TAU_VALUES


PIGMENT_CATEGORIES = [
    "植物染料",
    "皮纸",
    "矿物颜料",
    "中国画-曙红",
    "中国画-翡翠绿",
    "中国画-钴蓝",
    "中国画-其他",
    "颜彩",
    "其他",
]


def pigment_category(sample: str) -> str:
    s = str(sample)
    if "染料" in s:
        return "植物染料"
    if s.startswith("皮纸"):
        return "皮纸"
    if "矿物颜料" in s:
        return "矿物颜料"
    if s.startswith("中国画-曙红"):
        return "中国画-曙红"
    if s.startswith("中国画-翡翠绿"):
        return "中国画-翡翠绿"
    if s.startswith("中国画-钴蓝"):
        return "中国画-钴蓝"
    if s.startswith("中国画-"):
        return "中国画-其他"
    if s.startswith("颜彩"):
        return "颜彩"
    return "其他"


def _time_features(t: np.ndarray) -> dict[str, np.ndarray]:
    t = np.asarray(t, dtype=float)
    feats = {
        "t": t,
        "log1p_t": np.log1p(t),
        "sqrt_t": np.sqrt(t),
        "t_sq": t * t,
        "is_t_zero": (t == 0).astype(float),
    }
    for tau in TAU_VALUES:
        feats[f"exp_neg_t_over_tau_{int(tau)}"] = np.exp(-t / tau)
    return feats


def _color_features(L0: np.ndarray, a0: np.ndarray, b0: np.ndarray) -> dict[str, np.ndarray]:
    L0 = np.asarray(L0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    C0 = np.sqrt(a0 ** 2 + b0 ** 2)
    h0 = np.arctan2(b0, a0)
    L0n = np.clip(L0 / 100.0, 1e-6, 1.0 - 1e-6)
    km = ((1.0 - L0n) ** 2) / (2.0 * L0n)
    return {
        "L0": L0,
        "a0": a0,
        "b0": b0,
        "C0": C0,
        "sin_h0": np.sin(h0),
        "cos_h0": np.cos(h0),
        "L0_norm": L0n,
        "km_feature": km,
    }


def _align_one_hot(series: pd.Series, levels: list[str], prefix: str) -> pd.DataFrame:
    cats = pd.Categorical(series.astype(str), categories=levels)
    dummies = pd.get_dummies(cats, prefix=prefix, dtype=float)
    for lvl in levels:
        col = f"{prefix}_{lvl}"
        if col not in dummies.columns:
            dummies[col] = 0.0
    return dummies[[f"{prefix}_{lvl}" for lvl in levels]]


def build_feature_frame(
    df: pd.DataFrame,
    *,
    fit_levels: dict | None = None,
    feature_set: str = "rf",
) -> tuple[pd.DataFrame, dict]:
    """Build feature frame. fit_levels carries categorical levels to keep columns aligned
    across train/test. Returns (X_df, levels)."""
    df = df.copy()
    df["pigment_category"] = df["sample"].map(pigment_category)
    df["cond_is_uv"] = (df["aging_condition"] == "UV").astype(float)

    if fit_levels is None:
        levels = {
            "sample": sorted(df["sample"].astype(str).unique().tolist()),
            "aging_condition": sorted(df["aging_condition"].astype(str).unique().tolist()),
            "pigment_category": [c for c in PIGMENT_CATEGORIES if c in set(df["pigment_category"])],
            "sample_cond": sorted((df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)).unique().tolist()),
        }
    else:
        levels = fit_levels

    tfeat = _time_features(df["aging_time_day"].to_numpy())
    cfeat = _color_features(df["L0"].to_numpy(), df["a0"].to_numpy(), df["b0"].to_numpy())

    # Interaction of log(t+1) and condition
    tfeat["log1p_t_x_uv"] = tfeat["log1p_t"] * df["cond_is_uv"].to_numpy()
    tfeat["log1p_t_x_hh"] = tfeat["log1p_t"] * (1.0 - df["cond_is_uv"].to_numpy())

    numeric = pd.DataFrame({**tfeat, **cfeat})
    # Put aging_time_day first so GBM monotone constraints can reference index 0.
    # numeric has 't' (which is aging_time_day) as first key already.
    numeric = numeric.rename(columns={"t": "aging_time_day"})
    cols_ordered = ["aging_time_day"] + [c for c in numeric.columns if c != "aging_time_day"]
    numeric = numeric[cols_ordered]
    numeric["cond_is_uv"] = df["cond_is_uv"].to_numpy()

    oh_sample = _align_one_hot(df["sample"], levels["sample"], "sample")
    oh_cond = _align_one_hot(df["aging_condition"], levels["aging_condition"], "cond")
    oh_pig = _align_one_hot(df["pigment_category"], levels["pigment_category"], "pig")

    if feature_set == "rf":
        X = pd.concat([numeric.reset_index(drop=True), oh_sample.reset_index(drop=True),
                       oh_cond.reset_index(drop=True), oh_pig.reset_index(drop=True)], axis=1)
    elif feature_set == "gbm":
        sample_codes = pd.Categorical(df["sample"].astype(str), categories=levels["sample"]).codes.astype(float)
        pig_codes = pd.Categorical(df["pigment_category"].astype(str), categories=levels["pigment_category"]).codes.astype(float)
        numeric2 = numeric.copy()
        numeric2["sample_code"] = sample_codes
        numeric2["pigment_code"] = pig_codes
        X = numeric2
    elif feature_set == "linear":
        sc = df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)
        oh_sc = _align_one_hot(sc, levels["sample_cond"], "sc")
        X = pd.concat([numeric.reset_index(drop=True), oh_sc.reset_index(drop=True),
                       oh_pig.reset_index(drop=True)], axis=1)
    elif feature_set == "gpr":
        X = pd.concat([numeric.reset_index(drop=True), oh_pig.reset_index(drop=True),
                       oh_cond.reset_index(drop=True)], axis=1)
    elif feature_set == "curve":
        # Minimal frame — curve fit model uses meta_df directly.
        X = numeric[["aging_time_day"]].copy()
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return X.reset_index(drop=True), levels


def feature_sets() -> list[str]:
    return ["rf", "gbm", "linear", "curve", "gpr"]
