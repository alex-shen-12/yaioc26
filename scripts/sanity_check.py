"""Score each model's test predictions against physics sanity and CV proxy."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.config import METRICS_DIR
from src.data import load_test, load_train

PRED_FILES = {
    "rf_organizer": "outputs/predict_rf_organizer.csv",
    "rf_improved":  "outputs/predict_out.csv.rf_improved",  # placeholder, not used
    "xgb":          "outputs/predict_xgb.csv",
    "lgbm":         "outputs/predict_lgbm.csv",
    "curve":        "outputs/predict_curve.csv",
    "blend":        "outputs/predict_blend.csv",
}


def monotonic_violations(test_df, preds):
    """For groups with >1 prediction at different t, check monotonic non-decreasing."""
    df = test_df.copy()
    df["__k"] = df["sample"].astype(str) + "__" + df["aging_condition"].astype(str)
    df["pred"] = preds
    v = 0
    for _, g in df.groupby("__k"):
        if len(g) < 2:
            continue
        s = g.sort_values("aging_time_day")
        for i in range(1, len(s)):
            if s.iloc[i]["pred"] < s.iloc[i - 1]["pred"] - 1e-6:
                v += 1
    return v


def consistency_with_train(train_df, test_df, preds):
    """Mean over test rows of |pred - train_ymax_same_group| / (train_ymax + eps)
    expected to be close to "growth ratio" implied by time.
    We flag: how often pred < train_ymax when t_test > t_max_train (physics violation).
    """
    tdf = train_df.copy()
    tdf["__k"] = tdf["sample"].astype(str) + "__" + tdf["aging_condition"].astype(str)
    stats = tdf.groupby("__k").agg(ymax=("dietaE", "max"),
                                   tmax=("aging_time_day", "max")).to_dict("index")
    bad = 0
    gaps = []
    for i, row in enumerate(test_df.itertuples(index=False)):
        key = f"{row.sample}__{row.aging_condition}"
        s = stats.get(key)
        if not s:
            continue
        if row.aging_time_day > s["tmax"]:
            if preds[i] < s["ymax"] - 0.01:
                bad += 1
            gaps.append(preds[i] - s["ymax"])
    return bad, float(np.mean(gaps)) if gaps else 0.0


def main():
    train = load_train()
    test = load_test()

    rows = []
    for name, path in PRED_FILES.items():
        try:
            preds = pd.read_csv(path)["dietaE"].to_numpy()
        except Exception:
            continue
        v = monotonic_violations(test, preds)
        bad, mean_gap = consistency_with_train(train, test, preds)
        rows.append({
            "model": name,
            "n_preds": len(preds),
            "min": preds.min(),
            "max": preds.max(),
            "mean": preds.mean(),
            "monotonic_violations": v,  # same group, later t has smaller pred
            "below_train_ymax_on_extrap": bad,  # pred < ymax when t_test > t_max_train
            "mean_pred_minus_ymax": mean_gap,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS_DIR / "submission_sanity.csv", index=False)
    with pd.option_context("display.width", 180, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
