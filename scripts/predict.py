"""Produce the competition submission from a trained model."""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np

from src.config import MODELS_DIR, SUBMISSION_CSV, ensure_output_dirs
from src.constraints import apply_all
from src.data import load_test, load_train, write_submission
from src.models.curve_fit_model import CurveFitModel
from src.models.gbm_model import LGBMModel, XGBModel
from src.models.gpr_model import GPRModel
from src.models.linear_model import PolyModel, RidgeCVModel
from src.models.rf_model import RFBaselineModel, RFModel


FACTORIES = {
    "rf_organizer": lambda: RFBaselineModel(),
    "rf_improved": lambda: RFModel(),
    "xgb": lambda: XGBModel(),
    "lgbm": lambda: LGBMModel(),
    "ridge": lambda: RidgeCVModel(),
    "poly": lambda: PolyModel(),
    "curve": lambda: CurveFitModel(),
    "gpr": lambda: GPRModel(),
}


def blend_predictions(test_df, train_df, preds_by_model: dict) -> np.ndarray:
    """Blend curve-fit with a tree model based on extrapolation gap and group size.

    For each test row, weight w_curve depends on how much training info we have:
      - larger n_train → higher w_curve
      - larger extrap gap → lower w_curve
    """
    tdf = train_df.copy()
    tdf["__k"] = tdf["sample"].astype(str) + "__" + tdf["aging_condition"].astype(str)
    stats = tdf.groupby("__k").agg(n=("aging_time_day", "size"),
                                   t_max=("aging_time_day", "max")).to_dict("index")

    out = np.zeros(len(test_df), dtype=float)
    p_curve = preds_by_model["curve"]
    p_tree = preds_by_model.get("lgbm", preds_by_model.get("xgb", preds_by_model.get("rf_improved")))
    for i, row in enumerate(test_df.itertuples(index=False)):
        key = f"{row.sample}__{row.aging_condition}"
        n = stats.get(key, {}).get("n", 0)
        t_max = stats.get(key, {}).get("t_max", 1)
        gap = max(0.0, (row.aging_time_day - t_max) / max(1.0, t_max))
        w_curve = (n / (n + 2.0)) / (1.0 + gap)
        w_curve = min(max(w_curve, 0.0), 1.0)
        out[i] = w_curve * p_curve[i] + (1.0 - w_curve) * p_tree[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="curve", choices=list(FACTORIES) + ["blend"])
    ap.add_argument("--out", default=str(SUBMISSION_CSV))
    args = ap.parse_args()

    ensure_output_dirs()
    train = load_train()
    test = load_test()

    tm = train[["sample", "aging_condition", "aging_time_day"]]
    y_tr = train["dietaE"].to_numpy()
    pred_meta = test[["sample", "aging_condition", "aging_time_day"]]

    if args.model == "blend":
        preds_by = {}
        for name in ("curve", "lgbm"):
            m = FACTORIES[name]().fit(train)
            preds_by[name] = m.predict(test)
        raw = blend_predictions(test, train, preds_by)
    else:
        model = FACTORIES[args.model]().fit(train)
        raw = model.predict(test)
        model.save(MODELS_DIR / f"{args.model}.pkl")

    final = apply_all(tm, y_tr, pred_meta, raw)
    write_submission(final, args.out)
    print(f"[OK] submission: {args.out}  (rows={len(final)})")
    print("\n[Preview]")
    import pandas as pd
    prev = test[["sample", "aging_condition", "aging_time_day"]].copy()
    prev["dietaE_pred"] = final
    print(prev.to_string(index=False))


if __name__ == "__main__":
    main()
