"""Build side-by-side comparison of predictions across all models."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, ensure_output_dirs
from src.constraints import apply_all
from src.data import load_test, load_train
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


def main():
    ensure_output_dirs()
    train = load_train()
    test = load_test()

    out = test[["sample", "aging_condition", "aging_time_day"]].copy()
    tm = train[["sample", "aging_condition", "aging_time_day"]]
    y_tr = train["dietaE"].to_numpy()
    pred_meta = test[["sample", "aging_condition", "aging_time_day"]]

    for name, fac in FACTORIES.items():
        m = fac().fit(train)
        raw = m.predict(test)
        out[name] = apply_all(tm, y_tr, pred_meta, raw)

    # Blend
    p_curve = out["curve"].to_numpy()
    p_tree = out["lgbm"].to_numpy()
    blend = np.zeros(len(out))
    tdf = train.copy()
    tdf["__k"] = tdf["sample"].astype(str) + "__" + tdf["aging_condition"].astype(str)
    stats = tdf.groupby("__k").agg(n=("aging_time_day", "size"),
                                    t_max=("aging_time_day", "max")).to_dict("index")
    for i, row in enumerate(test.itertuples(index=False)):
        key = f"{row.sample}__{row.aging_condition}"
        n = stats.get(key, {}).get("n", 0)
        t_max = stats.get(key, {}).get("t_max", 1)
        gap = max(0.0, (row.aging_time_day - t_max) / max(1.0, t_max))
        w_curve = (n / (n + 2.0)) / (1.0 + gap)
        blend[i] = w_curve * p_curve[i] + (1.0 - w_curve) * p_tree[i]
    out["blend"] = blend

    out.to_csv(METRICS_DIR / "predictions_all_models.csv", index=False)
    print(f"[WROTE] {METRICS_DIR / 'predictions_all_models.csv'}")
    print("\n[All-model predictions]")
    with pd.option_context("display.max_rows", 200, "display.width", 200,
                           "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
