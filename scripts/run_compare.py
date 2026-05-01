"""Compare all models under random K-fold and time-holdout CV."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import METRICS_DIR, ensure_output_dirs
from src.data import load_train
from src.eval import (
    per_group_error,
    random_kfold_cv,
    results_to_df,
    time_holdout_cv,
)
from src.models.curve_fit_model import CurveFitModel
from src.models.gbm_model import LGBMModel, XGBModel
from src.models.gpr_model import GPRModel
from src.models.linear_model import PolyModel, RidgeCVModel
from src.models.rf_model import RFBaselineModel, RFModel


MODEL_FACTORIES = {
    "rf_organizer": lambda: RFBaselineModel(),
    "rf_improved": lambda: RFModel(),
    "xgb": lambda: XGBModel(),
    "lgbm": lambda: LGBMModel(),
    "ridge": lambda: RidgeCVModel(),
    "poly": lambda: PolyModel(degree=2),
    "curve": lambda: CurveFitModel(),
    "gpr": lambda: GPRModel(),
}


def main():
    ensure_output_dirs()
    train = load_train()
    print(f"[DATA] train rows={len(train)}")

    all_results = []
    for name, factory in MODEL_FACTORIES.items():
        print(f"\n=== {name} ===")
        try:
            r1 = random_kfold_cv(factory, train, model_name=name)
            print(f"  random-kfold  RMSE={r1.rmse:.4f}  score100={r1.score_100:.2f}  MAE={r1.mae:.4f}  time={r1.fit_seconds:.2f}s")
            all_results.append(r1)
        except Exception as e:
            print(f"  [ERR] random-kfold: {type(e).__name__}: {e}")

        try:
            r2 = time_holdout_cv(factory, train, model_name=name)
            print(f"  time-holdout  RMSE={r2.rmse:.4f}  score100={r2.score_100:.2f}  MAE={r2.mae:.4f}  time={r2.fit_seconds:.2f}s")
            all_results.append(r2)
        except Exception as e:
            print(f"  [ERR] time-holdout: {type(e).__name__}: {e}")

    df = results_to_df(all_results).sort_values(["cv_scheme", "rmse"]).reset_index(drop=True)
    df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    # Per-group error breakdown using the winner of time-holdout CV.
    th = df[df["cv_scheme"] == "time_holdout"].sort_values("rmse")
    if len(th) > 0:
        winner = th.iloc[0]["model"]
        print(f"\n[Winner by time-holdout] {winner}")

        # Re-run with the winner to collect per-group errors using holdout rows.
        from src.eval import _fit_predict  # type: ignore
        import numpy as np
        df_full = train.reset_index(drop=True).copy()
        df_full["__key"] = df_full["sample"].astype(str) + "__" + df_full["aging_condition"].astype(str)
        holdout_idx = []
        for key, sub in df_full.groupby("__key"):
            sub_nz = sub[sub["aging_time_day"] > 0]
            if len(sub_nz) < 2:
                continue
            holdout_idx.append(sub_nz["aging_time_day"].idxmax())
        holdout_idx = sorted(set(holdout_idx))
        va = df_full.loc[holdout_idx].drop(columns="__key").reset_index(drop=True)
        tr = df_full.drop(index=holdout_idx).drop(columns="__key").reset_index(drop=True)
        yt, yp = _fit_predict(MODEL_FACTORIES[winner], tr, va)
        pg = per_group_error(va[["sample", "aging_condition"]], yt, yp)
        pg.to_csv(METRICS_DIR / "per_group_errors.csv", index=False)
        print("\n[Per-group errors (winner, time-holdout)]")
        print(pg.to_string(index=False))


if __name__ == "__main__":
    main()
