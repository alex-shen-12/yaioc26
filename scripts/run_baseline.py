"""Improved RF baseline: end-to-end training + CV + submission."""
from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import METRICS_DIR, MODELS_DIR, SUBMISSION_CSV, ensure_output_dirs
from src.constraints import apply_all
from src.data import assert_t0_is_zero, load_test, load_train, write_submission
from src.eval import random_kfold_cv, results_to_df, time_holdout_cv
from src.models.rf_model import RFModel


def main():
    ensure_output_dirs()
    train = load_train()
    test = load_test()
    assert_t0_is_zero(train)
    print(f"[DATA] train rows={len(train)}, test rows={len(test)}")

    factory = lambda: RFModel()

    r1 = random_kfold_cv(factory, train, model_name="rf_improved")
    print(f"[CV random 5-fold]    RMSE={r1.rmse:.4f}  score100={r1.score_100:.2f}  MAE={r1.mae:.4f}  time={r1.fit_seconds:.2f}s")
    print(f"                      per-fold RMSE: {[round(x, 4) for x in r1.per_fold_rmse]}")

    r2 = time_holdout_cv(factory, train, model_name="rf_improved")
    print(f"[CV time-holdout]     RMSE={r2.rmse:.4f}  score100={r2.score_100:.2f}  MAE={r2.mae:.4f}  time={r2.fit_seconds:.2f}s")

    results_to_df([r1, r2]).to_csv(METRICS_DIR / "rf_improved_cv.csv", index=False)

    # Refit on full data and generate submission
    model = RFModel().fit(train)
    raw_preds = model.predict(test)
    tm = train[["sample", "aging_condition", "aging_time_day"]]
    y_tr = train["dietaE"].to_numpy()
    pred_meta = test[["sample", "aging_condition", "aging_time_day"]]
    preds = apply_all(tm, y_tr, pred_meta, raw_preds)

    write_submission(preds, SUBMISSION_CSV)
    model.save(MODELS_DIR / "rf_improved.pkl")
    print(f"[OK] submission: {SUBMISSION_CSV}")
    print(f"[OK] model saved: {MODELS_DIR / 'rf_improved.pkl'}")


if __name__ == "__main__":
    main()
