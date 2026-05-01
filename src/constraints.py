from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def zero_at_t0(meta: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=float).copy()
    mask = meta["aging_time_day"].to_numpy() == 0
    preds[mask] = 0.0
    return preds


def clip_nonneg(preds: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(preds, dtype=float), 0.0)


def isotonic_per_group(
    train_meta: pd.DataFrame | None,
    y_train: np.ndarray | None,
    pred_meta: pd.DataFrame,
    preds: np.ndarray,
) -> np.ndarray:
    """Enforce monotonically increasing predictions within each (sample, aging_condition)
    group. Uses training anchors when available so isotonic fit is informed by real data.
    """
    preds = np.asarray(preds, dtype=float).copy()
    out = preds.copy()

    pm = pred_meta.reset_index(drop=True).copy()
    pm["__pred_idx"] = np.arange(len(pm))
    pm["__key"] = pm["sample"].astype(str) + "__" + pm["aging_condition"].astype(str)

    if train_meta is not None and y_train is not None:
        tm = train_meta.reset_index(drop=True).copy()
        tm["__key"] = tm["sample"].astype(str) + "__" + tm["aging_condition"].astype(str)
        tm["__y"] = np.asarray(y_train, dtype=float)
    else:
        tm = None

    for key, sub_pred in pm.groupby("__key"):
        t_pred = sub_pred["aging_time_day"].to_numpy(dtype=float)
        y_pred_grp = preds[sub_pred["__pred_idx"].to_numpy()]
        if tm is not None:
            tm_sub = tm[tm["__key"] == key]
            t_anchor = tm_sub["aging_time_day"].to_numpy(dtype=float)
            y_anchor = tm_sub["__y"].to_numpy(dtype=float)
        else:
            t_anchor = np.array([], dtype=float)
            y_anchor = np.array([], dtype=float)

        t_all = np.concatenate([t_anchor, t_pred])
        y_all = np.concatenate([y_anchor, y_pred_grp])
        mark_pred = np.concatenate([np.zeros_like(t_anchor, dtype=bool),
                                    np.ones_like(t_pred, dtype=bool)])

        if len(t_all) < 2:
            continue
        order = np.argsort(t_all, kind="stable")
        t_sorted = t_all[order]
        y_sorted = y_all[order]
        mark_sorted = mark_pred[order]

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(t_sorted, y_sorted)
        y_fit = iso.predict(t_sorted)
        # Only overwrite prediction rows.
        pred_positions = sub_pred["__pred_idx"].to_numpy()
        # We must map sorted position back to pred row ids.
        sorted_pred_ids = []
        for i_sort, is_pred in enumerate(mark_sorted):
            if is_pred:
                sorted_pred_ids.append(i_sort)
        # Within sorted subset of pred rows, the original order within this group is by
        # increasing t (may include ties). Re-derive via rank of (t, pred_idx).
        pred_rows_in_group = sub_pred.sort_values(
            ["aging_time_day", "__pred_idx"], kind="stable"
        )["__pred_idx"].to_numpy()
        for i_sort_idx, row_id in zip(sorted_pred_ids, pred_rows_in_group):
            out[row_id] = y_fit[i_sort_idx]

    return out


def apply_all(
    train_meta: pd.DataFrame | None,
    y_train: np.ndarray | None,
    pred_meta: pd.DataFrame,
    preds: np.ndarray,
    *,
    do_zero: bool = True,
    do_clip: bool = True,
    do_isotonic: bool = True,
) -> np.ndarray:
    out = np.asarray(preds, dtype=float).copy()
    if do_clip:
        out = clip_nonneg(out)
    if do_isotonic:
        out = isotonic_per_group(train_meta, y_train, pred_meta, out)
    if do_zero:
        out = zero_at_t0(pred_meta, out)
    if do_clip:
        out = clip_nonneg(out)
    return out
