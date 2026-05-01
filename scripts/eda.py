"""Exploratory data analysis: group stats, curves per category, test coverage."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGS_DIR, METRICS_DIR, ensure_output_dirs
from src.data import (
    add_group_key,
    check_test_coverage,
    load_test,
    load_train,
    train_group_stats,
)
from src.features import pigment_category


def _set_cjk_font():
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    picked = next((c for c in candidates if c in available), None)
    if picked:
        plt.rcParams["font.family"] = picked
    plt.rcParams["axes.unicode_minus"] = False


def plot_curves_by_category(train_df, out_dir):
    tr = add_group_key(train_df)
    tr["cat"] = tr["sample"].map(pigment_category)
    for cat, sub in tr.groupby("cat"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, g in sub.groupby("group_key"):
            g = g.sort_values("aging_time_day")
            ax.plot(g["aging_time_day"], g["dietaE"], marker="o", label=key, alpha=0.7, linewidth=1)
        ax.set_xlabel("aging_time_day")
        ax.set_ylabel("dietaE")
        ax.set_title(f"dietaE vs t — {cat}")
        ax.grid(True, alpha=0.3)
        if len(sub.groupby("group_key")) <= 15:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        safe = cat.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"curves_{safe}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_log_t(train_df, out_dir):
    import numpy as np

    tr = train_df.copy()
    tr["cat"] = tr["sample"].map(pigment_category)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (cat, cond), sub in tr.groupby(["cat", "aging_condition"]):
        ax.scatter(np.log1p(sub["aging_time_day"]), sub["dietaE"],
                   label=f"{cat}/{cond}", alpha=0.7, s=20)
    ax.set_xlabel("log(1+t)")
    ax.set_ylabel("dietaE")
    ax.set_title("dietaE vs log(1+t) — by category & condition")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_log_t.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dirs()
    _set_cjk_font()

    train = load_train()
    test = load_test()
    print(f"[DATA] train={len(train)}, test={len(test)}")

    stats = train_group_stats(train)
    stats.to_csv(METRICS_DIR / "train_group_stats.csv", index=False)
    print(f"[WROTE] {METRICS_DIR / 'train_group_stats.csv'}  ({len(stats)} groups)")

    coverage = check_test_coverage(train, test)
    coverage.to_csv(METRICS_DIR / "test_coverage.csv", index=False)
    print(f"[WROTE] {METRICS_DIR / 'test_coverage.csv'}")
    print("\n[Test coverage summary]")
    print(coverage.to_string(index=False))

    plot_curves_by_category(train, FIGS_DIR)
    plot_scatter_log_t(train, FIGS_DIR)
    print(f"\n[WROTE] figs → {FIGS_DIR}")


if __name__ == "__main__":
    main()
