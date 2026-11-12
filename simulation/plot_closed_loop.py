"""Plot paired localization contrasts and the independent reporting validation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np


OUT = Path(__file__).resolve().parent / "results"
DATA = json.loads((OUT / "closed_loop_summary.json").read_text(encoding="utf-8"))
WIDTHS = (1.2, 1.6, 2.0)
COLORS = ("#1f6378", "#60768e", "#9a4a18")
MARKERS = ("s", "^", "o")


def style():
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("A Chinese sans-serif font is required")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "legend.frameon": False, "savefig.dpi": 400})


def save(fig, name):
    fig.savefig(OUT / name, dpi=400, facecolor="white")
    plt.close(fig)


def paired():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), sharex=True, sharey=True)
    keys = ("single_2km", "single_1km", "four_mean")
    labels = ("2 km 单网格", "1 km 单网格", "四网格均值")
    for ax, distance, letter in zip(axes, (1, 2), ("a", "b")):
        ax.set_title(f"{letter}  真位移 {distance} km", loc="left", fontweight="bold")
        ax.axvline(0, color="#5e636a", lw=.8, ls="--")
        for offset, sigma, color, marker in zip((-.20, 0, .20), WIDTHS, COLORS, MARKERS):
            result = next(r for r in DATA["groups"] if r["truth_distance_km"] == distance and r["sigma_km"] == sigma)
            comp = result["four_median_comparisons"]
            effects = np.array([comp[k]["difference_km"] for k in keys])
            bounds = np.array([comp[k]["ci95"] for k in keys]).T
            ax.errorbar(effects, np.arange(len(keys)) + offset,
                        xerr=np.vstack((effects - bounds[0], bounds[1] - effects)),
                        fmt=marker, color=color, lw=.95, capsize=2, ms=4, label=f"σ = {sigma:.1f} km")
        ax.set_yticks(range(len(keys)), labels)
        ax.set_xlabel("四网格中位数减对照的误差差值 (km)")
        ax.grid(axis="x", color="#dce1e4", lw=.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, bbox_to_anchor=(.53, .00), fontsize=8)
    fig.subplots_adjust(left=.19, bottom=.25, right=.98, top=.86, wspace=.14)
    save(fig, "closed_loop_ablation.png")


def reporting():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.95), sharey=True)
    for ax, mode, name in zip(axes, ("fixed", "calibrated"), ("a  固定 1 km", "b  校准后")):
        ax.set_title(name, loc="left", fontweight="bold")
        for ref in (.05, .80):
            ax.axhline(ref, color="#66717a", ls=":", lw=.9)
        for sigma, color, marker in zip(WIDTHS, COLORS, MARKERS):
            rows = [r for r in DATA["groups"] if r["sigma_km"] == sigma]
            x = [r["truth_distance_km"] for r in rows]
            y = [r["methods"]["four_median"][mode]["rate"] for r in rows]
            ax.plot(x, y, marker=marker, ms=4, color=color, lw=1.2, label=f"σ = {sigma:.1f} km")
        ax.set_ylim(-.03, 1.04)
        ax.set_xticks([0, .5, 1, 2, 3])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("真位移 (km)")
        ax.grid(axis="y", color="#dce1e4", lw=.5)
    axes[0].set_ylabel("检出位移的样本比例")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, bbox_to_anchor=(.53, .00), fontsize=8)
    fig.subplots_adjust(left=.12, bottom=.24, right=.98, top=.87, wspace=.16)
    save(fig, "closed_loop_reporting.png")


if __name__ == "__main__":
    style()
    paired()
    reporting()

