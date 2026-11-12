"""Build report-sized AIS diagnostics from the saved paired simulation records."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


METHODS = ("single", "four")
LABELS = {"single": "单网格", "four": "四网格中位数"}
COLORS = {"single": "#2D627A", "four": "#A85316"}
LINES = {"single": "--", "four": "-"}
MARKERS = {"single": "s", "four": "D"}
SCENARIOS = {"main": "主实验（标准差 1.2 km）", "wide": "较宽热点（标准差 2.0 km）"}
WIDTH_IN = 15.5 / 2.54
INK = "#26333D"
GRID = "#DCE3E7"


def style():
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename in ("msyh.ttc", "msyhbd.ttc", "simsun.ttc"):
        path = fonts / filename
        if path.is_file():
            font_manager.fontManager.addfont(path)
    for family in ("Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("请安装微软雅黑、宋体或 Noto Sans CJK SC 中文字体。")
    mpl.rcParams.update({
        "font.family": family, "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
        "hatch.linewidth": 0.65,
    })


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_trials(rows):
    groups = {}
    for row in rows:
        key = (int(row["seed"]), int(row["truth_distance_km"]), int(row["sector"]))
        method = row["method"]
        records = groups.setdefault(key, {})
        if method not in METHODS or method in records:
            raise ValueError(f"Invalid or duplicate method for trial {key}: {method}")
        records[method] = row
    expected = {(seed, 0, -1) for seed in range(100)}
    expected.update((seed, distance, sector)
                    for seed in range(100) for distance in (1, 2) for sector in range(8))
    if set(groups) != expected or any(set(pair) != set(METHODS) for pair in groups.values()):
        raise ValueError("Paired trials do not match the saved 100-seed design.")
    return groups


def direction_summary(pairs, stored_groups):
    statistics = []
    for distance in (1, 2):
        reference = next(group for group in stored_groups if group["truth_distance_km"] == distance)
        for sector in range(8):
            angles = [float(pairs[(seed, distance, sector)]["single"]["angle_deg"])
                      for seed in range(100)]
            if not all(45 * sector <= angle < 45 * (sector + 1) for angle in angles):
                raise ValueError(f"Angle outside its sector: {distance}, {sector}")
            record = {"truth_km": distance, "sector": sector,
                      "angle_interval_deg": [45 * sector, 45 * (sector + 1)], "pairs": 100}
            for method in METHODS:
                errors = [float(pairs[(seed, distance, sector)][method]["vector_error_km"])
                          for seed in range(100)]
                record[method] = float(np.mean(errors))
                np.testing.assert_allclose(record[method],
                                           reference["by_sector_mean_vector_error_km"][sector][method],
                                           rtol=0, atol=1e-12)
            record["four_minus_single_km"] = record["four"] - record["single"]
            statistics.append(record)
    return statistics


def zero_summary(pairs, stored_distance_zero):
    summary = {}
    for method in METHODS:
        rows = [pairs[(seed, 0, -1)][method] for seed in range(100)]
        lengths = np.array([float(row["est_length_km"]) for row in rows])
        reported = [int(row["reported"]) for row in rows]
        np.testing.assert_array_equal(lengths >= 1, reported)
        bins = (lengths == 0, (0 < lengths) & (lengths < 1), lengths == 1, lengths > 1)
        counts = [int(np.count_nonzero(condition)) for condition in bins]
        if sum(counts) != 100 or sum(counts[2:]) != stored_distance_zero[method]["reported_count"]:
            raise ValueError(f"Threshold bin counts mismatch for {method}.")
        values, frequencies = np.unique(lengths, return_counts=True)
        summary[method] = {"bin_counts": counts,
                           "exact_counts": [[float(value), int(count)]
                                            for value, count in zip(values, frequencies)]}
    return summary


def phase_summary(rows, stored):
    spans = {}
    statistics = {}
    for method in METHODS:
        by_seed = {int(row["seed"]): float(row["max_span_km"])
                   for row in rows if row["method"] == method}
        if set(by_seed) != set(range(100)):
            raise ValueError(f"Incomplete phase samples for {method}.")
        values = np.array([by_seed[seed] for seed in range(100)])
        spans[method] = values
        statistics[method] = {
            "mean_km": float(np.mean(values)),
            "median_km": float(np.median(values)),
            "p90_km": float(np.quantile(values, 0.9)),
            "fraction_at_or_below_1km": float(np.mean(values <= 1)),
        }
        for field in ("mean_km", "median_km", "p90_km"):
            np.testing.assert_allclose(statistics[method][field], stored[method][field],
                                       rtol=0, atol=1e-12)
    if len(rows) != 200:
        raise ValueError("Phase file must have exactly two records per seed.")
    changes = spans["four"] - spans["single"]
    statistics["paired_change"] = {
        "lower": int(np.count_nonzero(changes < -1e-12)),
        "equal": int(np.count_nonzero(np.abs(changes) <= 1e-12)),
        "higher": int(np.count_nonzero(changes > 1e-12)),
        "four_minus_single_mean_km": float(np.mean(changes)),
        "four_minus_single_ci95_km": stored["delta_mean_km"]["ci95"],
    }
    np.testing.assert_allclose(statistics["paired_change"]["four_minus_single_mean_km"],
                               stored["delta_mean_km"]["estimate"], rtol=0, atol=1e-12)
    return spans, statistics


def figure_pair(height_cm=7.7, bottom=0.19):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, height_cm / 2.54), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.79, bottom=bottom, wspace=0.16)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(length=3, width=0.6, pad=3)
    return fig, axes


def legend(fig, axes):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               fontsize=8.7, columnspacing=1.6, handlelength=2, handletextpad=0.5)


def direction_plot(records, out):
    fig, axes = figure_pair()
    for ax, distance in zip(axes, (1, 2)):
        rows = [row for row in records if row["truth_km"] == distance]
        sectors = [row["sector"] + 1 for row in rows]
        for method in METHODS:
            ax.plot(sectors, [row[method] for row in rows], color=COLORS[method],
                    ls=LINES[method], marker=MARKERS[method], markersize=4,
                    linewidth=1.25, markerfacecolor="white" if method == "single" else COLORS[method],
                    markeredgewidth=1, label=LABELS[method])
        ax.set(xlim=(0.7, 8.3), ylim=(0, 1.5), xticks=sectors,
               xlabel="方向扇区编号", title=f"真实位移 {distance} km")
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
    axes[0].set_ylabel("平均位移向量误差（km）")
    legend(fig, axes)
    save(fig, out / "direction_errors.png")


def zero_plot(summaries, out):
    fig, axes = figure_pair(height_cm=7.7)
    x = np.arange(4)
    for ax, (scenario, title) in zip(axes, SCENARIOS.items()):
        for method, offset in zip(METHODS, (-0.18, 0.18)):
            counts = summaries[scenario][method]["bin_counts"]
            bars = ax.bar(x + offset, counts, width=0.32, edgecolor=COLORS[method],
                          linewidth=0.8, color="#DBE9EF" if method == "single" else "#DF984F",
                          hatch="///" if method == "single" else None, label=LABELS[method])
            ax.bar_label(bars, labels=[str(count) if count else "" for count in counts],
                         padding=2, fontsize=7.5)
        ax.axvline(1.5, color="#666666", ls=(0, (2, 2)), lw=0.85)
        ax.set(ylim=(0, 107), xticks=x, xticklabels=["0", "(0, 1)", "1", ">1"],
               xlabel="估计位移长度（km）", title=title)
        ax.yaxis.set_major_locator(MultipleLocator(20))
    axes[0].set_ylabel("样本组数")
    legend(fig, axes)
    save(fig, out / "zero_displacement.png")


def phase_plot(spans, out):
    fig, axes = figure_pair(height_cm=7.7)
    maximum = max(values.max() for methods in spans.values() for values in methods.values())
    right = float(np.ceil(maximum * 2) / 2)
    for ax, (scenario, title) in zip(axes, SCENARIOS.items()):
        for method in METHODS:
            x, counts = np.unique(spans[scenario][method], return_counts=True)
            xp = np.r_[0, x, right]
            yp = np.r_[0, counts.cumsum() / counts.sum(), 1]
            ax.step(xp, yp, where="post", color=COLORS[method], ls=LINES[method],
                    linewidth=1.5, label=LABELS[method])
        ax.set(xlim=(0, right), ylim=(0, 1.04), xlabel="热点位置最大跨度（km）", title=title)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylabel("累积比例")
    legend(fig, axes)
    save(fig, out / "origin_sensitivity.png")


def save(fig, path):
    fig.savefig(path, dpi=480)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_results = Path(__file__).resolve().parent / "results"
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--out-dir", type=Path)
    options = parser.parse_args()
    source = options.results_dir
    out = options.out_dir or source
    stored = json.loads((source / "summary.json").read_text(encoding="utf-8"))["scenarios"]
    out.mkdir(parents=True, exist_ok=True)
    style()
    phase_spans, phase_stats, zeros = {}, {}, {}
    for scenario in SCENARIOS:
        pairs = group_trials(load_csv(source / f"{scenario}_trials.csv"))
        if scenario == "main":
            directions = direction_summary(pairs, stored[scenario]["distance"])
        zeros[scenario] = zero_summary(pairs, stored[scenario]["distance"][0])
        phase_spans[scenario], phase_stats[scenario] = phase_summary(
            load_csv(source / f"{scenario}_origin_sensitivity.csv"),
            stored[scenario]["phase_sensitivity"],
        )
    direction_plot(directions, out)
    zero_plot(zeros, out)
    phase_plot(phase_spans, out)
    evidence = {
        "direction_main_by_sector": directions,
        "zero_length_bins_km": ["= 0", "0 < length < 1", "= 1", "> 1"],
        "zero_by_scenario": zeros,
        "phase_by_scenario": phase_stats,
    }
    (out / "revision_figure_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
