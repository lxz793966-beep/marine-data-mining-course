"""Render two report figures from the saved AIS grid simulation outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WIDTH_CM = 16
DPI = 600
INK = "#26333D"
GRID = "#DCE3E7"
SINGLE = "#2D627A"
FOUR = "#A85316"
METHOD_STYLE = {
    "single": {"facecolor": "#DBE9EF", "edgecolor": SINGLE, "hatch": "///"},
    "four": {"facecolor": "#DF984F", "edgecolor": FOUR},
}
METHOD_NAMES = {"single": "单网格", "four": "四原点中位数"}
CENTER_STYLE = {
    "truth": {"marker": "x", "color": INK, "markerfacecolor": "none"},
    "single": {"marker": "s", "color": SINGLE, "markerfacecolor": "none"},
    "four": {"marker": "D", "color": FOUR, "markerfacecolor": "none"},
}


def set_style() -> None:
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
        "font.family": family,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.65,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "hatch.linewidth": 0.55,
    })


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_main_summary(summary: dict, trials: list[dict[str, str]], threshold: float) -> None:
    for group in summary["distance"]:
        distance = group["truth_distance_km"]
        for method in METHOD_STYLE:
            rows = [row for row in trials if int(row["truth_distance_km"]) == distance and row["method"] == method]
            errors = [float(row["vector_error_km"]) for row in rows]
            reports = [int(row["reported"]) for row in rows]
            stored = group[method]
            counts_match = len(rows) == group["n_pairs"] and sum(reports) == stored["reported_count"]
            metrics_match = np.allclose(
                [np.mean(errors), np.quantile(errors, 0.9), np.mean(reports)],
                [stored["mean_vector_error_km"], stored["p90_vector_error_km"], stored["report_rate"]],
                rtol=0,
                atol=1e-12,
            )
            threshold_matches = all(flag == (float(row["est_length_km"]) >= threshold) for row, flag in zip(rows, reports))
            if not (counts_match and metrics_match and threshold_matches):
                raise ValueError(f"主实验汇总与明细不一致。真实位移 {distance} km，方法 {method}。")


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(RESULTS / name, dpi=DPI)
    plt.close(figure)


def draw_grid_tiles(figure: plt.Figure, protocol: dict) -> None:
    width = protocol["cell_width_km"]
    figure.text(0.052, 0.865, "a  四种网格原点", weight="bold", fontsize=9)
    figure.text(0.052, 0.827, f"格宽 {width:g} km，原点坐标单位为 km", fontsize=7.3)

    for index, (origin_x, origin_y) in enumerate(protocol["origins_km"]):
        left = 0.054 + (index % 2) * 0.135
        bottom = 0.533 - (index // 2) * 0.259
        ax = figure.add_axes([left, bottom, 0.115, 0.20])
        ax.set(xlim=(0, 4), ylim=(0, 4), xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values():
            spine.set(visible=True, color=GRID, linewidth=0.65)
        ax.add_patch(Rectangle((origin_x, origin_y), width, width,
                               facecolor="#E6F0F4", edgecolor="none"))
        for value in np.arange(origin_x - width, 5, width):
            ax.axvline(value, color=SINGLE, linewidth=0.7, alpha=0.6)
        for value in np.arange(origin_y - width, 5, width):
            ax.axhline(value, color=SINGLE, linewidth=0.7, alpha=0.6)
        ax.plot(origin_x, origin_y, "o", color=FOUR, ms=3.5, clip_on=False)
        ax.set_title(f"({origin_x:g}, {origin_y:g})", fontsize=7.7, pad=3)


def draw_example(protocol: dict, centers: dict, rows: list[dict[str, str]]) -> None:
    figure = plt.figure(figsize=(WIDTH_CM / 2.54, 8.4 / 2.54))
    draw_grid_tiles(figure, protocol)
    width = protocol["cell_width_km"]
    origin_x, origin_y = protocol["origins_km"][0]
    expected_count = protocol["n_hotspot"] + protocol["n_background"]
    figure.text(0.400, 0.865, "b  初期点云", weight="bold", fontsize=9)
    figure.text(0.725, 0.865, "c  后期点云", weight="bold", fontsize=9)
    figure.text(0.400, 0.827, "两期按同一范围放大，浅线为单网格", fontsize=7.3)

    # Both panels use the same crop and display all points in the view.
    for period, left in enumerate((0.400, 0.725)):
        ax = figure.add_axes([left, 0.274, 0.244, 0.465])
        points = np.array([(float(row["x_km"]), float(row["y_km"])) for row in rows if int(row["period"]) == period])
        if len(points) != expected_count:
            raise ValueError(f"示例第 {period + 1} 期点数不等于实验设定。")
        ax.set(xlim=(15, 25), ylim=(14, 24), xticks=[16, 20, 24], yticks=[16, 20, 24], aspect="equal")
        for spine in ax.spines.values():
            spine.set(visible=True, color="#8997A1", linewidth=0.55)
        for value in np.arange(origin_x, 26, width):
            ax.axvline(value, color=GRID, linewidth=0.55, zorder=0)
        for value in np.arange(origin_y, 26, width):
            ax.axhline(value, color=GRID, linewidth=0.55, zorder=0)
        ax.scatter(points[:, 0], points[:, 1], s=3.1, color="#667580", linewidths=0, alpha=0.33, zorder=2)
        for method, style in CENTER_STYLE.items():
            x, y = centers[method][period]
            ax.plot(x, y, linestyle="none", markersize=6.4, markeredgewidth=1.25, zorder=4, **style)
        ax.set_xlabel("东向（km）", labelpad=2)
        ax.tick_params(length=2, pad=2, labelleft=period == 0)
        if period == 0:
            ax.set_ylabel("北向（km）", labelpad=2)

    legend_handles = [Line2D([], [], linestyle="none", markersize=5.8, markeredgewidth=1.2,
                             label="生成中心" if key == "truth" else METHOD_NAMES[key], **style)
                      for key, style in CENTER_STYLE.items()]
    figure.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.52, 0.122),
                  ncols=3, fontsize=7.6, columnspacing=1.5, handletextpad=0.35)
    lengths = {key: float(np.linalg.norm(np.diff(np.array(centers[key]), axis=0)[0])) for key in CENTER_STYLE}
    figure.text(0.5, 0.051,
                f"固定模拟例（种子 {centers['seed']}，方向扇区 {centers['sector'] + 1}）"
                f"　真值 {lengths['truth']:.2f} km　单网格 {lengths['single']:.2f} km　四原点 {lengths['four']:.2f} km",
                ha="center", fontsize=7.5)
    save(figure, "design.png")


def draw_results(summary: dict, threshold: float) -> None:
    figure = plt.figure(figsize=(WIDTH_CM / 2.54, 8.5 / 2.54))
    legend_handles = [Patch(label=METHOD_NAMES[method], linewidth=0.7, **style)
                      for method, style in METHOD_STYLE.items()]
    figure.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.53, 0.935),
                  ncols=2, fontsize=8, columnspacing=2, handlelength=1.8)
    distances = summary["distance"]
    x = np.arange(len(distances))
    positions = (0.092, 0.410, 0.735)
    metrics = ("mean_vector_error_km", "p90_vector_error_km", "report_rate")
    titles = ("a  平均向量误差", "b  P90 向量误差", "c  位移检出比例")
    max_error = max(group[method][metric] for group in distances for method in METHOD_STYLE for metric in metrics[:2])
    error_limit = max(2.1, np.ceil(max_error * 2) / 2 + 0.1)

    for left, metric, title in zip(positions, metrics, titles):
        ax = figure.add_axes([left, 0.292, 0.235, 0.464])
        ax.set_title(title, loc="left", weight="bold", pad=7)
        ax.set_xticks(x, [str(group["truth_distance_km"]) for group in distances])
        ax.set_xlim(-0.55, len(x) - 0.45)
        ax.tick_params(length=2.5, pad=2.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        is_rate = metric == "report_rate"
        if is_rate:
            ax.set(ylim=(0, 113), yticks=[0, 25, 50, 75, 100], ylabel="比例（%）")
        else:
            ax.set(ylim=(0, error_limit), yticks=np.arange(0, error_limit, 0.5))
            if metric == metrics[0]:
                ax.set_ylabel("误差（km）")

        for method, offset in zip(METHOD_STYLE, (-0.22, 0.22)):
            values = [group[method][metric] * (100 if is_rate else 1) for group in distances]
            bars = ax.bar(x + offset, values, width=0.34, linewidth=0.7, **METHOD_STYLE[method])
            labels = [f"{value:.0f}" if is_rate else f"{value:.2f}" for value in values]
            ax.bar_label(bars, labels=labels, fontsize=7.1, padding=2.3, color=INK)

    figure.text(0.53, 0.192, "真实位移（km）", ha="center", fontsize=8.1)
    counts = "、".join(str(group["n_pairs"]) for group in distances)
    figure.text(0.5, 0.117,
                f"主实验（σ = {summary['sigma_km']:g} km），各位移组依次有 {counts} 对模拟样本。",
                ha="center", fontsize=7.5)
    figure.text(0.5, 0.060,
                f"估计位移 ≥ {threshold:g} km 时检出位移。0 km 组的检出属于误报。",
                ha="center", fontsize=7.5)
    save(figure, "results.png")


def main() -> None:
    set_style()
    protocol = read_json(ROOT / "protocol.json")
    main_summary = read_json(RESULTS / "summary.json")["scenarios"]["main"]
    threshold = protocol["reporting_threshold_km"]
    check_main_summary(main_summary, read_csv(RESULTS / "main_trials.csv"), threshold)
    draw_example(protocol, read_json(RESULTS / "example_centers.json"), read_csv(RESULTS / "example_points.csv"))
    draw_results(main_summary, threshold)


if __name__ == "__main__":
    main()
