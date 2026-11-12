"""Compare single-grid and four-grid AIS report-density peak displacement."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
WIDTH = CFG["cell_width_km"]
ORIGINS = np.asarray(CFG["origins_km"], dtype=float)
SHIFTS = np.asarray(CFG["origin_sensitivity_shifts_km"], dtype=float)
SEEDS = list(range(CFG["simulation_seeds"][0], CFG["simulation_seeds"][1] + 1))
METHODS = ("single", "four")


def sample_cloud(rng: np.random.Generator, center: np.ndarray, sigma: float) -> np.ndarray:
    """Draw fixed counts of hotspot and uniform-background reports."""
    low, high = CFG["domain_km"]
    hotspot = rng.normal(center, sigma, size=(CFG["n_hotspot"], 2))
    rejected = np.any((hotspot < low) | (hotspot >= high), axis=1)
    while np.any(rejected):
        hotspot[rejected] = rng.normal(center, sigma, size=(int(rejected.sum()), 2))
        rejected = np.any((hotspot < low) | (hotspot >= high), axis=1)
    background = rng.uniform(low, high, size=(CFG["n_background"], 2))
    return np.vstack((hotspot, background))


def peak(points: np.ndarray, origin: np.ndarray, width: float = WIDTH) -> tuple[np.ndarray, int]:
    """Use half-open cells and average the centers of equally hot cells."""
    grid_indices = np.floor((points - origin) / width).astype(np.int64)
    cells, counts = np.unique(grid_indices, axis=0, return_counts=True)
    centers = origin + width * (cells + 0.5)
    lo, hi = CFG["eligible_cell_center_km"]
    eligible = np.all((centers >= lo) & (centers <= hi), axis=1)
    if not eligible.any():
        raise ValueError("No observed cell center falls in the search region")
    hottest = eligible & (counts == counts[eligible].max())
    return centers[hottest].mean(axis=0), int(hottest.sum())


def estimates(points: np.ndarray, shift: np.ndarray = SHIFTS[0]) -> tuple[np.ndarray, np.ndarray]:
    fitted = [peak(points, origin + shift) for origin in ORIGINS]
    centers = np.array([center for center, _ in fitted])
    ties = np.array([count for _, count in fitted])
    return np.vstack((centers[0], np.median(centers, axis=0))), ties


def max_pairwise(centers: np.ndarray) -> float:
    return float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def simulate(sigma: float, out: Path, prefix: str) -> tuple[list[dict], list[dict]]:
    trials = []
    phase_trials = []
    reference_cloud = None
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        initial_center = rng.uniform(*CFG["initial_center_phase_box_km"], size=2)
        first = sample_cloud(rng, initial_center, sigma)
        first_est, first_ties = estimates(first)
        per_phase = np.array([estimates(first, shift)[0] for shift in SHIFTS])
        for m, method in enumerate(METHODS):
            phase_trials.append({"seed": seed, "method": method, "max_span_km": max_pairwise(per_phase[:, m])})
        sectors = CFG["direction_sectors"]
        angles = (np.arange(sectors) + rng.uniform(size=sectors)) * (2 * np.pi / sectors)
        for distance in CFG["displacements_km"]:
            scenarios = [(-1, 0.0)] if distance == 0 else list(enumerate(angles))
            for sector, angle in scenarios:
                truth = distance * np.array([math.cos(angle), math.sin(angle)])
                second_center = initial_center + truth
                second = sample_cloud(rng, second_center, sigma)
                second_est, second_ties = estimates(second)
                for m, method in enumerate(METHODS):
                    delta = second_est[m] - first_est[m]
                    length = float(np.linalg.norm(delta))
                    trials.append({
                        "seed": seed, "truth_distance_km": distance, "sector": sector,
                        "angle_deg": "" if sector < 0 else math.degrees(angle), "method": method,
                        "initial_truth_x_km": initial_center[0], "initial_truth_y_km": initial_center[1],
                        "truth_dx_km": truth[0], "truth_dy_km": truth[1],
                        "initial_est_x_km": first_est[m, 0], "initial_est_y_km": first_est[m, 1],
                        "followup_est_x_km": second_est[m, 0], "followup_est_y_km": second_est[m, 1],
                        "est_dx_km": delta[0], "est_dy_km": delta[1], "est_length_km": length,
                        "vector_error_km": float(np.linalg.norm(delta - truth)),
                        "length_bias_km": length - distance,
                        "reported": int(length >= CFG["reporting_threshold_km"]),
                        "tied_origins_initial": int((first_ties[:1 if m == 0 else 4] > 1).sum()),
                        "tied_origins_followup": int((second_ties[:1 if m == 0 else 4] > 1).sum()),
                    })
                if seed == SEEDS[0] and distance == 1 and sector == 0:
                    reference_cloud = (first, second, initial_center, second_center, first_est, second_est)
    write_csv(out / f"{prefix}_trials.csv", trials)
    write_csv(out / f"{prefix}_origin_sensitivity.csv", phase_trials)
    if prefix == "main" and reference_cloud is not None:
        first, second, center_a, center_b, est_a, est_b = reference_cloud
        rows = [{"period": period, "x_km": x, "y_km": y,
                 "generated_component": "hotspot" if i < CFG["n_hotspot"] else "background"}
                for period, points in ((0, first), (1, second)) for i, (x, y) in enumerate(points)]
        write_csv(out / "example_points.csv", rows)
        (out / "example_centers.json").write_text(json.dumps({
            "seed": SEEDS[0], "truth_distance_km": 1, "sector": 0,
            "truth": [center_a.tolist(), center_b.tolist()],
            "single": [est_a[0].tolist(), est_b[0].tolist()],
            "four": [est_a[1].tolist(), est_b[1].tolist()],
        }, indent=2) + "\n", encoding="utf-8")
    return trials, phase_trials


def paired_ci(by_seed: np.ndarray, bootstrap_indices: np.ndarray) -> list[float]:
    means = by_seed[bootstrap_indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(trials: list[dict], phases: list[dict], draws: np.ndarray) -> dict:
    distance_results = []
    for distance in CFG["displacements_km"]:
        mrows = [[row for row in trials if row["method"] == method and row["truth_distance_km"] == distance]
                 for method in METHODS]
        assert [(r["seed"], r["sector"]) for r in mrows[0]] == [(r["seed"], r["sector"]) for r in mrows[1]]
        result = {"truth_distance_km": distance, "n_pairs": len(mrows[0])}
        for method, rows in zip(METHODS, mrows):
            errs = [row["vector_error_km"] for row in rows]
            result[method] = {
                "mean_vector_error_km": float(np.mean(errs)), "p90_vector_error_km": float(np.quantile(errs, 0.9)),
                "reported_count": int(sum(row["reported"] for row in rows)),
                "report_rate": float(np.mean([row["reported"] for row in rows])),
                "mean_length_bias_km": float(np.mean([row["length_bias_km"] for row in rows])),
            }
        for key in ("vector_error_km", "reported"):
            diff = np.array([b[key] - a[key] for a, b in zip(*mrows)])
            per_seed = diff.reshape(len(SEEDS), -1).mean(axis=1)
            result[f"delta_{key}"] = {"estimate": float(per_seed.mean()), "ci95": paired_ci(per_seed, draws)}
        if distance:
            result["by_sector_mean_vector_error_km"] = [{
                "sector": sector, **{method: float(np.mean([r["vector_error_km"] for r in rows if r["sector"] == sector]))
                                     for method, rows in zip(METHODS, mrows)}
            } for sector in range(CFG["direction_sectors"])]
        distance_results.append(result)

    phase_spans = [np.array([row["max_span_km"] for row in phases if row["method"] == method]) for method in METHODS]
    phase_result = {method: {"mean_km": float(x.mean()), "median_km": float(np.median(x)),
                             "p90_km": float(np.quantile(x, 0.9))} for method, x in zip(METHODS, phase_spans)}
    phase_diff = phase_spans[1] - phase_spans[0]
    phase_result["delta_mean_km"] = {"estimate": float(phase_diff.mean()), "ci95": paired_ci(phase_diff, draws)}
    return {"distance": distance_results, "phase_sensitivity": phase_result}


def main() -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    draws = rng.integers(0, len(SEEDS), size=(CFG["bootstrap_repeats"], len(SEEDS)))
    output = {"numpy_version": np.__version__, "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
              "scenarios": {}}
    for prefix, sigma in (("main", CFG["hotspot_sigma_km"]), ("wide", CFG["wider_hotspot_sigma_km"])):
        trials, phases = simulate(sigma, out, prefix)
        output["scenarios"][prefix] = {"sigma_km": sigma, **summarize(trials, phases, draws)}
    (out / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for scenario, result in output["scenarios"].items():
        print(scenario, [(row["truth_distance_km"], row["n_pairs"],
                         [round(row[m]["mean_vector_error_km"], 4) for m in METHODS],
                         [row[m]["report_rate"] for m in METHODS]) for row in result["distance"]])
        print("origin-span mean:", [round(result["phase_sensitivity"][m]["mean_km"], 4) for m in METHODS])


if __name__ == "__main__":
    main()

