"""Calibrate AIS movement reporting, then test it on independent simulation pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

from experiment import CFG, ORIGINS, ROOT, SEEDS, peak, sample_cloud, write_csv


CONFIG_PATH = ROOT / "closed_loop_protocol.json"
SETTINGS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
METHODS = SETTINGS["methods"]
PHASES = ("calibration", "validation")
OUT = ROOT / "results"
FROZEN = OUT / "closed_loop_thresholds.json"
BASELINE_COMPARISONS = ("single_2km", "single_1km", "four_mean")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config():
    used = set(SEEDS)
    for phase in PHASES:
        for cohort in SETTINGS[phase]:
            seeds = set(range(cohort["start_seed"], cohort["start_seed"] + cohort["n"]))
            if used & seeds:
                raise ValueError("Simulation seeds overlap")
            used |= seeds
    if not 0 < SETTINGS["calibration_alpha"] < SETTINGS["validation_false_report_limit"] < 1:
        raise ValueError("Calibration alpha must leave headroom below the validation limit")
    if SETTINGS["minimum_estimated_km"] != CFG["reporting_threshold_km"]:
        raise ValueError("The original minimum reportable distance has changed")


def fit(points):
    peaks = np.array([peak(points, origin)[0] for origin in ORIGINS])
    fine_grid = peak(points, ORIGINS[0], width=1.0)[0]
    return np.vstack((peaks[0], fine_grid, peaks.mean(axis=0), np.median(peaks, axis=0)))


def iter_pairs(phase):
    distances = (0,) if phase == "calibration" else SETTINGS["distances_km"]
    for cohort in SETTINGS[phase]:
        sigma = cohort["sigma_km"]
        start = cohort["start_seed"]
        for seed in range(start, start + cohort["n"]):
            rng = np.random.default_rng(seed)
            center = rng.uniform(*CFG["initial_center_phase_box_km"], size=2)
            initial = fit(sample_cloud(rng, center, sigma))
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([math.cos(angle), math.sin(angle)])
            for distance in distances:
                truth = distance * direction
                followup = fit(sample_cloud(rng, center + truth, sigma))
                yield sigma, seed, distance, angle, truth, followup - initial
        print(f"{phase} sigma={sigma:g}: {cohort['n']} seeds", flush=True)


def magnitude(delta):
    return np.round(np.linalg.norm(delta, axis=-1), SETTINGS["score_decimal_places"])


def calibrate():
    rows = []
    groups = defaultdict(list)
    for sigma, seed, _, _, _, delta in iter_pairs("calibration"):
        for method, score in zip(METHODS, magnitude(delta)):
            groups[sigma, method].append(float(score))
            rows.append(dict(sigma_km=sigma, seed=seed, method=method, est_length_km=score))
    path = OUT / "closed_loop_calibration.csv"
    write_csv(path, rows)
    thresholds = {}
    for (sigma, method), scores in groups.items():
        n = len(scores)
        rank = math.ceil((n + 1) * (1 - SETTINGS["calibration_alpha"]))
        if rank > n:
            raise ValueError("Calibration sample too small for the requested alpha")
        threshold = float(np.partition(scores, rank - 1)[rank - 1])
        thresholds.setdefault(method, {})[str(sigma)] = {
            "n": n,
            "rank": rank,
            "threshold_km": threshold,
            "at_threshold": sum(score == threshold for score in scores),
            "above_threshold": sum(score > threshold for score in scores),
        }
    frozen = {
        "protocol_sha256": digest(CONFIG_PATH),
        "source_protocol_sha256": digest(ROOT / "protocol.json"),
        "calibration_csv_sha256": digest(path),
        "calibration_alpha": SETTINGS["calibration_alpha"],
        "thresholds": thresholds,
        "shared_by_method_km": {m: max(v["threshold_km"] for v in d.values()) for m, d in thresholds.items()},
    }
    save_json(FROZEN, frozen)
    print(f"Frozen thresholds: {frozen['shared_by_method_km']}", flush=True)


def binomial_bounds(k, n, tail):
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return [lower, upper]


def summarize(rows, thresholds):
    groups = defaultdict(list)
    for row in rows:
        key = row["sigma_km"], row["truth_distance_km"], row["method"]
        groups[key].append(row)

    tail = (1 - SETTINGS["family_confidence"]) / len(SETTINGS["validation"])
    draw_rng = np.random.default_rng(SETTINGS["paired_bootstrap_seed"])
    cohort_results = []
    for cohort in SETTINGS["validation"]:
        sigma = cohort["sigma_km"]
        n = cohort["n"]
        for distance in SETTINGS["distances_km"]:
            section = {"sigma_km": sigma, "truth_distance_km": distance, "pairs": n, "methods": {}}
            vectors = {}
            for method in METHODS:
                group = groups[sigma, distance, method]
                errors = np.asarray([r["vector_error_km"] for r in group])
                assert len(errors) == n
                vectors[method] = errors
                rates = {}
                for name in ("fixed", "calibrated"):
                    k = sum(r[f"reported_{name}"] for r in group)
                    rates[name] = {"reported": k, "rate": k / n, "bounds": binomial_bounds(k, n, tail)}
                section["methods"][method] = {
                    "mean_vector_error_km": float(errors.mean()),
                    "p90_vector_error_km": float(np.quantile(errors, .90)),
                    **rates,
                }
            if distance in (1, 2):
                draws = draw_rng.integers(0, n, (SETTINGS["paired_bootstrap_reps"], n), dtype=np.int32)
                section["four_median_comparisons"] = {}
                for method in BASELINE_COMPARISONS:
                    delta = vectors["four_median"] - vectors[method]
                    boot = delta[draws].mean(axis=1)
                    section["four_median_comparisons"][method] = {
                        "difference_km": float(delta.mean()),
                        "ci95": np.quantile(boot, [.025, .975]).tolist(),
                    }
            cohort_results.append(section)
    decision = {}
    for method in METHODS:
        nulls = [r["methods"][method]["calibrated"] for r in cohort_results if r["truth_distance_km"] == 0]
        false_report_ok = all(x["bounds"][1] <= SETTINGS["validation_false_report_limit"] for x in nulls)
        accepted_distances = []
        for distance in SETTINGS["distances_km"]:
            if distance < SETTINGS["minimum_estimated_km"]:
                continue
            alts = [r["methods"][method]["calibrated"] for r in cohort_results if r["truth_distance_km"] == distance]
            if false_report_ok and all(x["bounds"][0] >= SETTINGS["validation_detection_target"] for x in alts):
                accepted_distances.append(distance)
        decision[method] = {"false_report_ok": false_report_ok, "accepted_tested_distances_km": accepted_distances}
    return {"tail_probability_per_width": tail, "shared_thresholds_km": thresholds,
            "decision": decision, "groups": cohort_results}


def validate():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != digest(CONFIG_PATH) or frozen["source_protocol_sha256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Protocol has changed since calibration")
    if frozen["calibration_csv_sha256"] != digest(OUT / "closed_loop_calibration.csv"):
        raise ValueError("Calibration records differ from the frozen threshold file")
    thresholds = frozen["shared_by_method_km"]
    minimum = SETTINGS["minimum_estimated_km"]
    rows = []
    for sigma, seed, distance, angle, truth, delta in iter_pairs("validation"):
        for i, (method, score) in enumerate(zip(METHODS, magnitude(delta))):
            rows.append(dict(
                sigma_km=sigma, seed=seed, truth_distance_km=distance, angle_rad=angle,
                method=method, truth_dx_km=truth[0], truth_dy_km=truth[1],
                est_dx_km=delta[i, 0], est_dy_km=delta[i, 1],
                est_length_km=score, vector_error_km=float(np.linalg.norm(delta[i] - truth)),
                reported_fixed=int(score >= minimum),
                reported_calibrated=int(score >= minimum and score > thresholds[method]),
            ))
    path = OUT / "closed_loop_validation.csv"
    write_csv(path, rows)
    result = summarize(rows, thresholds)
    result.update(thresholds_sha256=digest(FROZEN), validation_csv_sha256=digest(path))
    save_json(OUT / "closed_loop_summary.json", result)
    print(json.dumps(result["decision"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    options = parser.parse_args()
    validate_config()
    OUT.mkdir(exist_ok=True)
    if options.phase == "calibration":
        calibrate()
    else:
        validate()


if __name__ == "__main__":
    main()

