"""Compare all four fixed-origin baselines using the original main trials."""

import csv
import json
import math

import numpy as np

from experiment import CFG, ORIGINS, ROOT, SEEDS, peak, sample_cloud


def positions(points):
    single = np.array([peak(points, origin)[0] for origin in ORIGINS])
    return np.vstack([single, np.median(single, axis=0)])


def main():
    with (ROOT / "results" / "main_trials.csv").open(encoding="utf-8", newline="") as handle:
        reference = {(int(r["seed"]), int(r["truth_distance_km"]), int(r["sector"]), r["method"]): r
                     for r in csv.DictReader(handle)}
    trials = {distance: [] for distance in CFG["displacements_km"]}
    sectors = CFG["direction_sectors"]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        start = rng.uniform(*CFG["initial_center_phase_box_km"], size=2)
        first = positions(sample_cloud(rng, start, CFG["hotspot_sigma_km"]))
        angles = (np.arange(sectors) + rng.uniform(size=sectors)) * (2 * np.pi / sectors)
        for distance, group in trials.items():
            directions = [(-1, 0.0)] if distance == 0 else enumerate(angles)
            for sector, angle in directions:
                truth = distance * np.array([math.cos(angle), math.sin(angle)])
                second = positions(sample_cloud(rng, start + truth, CFG["hotspot_sigma_km"]))
                delta = second - first
                for idx, name in ((0, "single"), (4, "four")):
                    saved = reference[seed, distance, sector, name]
                    np.testing.assert_allclose(delta[idx], [float(saved["est_dx_km"]), float(saved["est_dy_km"])], atol=1e-12, rtol=0)
                group.append((np.linalg.norm(delta - truth, axis=1), np.linalg.norm(delta, axis=1)))
    output = {"same_saved_pairs_verified": len(reference), "order": [origin.tolist() for origin in ORIGINS] + ["four-origin median"], "groups": []}
    for distance, group in trials.items():
        errors = np.array([e for e, _ in group])
        lengths = np.array([length for _, length in group])
        output["groups"].append({"truth_km": distance, "pairs": len(group), "mean_error_km": errors.mean(axis=0).tolist(),
                                 "p90_error_km": np.quantile(errors, 0.9, axis=0).tolist(),
                                 "reports_at_1km": (lengths >= CFG["reporting_threshold_km"]).sum(axis=0).tolist()})
    result = ROOT / "results" / "origin_baseline_review.json"
    result.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
