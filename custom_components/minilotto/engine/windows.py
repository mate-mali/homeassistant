"""
Step 3 — Window analysis & weight calculation.
Computes completion windows, frequency weights, and per-number lookup.
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

TOP_N = 40


def _find_completion_window(draws_matrix, start_idx: int) -> int | None:
    seen = set()
    for width in range(1, start_idx + 2):
        row = start_idx - width + 1
        if row < 0:
            return None
        for val in draws_matrix[row]:
            seen.add(val)
        if len(seen) == 42:
            return width
    return None


def calculate_windows(data_dir) -> dict:
    """
    Run full window analysis.
    Returns {"windows": [...], "weights": {...}, "coverage_pct": float}.
    """
    csv_path = data_dir / "minilotto_2008_onwards.csv"
    df = pd.read_csv(csv_path)
    draws = df[["L1", "L2", "L3", "L4", "L5"]].values.astype(int)
    n_draws = len(draws)

    _LOGGER.info("Computing completion windows for %d draws...", n_draws)

    window_sizes = []
    for i in range(n_draws):
        w = _find_completion_window(draws, i)
        if w is not None:
            window_sizes.append({"draw_index": i, "window_size": w})

    ws_df = pd.DataFrame(window_sizes)
    ws_df.to_csv(data_dir / "window_size_distribution.csv", index=False)

    freq = ws_df["window_size"].value_counts().sort_values(ascending=False)
    total = len(ws_df)

    top_windows = sorted(freq.head(TOP_N).index.tolist())
    top_counts = {int(w): int(freq[w]) for w in top_windows}
    count_sum = sum(top_counts.values())
    weights = {str(w): round(top_counts[w] / count_sum, 6) for w in top_windows}
    coverage = 100 * count_sum / total

    _LOGGER.info(
        "Selected %d windows (%d..%d), coverage %.1f%%",
        TOP_N,
        top_windows[0],
        top_windows[-1],
        coverage,
    )

    # Per-number frequency lookup
    _LOGGER.info("Computing per-number frequency lookup...")
    freq_rows = []
    for window in top_windows:
        num_counts = {n: 0 for n in range(1, 43)}
        num_appearances = {n: 0 for n in range(1, 43)}
        valid = 0
        for start_idx in range(window - 1, n_draws):
            valid += 1
            window_draws = draws[start_idx - window + 1 : start_idx + 1]
            for num in range(1, 43):
                c = int(np.sum(window_draws == num))
                num_counts[num] += c
                if c > 0:
                    num_appearances[num] += 1
        for num in range(1, 43):
            avg_freq = num_counts[num] / valid if valid else 0
            app_rate = num_appearances[num] / valid if valid else 0
            freq_rows.append(
                {
                    "window_size": window,
                    "number": num,
                    "avg_frequency": round(avg_freq, 4),
                    "appearance_rate_pct": round(app_rate * 100, 2),
                }
            )

    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(data_dir / "window_frequency_lookup.csv", index=False)

    config = {
        "generated": datetime.now().isoformat(),
        "total_draws": int(total),
        "top_n": TOP_N,
        "windows": top_windows,
        "weights": weights,
        "coverage_pct": round(coverage, 2),
        "statistics": {
            "min_window": int(freq.index.min()),
            "max_window": int(freq.index.max()),
            "mean_window": round(float(ws_df["window_size"].mean()), 2),
            "median_window": int(ws_df["window_size"].median()),
        },
    }
    with open(data_dir / "window_weights.json", "w") as f:
        json.dump(config, f, indent=2)

    _LOGGER.info("Window analysis complete")
    return config
