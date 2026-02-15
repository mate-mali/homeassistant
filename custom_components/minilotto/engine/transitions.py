"""
Step 4 — Markov transition weight calculation.
Computes what window is likely to follow the current one.
"""

import json
import logging
from collections import Counter
from datetime import datetime

import pandas as pd

_LOGGER = logging.getLogger(__name__)

TOP_K = 5
FOLLOWER_SHARE = 0.70
BASELINE_SHARE = 0.30


def build_transition_matrix(window_series):
    """Return {src: {dst: count}} transition counts."""
    matrix = {}
    for i in range(len(window_series) - 1):
        src = int(window_series[i])
        dst = int(window_series[i + 1])
        matrix.setdefault(src, Counter())[dst] += 1
    return matrix


def transition_weights_for(current_window: int, matrix: dict, all_windows: list) -> dict:
    """Smoothed weight dict for all_windows given the current window."""
    followers = matrix.get(current_window, {})
    total = sum(followers.values())

    if total == 0:
        n = len(all_windows)
        return {str(w): round(1.0 / n, 6) for w in all_windows}

    raw = {dst: cnt / total for dst, cnt in followers.items()}
    top_k = sorted(raw.items(), key=lambda x: -x[1])[:TOP_K]
    top_k_windows = {w for w, _ in top_k}
    top_k_prob_sum = sum(p for _, p in top_k)

    weights = {}
    non_top = [w for w in all_windows if w not in top_k_windows]

    for w, prob in top_k:
        weights[str(w)] = round(FOLLOWER_SHARE * (prob / top_k_prob_sum), 6)

    if non_top:
        per_w = BASELINE_SHARE / len(non_top)
        for w in non_top:
            weights[str(w)] = round(per_w, 6)
    else:
        bump = BASELINE_SHARE / len(top_k)
        for w, _ in top_k:
            weights[str(w)] = round(weights[str(w)] + bump, 6)

    return weights


def calculate_transition_weights(data_dir) -> dict:
    """
    Build transition matrix and compute today's weights.
    Returns the full output dict (saved as transition_matrix.json).
    """
    ws_path = data_dir / "window_size_distribution.csv"
    if not ws_path.exists():
        raise FileNotFoundError(f"{ws_path} not found — run windows step first")

    ws_df = pd.read_csv(ws_path)
    window_series = ws_df["window_size"].values

    cfg_path = data_dir / "window_weights.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    all_windows = cfg["windows"]

    current_window = int(window_series[-1])
    matrix = build_transition_matrix(window_series)
    weights = transition_weights_for(current_window, matrix, all_windows)

    followers_raw = matrix.get(current_window, {})
    total = sum(followers_raw.values())
    top5 = []
    if total:
        for dst, cnt in sorted(followers_raw.items(), key=lambda x: -x[1])[:TOP_K]:
            top5.append(
                {
                    "window": int(dst),
                    "count": int(cnt),
                    "probability": round(cnt / total, 4),
                }
            )

    _LOGGER.info(
        "Current window: %d, top follower: W%d (%.0f%%)",
        current_window,
        top5[0]["window"] if top5 else 0,
        (top5[0]["probability"] * 100) if top5 else 0,
    )

    matrix_json = {}
    for src, dsts in matrix.items():
        matrix_json[str(src)] = {str(d): int(c) for d, c in dsts.items()}

    output = {
        "generated": datetime.now().isoformat(),
        "current_window": current_window,
        "top_k": TOP_K,
        "follower_share": FOLLOWER_SHARE,
        "baseline_share": BASELINE_SHARE,
        "top5_followers": top5,
        "current_weights": weights,
        "full_matrix": matrix_json,
    }

    out_path = data_dir / "transition_matrix.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _LOGGER.info("Transition weights saved")
    return output
