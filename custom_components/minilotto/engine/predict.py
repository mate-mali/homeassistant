"""
Step 5b — Daily predictions (4 variants).
Step 6 — Top picks & ensemble.
"""

import json
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from .training import needs_retraining, train_models

_LOGGER = logging.getLogger(__name__)


def _load_freq_lookup(data_dir):
    freq_df = pd.read_csv(data_dir / "window_frequency_lookup.csv")
    lookup = {}
    for _, row in freq_df.iterrows():
        key = (int(row["window_size"]), int(row["number"]))
        lookup[key] = {
            "avg": row["avg_frequency"],
            "appearance_rate": row["appearance_rate_pct"] / 100,
        }
    return lookup


def _build_feature_row(draws, idx, windows, freq_lookup, weights=None):
    features = []
    for window in windows:
        start = idx - window + 1
        if start < 0:
            features.extend([0.0] * 42 * 5)
            continue
        window_draws = draws[start : idx + 1]
        w = weights[str(window)] if weights else 1.0
        for num in range(1, 43):
            count = int(np.sum(window_draws == num))
            recent = 1.0 if num in draws[idx] else 0.0
            hist = freq_lookup.get((window, num), {})
            hist_avg = hist.get("avg", 3.5)
            app_rate = hist.get("appearance_rate", 0.96)
            deviation = count - hist_avg
            ratio = count / (hist_avg + 0.01) if hist_avg > 0 else float(count)
            features.extend([count * w, recent, deviation * w, ratio * w, app_rate])
    return features


def _predict_one(model_path, draws, freq_lookup, variant_weights=None):
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    windows = data["windows"]
    models = data["models"]
    scaler = data["scaler"]
    model_name = data["model"]
    weights = data.get("weights", variant_weights)

    idx = len(draws) - 1
    row = _build_feature_row(draws, idx, windows, freq_lookup, weights=weights)
    X = np.array(row, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)

    scores = np.zeros(42)
    for num in range(1, 43):
        scores[num - 1] = models[num].predict_proba(X_scaled)[0, 1]
    return scores, windows, model_name


def _save_predictions(scores, windows, model_name, out_dir):
    ranking = np.argsort(-scores)
    predictions = {}
    for num in range(1, 43):
        rank = int(np.where(ranking == num - 1)[0][0]) + 1
        predictions[str(num)] = {
            "number": num,
            "score": round(float(scores[num - 1]), 6),
            "rank": rank,
        }

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{model_name}_{date_str}.json"
    filepath = out_dir / filename

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "windows": windows,
        "predictions": predictions,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    return filepath, ranking, predictions


def generate_predictions(paths: dict) -> dict:
    """
    Auto-retrain if needed, then generate all 4 prediction variants.
    Returns a dict with top picks and ensemble for sensor consumption.
    """
    data_dir = paths["data"]

    # Auto-retrain if stale
    retrained = False
    if needs_retraining(paths):
        _LOGGER.info("Models stale — auto-retraining...")
        train_models(paths)
        retrained = True

    df = pd.read_csv(data_dir / "minilotto_2008_onwards.csv")
    draws = df[["L1", "L2", "L3", "L4", "L5"]].values.astype(int)
    freq_lookup = _load_freq_lookup(data_dir)

    # Load transition weights
    trans_weights = None
    trans_data = None
    trans_path = data_dir / "transition_matrix.json"
    if trans_path.exists():
        with open(trans_path) as f:
            trans_data = json.load(f)
        trans_weights = trans_data["current_weights"]

    # MP window weights
    mp_window_weights = None
    if trans_data:
        mp_window_id = trans_data["top5_followers"][0]["window"]
        trans_pkl = paths["models_transition"] / "lightgbm_transition.pkl"
        if trans_pkl.exists():
            with open(trans_pkl, "rb") as f:
                _td = pickle.load(f)
            mp_window_weights = {
                str(w): (1.0 if w == mp_window_id else 0.0) for w in _td["windows"]
            }

    variants = [
        (
            paths["models_standard"] / "lightgbm_standard.pkl",
            None,
            paths["predictions_standard"],
            "lightgbm_standard",
        ),
        (
            paths["models_weighted"] / "lightgbm_weighted.pkl",
            None,
            paths["predictions_weighted"],
            "lightgbm_weighted",
        ),
        (
            paths["models_transition"] / "lightgbm_transition.pkl",
            trans_weights,
            paths["predictions_transition"],
            "lightgbm_transition",
        ),
        (
            paths["models_transition"] / "lightgbm_transition.pkl",
            mp_window_weights,
            paths["predictions_mp_window"],
            "lightgbm_mp_window",
        ),
    ]

    all_scores = {}
    model_results = {}

    for model_path, extra_w, out_dir, variant_name in variants:
        if not model_path.exists():
            _LOGGER.warning("Model %s not found — skipping", model_path.name)
            continue

        scores, windows, model_name = _predict_one(
            model_path, draws, freq_lookup, variant_weights=extra_w
        )

        # Override for mp_window
        actual_name = variant_name
        filepath, ranking, predictions = _save_predictions(
            scores, windows, actual_name, out_dir
        )

        # Top 5 for this variant
        top5 = []
        for i in range(5):
            num = int(ranking[i]) + 1
            top5.append(
                {"number": num, "score": round(float(scores[num - 1]), 4), "rank": i + 1}
            )

        model_results[actual_name] = {
            "file": filepath.name,
            "top5": top5,
        }

        # Accumulate for ensemble
        for v in predictions.values():
            num = v["number"]
            all_scores.setdefault(num, []).append(v["score"])

        _LOGGER.info("%s top 5: %s", actual_name, [t["number"] for t in top5])

    # Ensemble
    ensemble_top5 = []
    if all_scores:
        avg = {num: sum(s) / len(s) for num, s in all_scores.items()}
        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        for i, (num, sc) in enumerate(ranked[:5], 1):
            ensemble_top5.append(
                {"number": num, "score": round(sc, 4), "rank": i}
            )

    # Get last draw info
    last_row = df.iloc[-1]
    last_draw = [int(last_row[c]) for c in ["L1", "L2", "L3", "L4", "L5"]]
    last_draw_date = (
        f"{int(last_row['Dzien'])}/{int(last_row['Miesiac'])}/{int(last_row['Rok'])}"
    )

    result = {
        "timestamp": datetime.now().isoformat(),
        "predictions_date": datetime.now().strftime("%Y-%m-%d"),
        "retrained": retrained,
        "models": model_results,
        "ensemble_top5": ensemble_top5,
        "ensemble_numbers": [t["number"] for t in ensemble_top5],
        "last_draw": last_draw,
        "last_draw_date": last_draw_date,
        "total_draws": len(draws),
    }

    return result
