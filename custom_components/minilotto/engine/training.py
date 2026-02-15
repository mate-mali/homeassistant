"""
Step 5a — LightGBM Training (3 variants: standard, weighted, transition).
Runs automatically if models are stale (not trained this calendar month).
"""

import json
import logging
import pickle
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .transitions import build_transition_matrix, transition_weights_for

_LOGGER = logging.getLogger(__name__)


def _load_config(data_dir):
    cfg_path = data_dir / "window_weights.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found")
    with open(cfg_path) as f:
        cfg = json.load(f)

    freq_df = pd.read_csv(data_dir / "window_frequency_lookup.csv")
    freq_lookup = {}
    for _, row in freq_df.iterrows():
        key = (int(row["window_size"]), int(row["number"]))
        freq_lookup[key] = {
            "avg": row["avg_frequency"],
            "appearance_rate": row["appearance_rate_pct"] / 100,
        }
    return cfg, freq_lookup


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


def _train_variant(X_train, y_train, X_test, y_test, label):
    models = {}
    for num in range(1, 43):
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=12,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train[:, num - 1])
        models[num] = clf

    score = (
        sum(models[n].score(X_test, y_test[:, n - 1]) for n in range(1, 43)) / 42
    )
    _LOGGER.info("%s average test accuracy: %.4f", label, score)
    return models, score


def needs_retraining(paths: dict) -> bool:
    """Return True if models haven't been trained this calendar month."""
    ref = paths["models_standard"] / "lightgbm_standard.pkl"
    if not ref.exists():
        return True
    model_mtime = datetime.fromtimestamp(ref.stat().st_mtime)
    now = datetime.now()
    return (model_mtime.year, model_mtime.month) != (now.year, now.month)


def train_models(paths: dict) -> dict:
    """
    Train all 3 LightGBM variants. Returns accuracy dict.
    This is CPU-intensive (~5-15 minutes).
    """
    t0 = time.time()
    data_dir = paths["data"]
    cfg, freq_lookup = _load_config(data_dir)
    windows = cfg["windows"]
    weights = cfg["weights"]

    df = pd.read_csv(data_dir / "minilotto_2008_onwards.csv")
    draws = df[["L1", "L2", "L3", "L4", "L5"]].values.astype(int)
    n = len(draws)
    max_w = max(windows)

    ws_df = pd.read_csv(data_dir / "window_size_distribution.csv")
    window_series = ws_df["window_size"].values
    trans_matrix = build_transition_matrix(window_series)

    _LOGGER.info("Building feature matrices (%d rows)...", n - max_w - 1)

    X_std, X_wgt, X_trn, Y = [], [], [], []
    for idx in range(max_w, n - 1):
        next_draw = draws[idx + 1]
        target = np.zeros(42)
        for num in next_draw:
            target[int(num) - 1] = 1
        Y.append(target)

        X_std.append(_build_feature_row(draws, idx, windows, freq_lookup, weights=None))
        X_wgt.append(
            _build_feature_row(draws, idx, windows, freq_lookup, weights=weights)
        )

        draw_window = (
            int(window_series[idx])
            if idx < len(window_series)
            else int(window_series[-1])
        )
        tw = transition_weights_for(draw_window, trans_matrix, windows)
        X_trn.append(_build_feature_row(draws, idx, windows, freq_lookup, weights=tw))

    X_std = np.array(X_std, dtype=np.float32)
    X_wgt = np.array(X_wgt, dtype=np.float32)
    X_trn = np.array(X_trn, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    split = int(0.8 * len(X_std))

    scaler_std = StandardScaler()
    X_std_tr = scaler_std.fit_transform(X_std[:split])
    X_std_te = scaler_std.transform(X_std[split:])

    scaler_wgt = StandardScaler()
    X_wgt_tr = scaler_wgt.fit_transform(X_wgt[:split])
    X_wgt_te = scaler_wgt.transform(X_wgt[split:])

    scaler_trn = StandardScaler()
    X_trn_tr = scaler_trn.fit_transform(X_trn[:split])
    X_trn_te = scaler_trn.transform(X_trn[split:])

    Y_train, Y_test = Y[:split], Y[split:]

    _LOGGER.info("Training STANDARD...")
    models_std, acc_std = _train_variant(
        X_std_tr, Y_train, X_std_te, Y_test, "STANDARD"
    )
    _save_pkl(
        paths["models_standard"] / "lightgbm_standard.pkl",
        "lightgbm_standard",
        models_std,
        scaler_std,
        windows,
        acc_std,
    )

    _LOGGER.info("Training WEIGHTED...")
    models_wgt, acc_wgt = _train_variant(
        X_wgt_tr, Y_train, X_wgt_te, Y_test, "WEIGHTED"
    )
    _save_pkl(
        paths["models_weighted"] / "lightgbm_weighted.pkl",
        "lightgbm_weighted",
        models_wgt,
        scaler_wgt,
        windows,
        acc_wgt,
        weights=weights,
    )

    _LOGGER.info("Training TRANSITION...")
    models_trn, acc_trn = _train_variant(
        X_trn_tr, Y_train, X_trn_te, Y_test, "TRANSITION"
    )
    _save_pkl(
        paths["models_transition"] / "lightgbm_transition.pkl",
        "lightgbm_transition",
        models_trn,
        scaler_trn,
        windows,
        acc_trn,
    )

    elapsed = time.time() - t0
    _LOGGER.info("Training complete in %.0fs", elapsed)

    return {
        "standard_accuracy": round(acc_std, 4),
        "weighted_accuracy": round(acc_wgt, 4),
        "transition_accuracy": round(acc_trn, 4),
        "elapsed_seconds": round(elapsed, 1),
    }


def _save_pkl(path, model_name, models, scaler, windows, accuracy, weights=None):
    data = {
        "model": model_name,
        "models": models,
        "scaler": scaler,
        "windows": windows,
        "accuracy": float(accuracy),
        "trained": datetime.now().isoformat(),
    }
    if weights:
        data["weights"] = weights
    with open(path, "wb") as f:
        pickle.dump(data, f)
