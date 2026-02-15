"""
Step 2 — Check accuracy of yesterday's predictions against actual draw.
Returns accuracy dict or None if no draw yesterday.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta

import pandas as pd

_LOGGER = logging.getLogger(__name__)


def _find_prediction_file(pred_dir, date_compact: str):
    candidates = sorted(pred_dir.glob(f"*_{date_compact}.json"))
    return candidates[-1] if candidates else None


def _analyse_model(pred_file, actual_numbers: list[int]) -> dict:
    with open(pred_file) as f:
        data = json.load(f)

    preds = data["predictions"]
    ranks = []
    details = {}
    for num in actual_numbers:
        entry = preds.get(str(num))
        if entry:
            ranks.append(entry["rank"])
            details[str(num)] = {"rank": entry["rank"], "score": entry["score"]}

    stats = (
        {
            "min": min(ranks),
            "max": max(ranks),
            "median": statistics.median(ranks),
            "mean": round(statistics.mean(ranks), 2),
            "count": len(ranks),
        }
        if ranks
        else {}
    )

    return {
        "winning_number_ranks": ranks,
        "number_details": details,
        "statistics": stats,
        "prediction_file": pred_file.name,
    }


def check_accuracy(paths: dict) -> dict | None:
    """
    Compare yesterday's predictions with actual draw.
    Saves accuracy JSON and returns the result dict, or None if no draw.
    """
    yesterday = datetime.now() - timedelta(days=1)
    y_str = yesterday.strftime("%Y-%m-%d")
    y_compact = yesterday.strftime("%Y%m%d")
    y_display = yesterday.strftime("%d/%m/%Y")

    csv_path = paths["data"] / "minilotto_2008_onwards.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    row = df[
        (df["Dzien"] == yesterday.day)
        & (df["Miesiac"] == yesterday.month)
        & (df["Rok"] == yesterday.year)
    ]
    if row.empty:
        _LOGGER.info("No draw found for %s — nothing to check", y_display)
        return None

    actual = [int(row.iloc[0][c]) for c in ["L1", "L2", "L3", "L4", "L5"]]
    _LOGGER.info("Actual draw %s: %s", y_display, actual)

    result = {
        "date": y_display,
        "actual_numbers": actual,
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    all_ranks = []
    variants = [
        ("lightgbm_standard", paths["predictions_standard"]),
        ("lightgbm_weighted", paths["predictions_weighted"]),
        ("lightgbm_transition", paths["predictions_transition"]),
        ("lightgbm_mp_window", paths["predictions_mp_window"]),
    ]

    for name, pred_dir in variants:
        pred_file = _find_prediction_file(pred_dir, y_compact)
        if pred_file is None:
            continue
        info = _analyse_model(pred_file, actual)
        result["models"][name] = info
        all_ranks.extend(info["winning_number_ranks"])
        _LOGGER.info(
            "%s ranks=%s mean=%s",
            name,
            info["winning_number_ranks"],
            info["statistics"].get("mean", "?"),
        )

    if all_ranks:
        result["ensemble"] = {
            "all_ranks": all_ranks,
            "statistics": {
                "min": min(all_ranks),
                "max": max(all_ranks),
                "median": statistics.median(all_ranks),
                "mean": round(statistics.mean(all_ranks), 2),
                "total": len(all_ranks),
            },
        }

    out = paths["accuracy"] / f"accuracy_{y_str}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    _LOGGER.info("Saved accuracy: %s", out.name)

    return result
