"""
Pipeline orchestrator — runs the full 6-step daily routine.
"""

import logging
from datetime import datetime

from .paths import get_paths
from .update_data import update_lottery_data
from .accuracy import check_accuracy
from .windows import calculate_windows
from .transitions import calculate_transition_weights
from .predict import generate_predictions

_LOGGER = logging.getLogger(__name__)


def run_full_pipeline(root_dir: str, data_url: str) -> dict:
    """
    Execute the full daily pipeline:
      1. Update CSV with latest draws
      2. Check accuracy of yesterday's predictions
      3. Recalculate window distributions
      4. Recalculate transition weights
      5+6. Generate predictions (auto-retrains if needed) + top picks

    Returns a combined result dict for sensor consumption.
    """
    paths = get_paths(root_dir)
    result = {"timestamp": datetime.now().isoformat(), "steps": {}}

    # Step 1 — Update data
    _LOGGER.info("Pipeline step 1/6: Updating lottery data...")
    try:
        update_result = update_lottery_data(paths["data"], data_url)
        result["steps"]["update_data"] = update_result
    except Exception as exc:
        _LOGGER.error("Step 1 failed: %s", exc)
        result["steps"]["update_data"] = {"error": str(exc)}

    # Step 2 — Check accuracy
    _LOGGER.info("Pipeline step 2/6: Checking accuracy...")
    try:
        accuracy_result = check_accuracy(paths)
        result["steps"]["accuracy"] = (
            accuracy_result if accuracy_result else {"info": "no draw yesterday"}
        )
    except Exception as exc:
        _LOGGER.error("Step 2 failed: %s", exc)
        result["steps"]["accuracy"] = {"error": str(exc)}

    # Step 3 — Calculate windows
    _LOGGER.info("Pipeline step 3/6: Calculating windows...")
    try:
        windows_result = calculate_windows(paths["data"])
        result["steps"]["windows"] = {
            "windows_count": len(windows_result.get("windows", [])),
            "coverage_pct": windows_result.get("coverage_pct"),
        }
    except Exception as exc:
        _LOGGER.error("Step 3 failed: %s", exc)
        result["steps"]["windows"] = {"error": str(exc)}

    # Step 4 — Calculate transition weights
    _LOGGER.info("Pipeline step 4/6: Calculating transition weights...")
    try:
        trans_result = calculate_transition_weights(paths["data"])
        result["steps"]["transitions"] = {
            "current_window": trans_result.get("current_window"),
            "top_follower": (
                trans_result["top5_followers"][0]
                if trans_result.get("top5_followers")
                else None
            ),
        }
    except Exception as exc:
        _LOGGER.error("Step 4 failed: %s", exc)
        result["steps"]["transitions"] = {"error": str(exc)}

    # Step 5+6 — Predict + top picks
    _LOGGER.info("Pipeline step 5-6/6: Generating predictions...")
    try:
        pred_result = generate_predictions(paths)
        result["predictions"] = pred_result
    except Exception as exc:
        _LOGGER.error("Step 5-6 failed: %s", exc)
        result["predictions"] = {"error": str(exc)}

    _LOGGER.info("Pipeline complete")
    return result
