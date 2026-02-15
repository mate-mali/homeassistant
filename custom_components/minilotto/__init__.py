"""
Mini Lotto integration for Home Assistant.
Runs a daily prediction pipeline and exposes results as sensors.
"""

import logging
import shutil
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from .const import (
    DOMAIN,
    CONF_DATA_URL,
    CONF_SCAN_INTERVAL_HOURS,
    CONF_NOTIFY_SERVICE,
    DEFAULT_DATA_URL,
    DEFAULT_SCAN_INTERVAL_HOURS,
    SERVICE_RUN_PIPELINE,
    SERVICE_RETRAIN,
)

# NOTE: Do NOT import coordinator or engine modules at module level.
# Heavy dependencies (pandas, lightgbm, etc.) would be loaded when HA
# imports this package to show the config flow, before requirements
# are guaranteed to be installed.

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Mini Lotto from a config entry."""
    from .coordinator import MiniLottoCoordinator  # noqa: E402 — lazy import

    hass.data.setdefault(DOMAIN, {})

    data_dir = entry.data.get("data_dir", "/config/minilotto")
    data_url = entry.data.get(CONF_DATA_URL, DEFAULT_DATA_URL)
    scan_hours = entry.data.get(CONF_SCAN_INTERVAL_HOURS, DEFAULT_SCAN_INTERVAL_HOURS)
    notify_service = entry.data.get(CONF_NOTIFY_SERVICE, "")

    # Ensure data directory has the base CSV
    await hass.async_add_executor_job(_ensure_data_dir, data_dir)

    coordinator = MiniLottoCoordinator(hass, data_dir, data_url, scan_hours)

    # Initial data fetch
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up sensors
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    async def handle_run_pipeline(call: ServiceCall) -> None:
        _LOGGER.info("Manual pipeline run triggered via service")
        result = await coordinator.async_run_pipeline_now()
        await _send_notification(hass, notify_service, result)

    async def handle_retrain(call: ServiceCall) -> None:
        _LOGGER.info("Manual retrain triggered via service")
        from .engine.paths import get_paths
        from .engine.training import train_models

        paths = get_paths(data_dir)
        await hass.async_add_executor_job(train_models, paths)
        _LOGGER.info("Retrain complete — running predictions...")
        await coordinator.async_run_pipeline_now()

    hass.services.async_register(DOMAIN, SERVICE_RUN_PIPELINE, handle_run_pipeline)
    hass.services.async_register(DOMAIN, SERVICE_RETRAIN, handle_retrain)

    # Send notification after first successful run
    if notify_service and coordinator.data:
        await _send_notification(hass, notify_service, coordinator.data)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    # Unregister services if no entries left
    if not hass.data[DOMAIN]:
        hass.services.async_remove(DOMAIN, SERVICE_RUN_PIPELINE)
        hass.services.async_remove(DOMAIN, SERVICE_RETRAIN)

    return unload_ok


def _ensure_data_dir(data_dir: str) -> None:
    """Create data directory structure. Copy base CSV if it exists in the integration folder."""
    root = Path(data_dir)
    for sub in [
        "data",
        "trained_models/standard",
        "trained_models/weighted",
        "trained_models/transition",
        "predictions/standard",
        "predictions/weighted",
        "predictions/transition",
        "predictions/mp_window",
        "accuracy",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Check if base CSV exists; if not, check integration's seed_data folder
    csv_dest = root / "data" / "minilotto_2008_onwards.csv"
    if not csv_dest.exists():
        seed = Path(__file__).parent / "seed_data" / "minilotto_2008_onwards.csv"
        if seed.exists():
            shutil.copy2(seed, csv_dest)
            _LOGGER.info("Copied seed CSV to %s", csv_dest)
        else:
            _LOGGER.warning(
                "No base CSV found at %s — place minilotto_2008_onwards.csv in %s/data/",
                seed,
                data_dir,
            )


async def _send_notification(
    hass: HomeAssistant, notify_service: str, result: dict
) -> None:
    """Send a notification with today's predictions."""
    if not notify_service:
        return

    preds = result.get("predictions", {})
    ensemble = preds.get("ensemble_top5", [])
    if not ensemble:
        return

    numbers = ", ".join(f"#{t['number']}" for t in ensemble)
    scores = ", ".join(f"{t['score']:.3f}" for t in ensemble)
    last_draw = preds.get("last_draw", [])
    last_date = preds.get("last_draw_date", "?")

    # Check accuracy
    acc = result.get("steps", {}).get("accuracy", {})
    acc_text = ""
    if acc and "ensemble" in acc:
        mean_rank = acc["ensemble"]["statistics"].get("mean", "?")
        acc_text = f"\nYesterday's accuracy: mean rank {mean_rank}/42"

    retrained = preds.get("retrained", False)
    retrain_text = "\n(Models were retrained)" if retrained else ""

    message = (
        f"Mini Lotto Predictions\n"
        f"Ensemble: {numbers}\n"
        f"Scores: {scores}\n"
        f"Last draw ({last_date}): {', '.join(str(n) for n in last_draw)}"
        f"{acc_text}{retrain_text}"
    )

    try:
        # notify_service could be "notify.mobile_app_phone"
        domain, service = notify_service.split(".", 1)
        await hass.services.async_call(
            domain,
            service,
            {"title": "Mini Lotto Picks", "message": message},
            blocking=False,
        )
    except Exception as exc:
        _LOGGER.warning("Failed to send notification: %s", exc)
