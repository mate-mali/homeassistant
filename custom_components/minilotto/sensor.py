"""
Sensor entities for Mini Lotto integration.
Exposes prediction data, accuracy, and pipeline status to HA.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN

if TYPE_CHECKING:
    from .coordinator import MiniLottoCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mini Lotto sensors from a config entry."""
    from .coordinator import MiniLottoCoordinator  # noqa: E402 — lazy import

    coordinator: MiniLottoCoordinator = hass.data[DOMAIN][entry.entry_id]

    sensors = [
        MiniLottoEnsembleSensor(coordinator, entry),
        MiniLottoStandardSensor(coordinator, entry),
        MiniLottoWeightedSensor(coordinator, entry),
        MiniLottoTransitionSensor(coordinator, entry),
        MiniLottoMpWindowSensor(coordinator, entry),
        MiniLottoAccuracySensor(coordinator, entry),
        MiniLottoLastDrawSensor(coordinator, entry),
        MiniLottoPipelineStatusSensor(coordinator, entry),
    ]

    async_add_entities(sensors, True)


class MiniLottoBaseSensor(CoordinatorEntity, SensorEntity):
    """Base class for Mini Lotto sensors."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: MiniLottoCoordinator,
        entry: ConfigEntry,
        key: str,
        name: str,
        icon: str = "mdi:clover",
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_name = name
        self._attr_icon = icon
        self._key = key

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, "minilotto_predictor")},
            "name": "Mini Lotto Predictor",
            "manufacturer": "Custom",
            "model": "LightGBM Ensemble",
            "sw_version": "1.0.0",
        }


class MiniLottoEnsembleSensor(MiniLottoBaseSensor):
    """Shows the ensemble top 5 picks as the main sensor."""

    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "ensemble", "Ensemble Top 5", "mdi:star")

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        if not self.coordinator.data:
            return None
        preds = self.coordinator.data.get("predictions", {})
        numbers = preds.get("ensemble_numbers", [])
        return ", ".join(str(n) for n in numbers) if numbers else None

    @property
    def extra_state_attributes(self) -> dict:
        if not self.coordinator.data:
            return {}
        preds = self.coordinator.data.get("predictions", {})
        return {
            "top_picks": preds.get("ensemble_top5", []),
            "predictions_date": preds.get("predictions_date"),
            "total_draws": preds.get("total_draws"),
            "retrained": preds.get("retrained", False),
        }


class _ModelVariantSensor(MiniLottoBaseSensor):
    """Generic sensor for a single model variant."""

    def __init__(self, coordinator, entry, variant_key: str, name: str):
        super().__init__(coordinator, entry, variant_key, name, "mdi:dice-multiple")
        self._variant_key = variant_key

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        if not self.coordinator.data:
            return None
        preds = self.coordinator.data.get("predictions", {})
        models = preds.get("models", {})
        model_data = models.get(self._variant_key, {})
        top5 = model_data.get("top5", [])
        return ", ".join(str(t["number"]) for t in top5) if top5 else None

    @property
    def extra_state_attributes(self) -> dict:
        if not self.coordinator.data:
            return {}
        preds = self.coordinator.data.get("predictions", {})
        models = preds.get("models", {})
        model_data = models.get(self._variant_key, {})
        return {
            "top5": model_data.get("top5", []),
            "file": model_data.get("file"),
        }


class MiniLottoStandardSensor(_ModelVariantSensor):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "lightgbm_standard", "Standard Top 5")


class MiniLottoWeightedSensor(_ModelVariantSensor):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "lightgbm_weighted", "Weighted Top 5")


class MiniLottoTransitionSensor(_ModelVariantSensor):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "lightgbm_transition", "Transition Top 5")


class MiniLottoMpWindowSensor(_ModelVariantSensor):
    def __init__(self, coordinator, entry):
        super().__init__(coordinator, entry, "lightgbm_mp_window", "MP Window Top 5")


class MiniLottoAccuracySensor(MiniLottoBaseSensor):
    """Shows the latest accuracy check result."""

    def __init__(self, coordinator, entry):
        super().__init__(
            coordinator, entry, "accuracy", "Accuracy", "mdi:target"
        )

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        if not self.coordinator.data:
            return None
        acc = self.coordinator.data.get("steps", {}).get("accuracy", {})
        if "error" in acc or "info" in acc:
            return acc.get("info", acc.get("error"))
        ensemble = acc.get("ensemble", {})
        stats = ensemble.get("statistics", {})
        mean = stats.get("mean")
        return f"Mean rank: {mean}" if mean else None

    @property
    def extra_state_attributes(self) -> dict:
        if not self.coordinator.data:
            return {}
        acc = self.coordinator.data.get("steps", {}).get("accuracy", {})
        return {
            "date": acc.get("date"),
            "actual_numbers": acc.get("actual_numbers"),
            "models": {
                name: {
                    "ranks": m.get("winning_number_ranks"),
                    "mean": m.get("statistics", {}).get("mean"),
                }
                for name, m in acc.get("models", {}).items()
            },
            "ensemble_mean": acc.get("ensemble", {})
            .get("statistics", {})
            .get("mean"),
        }


class MiniLottoLastDrawSensor(MiniLottoBaseSensor):
    """Shows the last drawn numbers."""

    def __init__(self, coordinator, entry):
        super().__init__(
            coordinator, entry, "last_draw", "Last Draw", "mdi:numeric"
        )

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        if not self.coordinator.data:
            return None
        preds = self.coordinator.data.get("predictions", {})
        draw = preds.get("last_draw", [])
        return ", ".join(str(n) for n in draw) if draw else None

    @property
    def extra_state_attributes(self) -> dict:
        if not self.coordinator.data:
            return {}
        preds = self.coordinator.data.get("predictions", {})
        return {
            "date": preds.get("last_draw_date"),
            "numbers": preds.get("last_draw"),
            "total_draws": preds.get("total_draws"),
        }


class MiniLottoPipelineStatusSensor(MiniLottoBaseSensor):
    """Shows pipeline run status and timing."""

    def __init__(self, coordinator, entry):
        super().__init__(
            coordinator,
            entry,
            "pipeline_status",
            "Pipeline Status",
            "mdi:pipe",
        )

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        if not self.coordinator.data:
            return "idle"
        steps = self.coordinator.data.get("steps", {})
        errors = [k for k, v in steps.items() if isinstance(v, dict) and "error" in v]
        if errors:
            return f"errors: {', '.join(errors)}"
        return "ok"

    @property
    def extra_state_attributes(self) -> dict:
        if not self.coordinator.data:
            return {}
        return {
            "last_run": self.coordinator.data.get("timestamp"),
            "steps": {
                k: "error" if isinstance(v, dict) and "error" in v else "ok"
                for k, v in self.coordinator.data.get("steps", {}).items()
            },
            "retrained": self.coordinator.data.get("predictions", {}).get(
                "retrained", False
            ),
        }
