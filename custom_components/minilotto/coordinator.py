"""
Data Update Coordinator for Mini Lotto.
Runs the pipeline on a schedule and provides data to sensors.
"""

import asyncio
import logging
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN
from .engine.pipeline import run_full_pipeline

_LOGGER = logging.getLogger(__name__)


class MiniLottoCoordinator(DataUpdateCoordinator):
    """Coordinator that runs the full Mini Lotto pipeline on schedule."""

    def __init__(
        self,
        hass: HomeAssistant,
        root_dir: str,
        data_url: str,
        scan_interval_hours: int,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(hours=scan_interval_hours),
        )
        self._root_dir = root_dir
        self._data_url = data_url
        self._last_pipeline_result: dict | None = None

    async def _async_update_data(self) -> dict:
        """Run the pipeline in an executor to avoid blocking the event loop."""
        try:
            result = await self.hass.async_add_executor_job(
                run_full_pipeline, self._root_dir, self._data_url
            )
            self._last_pipeline_result = result
            return result
        except Exception as exc:
            raise UpdateFailed(f"Pipeline failed: {exc}") from exc

    async def async_run_pipeline_now(self) -> dict:
        """Force an immediate pipeline run (called from services)."""
        return await self._async_update_data()

    @property
    def last_result(self) -> dict | None:
        return self._last_pipeline_result
