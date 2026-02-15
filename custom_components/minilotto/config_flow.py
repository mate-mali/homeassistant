"""Config flow for Mini Lotto integration."""

import logging
import os

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    DOMAIN,
    CONF_DATA_URL,
    CONF_SCAN_INTERVAL_HOURS,
    CONF_NOTIFY_SERVICE,
    DEFAULT_DATA_URL,
    DEFAULT_SCAN_INTERVAL_HOURS,
)

_LOGGER = logging.getLogger(__name__)


class MiniLottoConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the config flow for Mini Lotto."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step — user clicks 'Add Integration'."""
        errors = {}

        if user_input is not None:
            # Validate the data directory is writable
            data_dir = user_input.get("data_dir", "/config/minilotto")
            try:
                os.makedirs(data_dir, exist_ok=True)
            except OSError:
                errors["data_dir"] = "cannot_create_dir"

            if not errors:
                return self.async_create_entry(
                    title="Mini Lotto Predictor",
                    data=user_input,
                )

        data_schema = vol.Schema(
            {
                vol.Required("data_dir", default="/config/minilotto"): str,
                vol.Required(CONF_DATA_URL, default=DEFAULT_DATA_URL): str,
                vol.Required(
                    CONF_SCAN_INTERVAL_HOURS,
                    default=DEFAULT_SCAN_INTERVAL_HOURS,
                ): vol.All(int, vol.Range(min=1, max=24)),
                vol.Optional(CONF_NOTIFY_SERVICE, default=""): str,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return MiniLottoOptionsFlow(config_entry)


class MiniLottoOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Mini Lotto."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = self._config_entry.data

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SCAN_INTERVAL_HOURS,
                        default=current.get(
                            CONF_SCAN_INTERVAL_HOURS, DEFAULT_SCAN_INTERVAL_HOURS
                        ),
                    ): vol.All(int, vol.Range(min=1, max=24)),
                    vol.Optional(
                        CONF_NOTIFY_SERVICE,
                        default=current.get(CONF_NOTIFY_SERVICE, ""),
                    ): str,
                }
            ),
        )
