"""Constants for the Mini Lotto integration."""

DOMAIN = "minilotto"
CONF_DATA_URL = "data_url"
CONF_SCAN_INTERVAL_HOURS = "scan_interval_hours"
CONF_NOTIFY_SERVICE = "notify_service"

DEFAULT_DATA_URL = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto"
DEFAULT_SCAN_INTERVAL_HOURS = 6

ATTR_TOP_PICKS = "top_picks"
ATTR_ENSEMBLE = "ensemble"
ATTR_LAST_DRAW = "last_draw"
ATTR_LAST_DRAW_DATE = "last_draw_date"
ATTR_ACCURACY = "accuracy"
ATTR_MODEL_DETAILS = "model_details"
ATTR_NEXT_DRAW = "next_draw"
ATTR_PREDICTIONS_DATE = "predictions_date"

SERVICE_RUN_PIPELINE = "run_pipeline"
SERVICE_RETRAIN = "retrain_models"
