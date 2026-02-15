# Mini Lotto Predictor — Home Assistant Integration

A HACS-ready custom integration that runs a daily Mini Lotto prediction pipeline
using LightGBM models with window-based features and Markov transition analysis.

## What It Does

Every N hours (configurable, default 6), the integration automatically:

1. **Downloads** latest draw results from multipasko.pl
2. **Checks accuracy** of yesterday's predictions against the actual draw
3. **Recalculates** completion window distributions
4. **Updates** Markov transition weights
5. **Generates predictions** from 4 model variants (auto-retrains monthly)
6. **Produces an ensemble** top-5 picks

Results are exposed as Home Assistant sensors and optionally sent as notifications.

## Sensors Created

| Sensor | Description |
|--------|-------------|
| `sensor.mini_lotto_ensemble_top_5` | Main ensemble picks (comma-separated) |
| `sensor.mini_lotto_standard_top_5` | Standard model picks |
| `sensor.mini_lotto_weighted_top_5` | Weighted model picks |
| `sensor.mini_lotto_transition_top_5` | Transition model picks |
| `sensor.mini_lotto_mp_window_top_5` | MP Window model picks |
| `sensor.mini_lotto_accuracy` | Yesterday's accuracy (mean rank) |
| `sensor.mini_lotto_last_draw` | Latest drawn numbers |
| `sensor.mini_lotto_pipeline_status` | Pipeline health status |

Each sensor includes detailed attributes (scores, ranks, model details).

## Services

| Service | Description |
|---------|-------------|
| `minilotto.run_pipeline` | Force-run the full pipeline now |
| `minilotto.retrain_models` | Force retrain all models, then predict |

## Installation

### HACS (Recommended)

1. Add this repository as a custom repository in HACS
2. Install "Mini Lotto Predictor"
3. Restart Home Assistant
4. Go to Settings → Integrations → Add → "Mini Lotto"

### Manual

1. Copy `custom_components/minilotto/` to your HA `config/custom_components/` folder
2. Place your `minilotto_2008_onwards.csv` in `/config/minilotto/data/`
   - OR place it in `custom_components/minilotto/seed_data/` before first run
3. Restart Home Assistant
4. Add the integration via Settings → Integrations

## Initial Data Setup

The integration needs the historical CSV file to work. You have two options:

### Option A: Seed Data (automatic)
Place `minilotto_2008_onwards.csv` in:
```
config/custom_components/minilotto/seed_data/minilotto_2008_onwards.csv
```
It will be auto-copied on first run.

### Option B: Manual Placement
Place it directly in your configured data directory:
```
config/minilotto/data/minilotto_2008_onwards.csv
```

If you also have pre-trained models, copy the `trained_models/` folder too.
Otherwise, models will be trained automatically on first run (~5-15 minutes).

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| Data directory | `/config/minilotto` | Where all data/models/predictions are stored |
| Data URL | multipasko.pl CSV | Source for lottery results |
| Scan interval | 6 hours | How often the pipeline runs |
| Notify service | *(empty)* | e.g. `notify.mobile_app_your_phone` |

## Example Automation

```yaml
automation:
  - alias: "Mini Lotto Daily Run"
    trigger:
      - platform: time
        at: "09:00:00"
    action:
      - service: minilotto.run_pipeline
```

## Example Dashboard Card

```yaml
type: entities
title: Mini Lotto Predictions
entities:
  - entity: sensor.mini_lotto_ensemble_top_5
    name: Today's Picks
  - entity: sensor.mini_lotto_accuracy
    name: Yesterday's Accuracy
  - entity: sensor.mini_lotto_last_draw
    name: Last Draw
  - entity: sensor.mini_lotto_pipeline_status
    name: Pipeline Status
```

## Requirements

The integration installs these Python packages via HA:
- pandas, numpy, lightgbm, scikit-learn, requests
