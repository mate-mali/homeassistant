"""
Step 1 — Download latest Mini Lotto results from multipasko.pl.
Returns dict with update status.
"""

import io
import logging
from datetime import datetime

import pandas as pd
import requests

_LOGGER = logging.getLogger(__name__)


def update_lottery_data(data_dir, url: str) -> dict:
    """
    Download latest draws and append to CSV.
    Returns {"new_draws": int, "total_draws": int, "latest_date": str}.
    """
    csv_path = data_dir / "minilotto_2008_onwards.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Base CSV not found at {csv_path}")

    df_existing = pd.read_csv(csv_path)
    last_row = df_existing.iloc[-1]
    last_date = datetime(
        int(last_row["Rok"]), int(last_row["Miesiac"]), int(last_row["Dzien"])
    )

    _LOGGER.info(
        "Current data: %d draws, latest: %s",
        len(df_existing),
        last_date.strftime("%d/%m/%Y"),
    )

    try:
        response = requests.get(url, timeout=15)
        response.encoding = "utf-8"
        response.raise_for_status()
    except requests.RequestException as exc:
        _LOGGER.warning("Network error fetching data: %s", exc)
        return {
            "new_draws": 0,
            "total_draws": len(df_existing),
            "latest_date": last_date.strftime("%d/%m/%Y"),
            "error": str(exc),
        }

    df_latest = pd.read_csv(io.StringIO(response.text), sep=";")
    new_rows = []

    if all(c in df_latest.columns for c in ("Dzien", "Miesiac", "Rok", "L1", "L5")):
        for _, row in df_latest.iterrows():
            try:
                draw_date = datetime(
                    int(row["Rok"]), int(row["Miesiac"]), int(row["Dzien"])
                )
                if draw_date > last_date:
                    new_rows.append(
                        {
                            "Numer": int(
                                row.get("Numer", len(df_existing) + len(new_rows) + 1)
                            ),
                            "Dzien": int(row["Dzien"]),
                            "Miesiac": int(row["Miesiac"]),
                            "Rok": int(row["Rok"]),
                            "L1": int(row["L1"]),
                            "L2": int(row["L2"]),
                            "L3": int(row["L3"]),
                            "L4": int(row["L4"]),
                            "L5": int(row["L5"]),
                        }
                    )
            except (ValueError, KeyError):
                continue

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df_updated = pd.concat([df_existing, new_df], ignore_index=True)
        df_updated = df_updated.drop_duplicates(
            subset=["Dzien", "Miesiac", "Rok"], keep="last"
        )
        df_updated.to_csv(csv_path, index=False)
        latest = df_updated.iloc[-1]
        latest_date_str = (
            f"{int(latest['Dzien'])}/{int(latest['Miesiac'])}/{int(latest['Rok'])}"
        )
        _LOGGER.info("Added %d new draws. Total: %d", len(new_rows), len(df_updated))
    else:
        latest_date_str = last_date.strftime("%d/%m/%Y")
        _LOGGER.info("No new draws — data is current")

    return {
        "new_draws": len(new_rows),
        "total_draws": len(df_existing) + len(new_rows),
        "latest_date": latest_date_str,
    }
