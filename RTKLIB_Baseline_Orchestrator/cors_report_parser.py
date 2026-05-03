
from __future__ import annotations

from pathlib import Path
import math
import re
import pandas as pd
from models import CorsSolution


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def _find_final_solution_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if {"Coordinate", "Source column", "Final mean value"}.issubset(cols):
            return df
    raise ValueError("Could not find final GNSS station coordinate solution table in report.")


def _find_daily_solution_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    required = {"run_label", "dataset_name", "station_id", "X_m", "Y_m", "Z_m"}
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if required.issubset(cols):
            return df
    return None


def _extract_coordinate(final_df: pd.DataFrame, coordinate: str, column: str) -> tuple[float, float | None]:
    rows = final_df[
        (final_df["Coordinate"].astype(str).str.strip().str.lower() == coordinate.lower())
        | (final_df["Source column"].astype(str).str.strip() == column)
    ]

    if rows.empty:
        raise ValueError(f"Coordinate row not found in final solution table: {coordinate} / {column}")

    row = rows.iloc[0]
    mean_value = _to_float(row.get("Final mean value"))
    std_value = _to_float(row.get("Standard deviation"))

    if mean_value is None:
        raise ValueError(f"Final mean value is empty for coordinate {coordinate}")

    return mean_value, std_value


def parse_cors_report(report_path: str | Path) -> CorsSolution:
    report_path = Path(report_path).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    tables = pd.read_html(str(report_path))
    final_df = _find_final_solution_table(tables)
    daily_df = _find_daily_solution_table(tables)

    x, sx = _extract_coordinate(final_df, "X", "X_m")
    y, sy = _extract_coordinate(final_df, "Y", "Y_m")
    z, sz = _extract_coordinate(final_df, "Z", "Z_m")

    station_id = ""
    n_solutions = None

    if daily_df is not None and "station_id" in daily_df.columns and len(daily_df) > 0:
        station_values = daily_df["station_id"].dropna().astype(str).unique()
        if len(station_values) > 0:
            station_id = station_values[0]

    if "Number of daily/per-file solutions" in final_df.columns:
        vals = pd.to_numeric(final_df["Number of daily/per-file solutions"], errors="coerce").dropna()
        if len(vals) > 0:
            n_solutions = int(vals.iloc[0])

    return CorsSolution(
        station_id=station_id,
        X_m=float(x),
        Y_m=float(y),
        Z_m=float(z),
        std_X_m=sx,
        std_Y_m=sy,
        std_Z_m=sz,
        n_solutions=n_solutions,
        source_report_path=report_path,
        final_table=final_df,
        daily_table=daily_df,
    )
