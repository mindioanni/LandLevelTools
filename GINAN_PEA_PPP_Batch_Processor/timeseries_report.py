from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import base64
import html
import io
import math
import re
import sys
import pandas as pd
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


PLOT_COLUMN_MAP = {
    "X": "X_m",
    "Y": "Y_m",
    "Z": "Z_m",
    "lon": "lon_deg",
    "lat": "lat_deg",
    "h": "h_m",
    "X_m": "X_m",
    "Y_m": "Y_m",
    "Z_m": "Z_m",
    "lon_deg": "lon_deg",
    "lat_deg": "lat_deg",
    "h_m": "h_m",
}


def _load_default_config() -> dict:
    try:
        import paths_config
        return paths_config.get_default_config()
    except Exception:
        return {}


def _load_default_convergence_config() -> dict:
    try:
        import position_timeseries
        return dict(position_timeseries.DEFAULT_CONVERGENCE_CONFIG)
    except Exception:
        return {}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"

    return str(value)


def _fmt_m(value) -> str:
    return _fmt(value, digits=4)


def _fmt_deg(value) -> str:
    return _fmt(value, digits=10)


def _fmt_sci(value, digits: int = 4) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    try:
        value = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(value):
        return ""

    return f"{value:.{digits}e}"


def _value_from_row(df: pd.DataFrame, column: str, default=""):
    if column not in df.columns or len(df) == 0:
        return default

    value = df[column].iloc[0]

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return value


def _metadata_value(metadata: dict, key: str, default=""):
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def _infer_sampling_interval_sec(df: pd.DataFrame):
    if "run_label" not in df.columns:
        return None

    for value in df["run_label"].dropna().astype(str):
        match = re.search(r"_([0-9]+)s$", value)
        if match:
            return int(match.group(1))

    return None


def _find_representative_yaml(timeseries_path: Path, df: pd.DataFrame) -> Path | None:
    # New layout: RAW_ROOT/GINAN_process/timeseries.out and RAW_ROOT/yaml/*.yaml.
    # Legacy fallback: GINAN_process/yaml/*.yaml.
    candidate_yaml_dirs = [
        timeseries_path.parent.parent / "yaml",
        timeseries_path.parent / "yaml",
    ]

    labels = []
    if "run_label" in df.columns:
        labels = [str(v) for v in df["run_label"].dropna().tolist()]

    for yaml_dir in candidate_yaml_dirs:
        if not yaml_dir.exists():
            continue

        for label in labels:
            candidates = sorted(yaml_dir.glob(f"{label}*.yaml"))
            if candidates:
                return candidates[0]

        candidates = sorted(yaml_dir.glob("*.yaml"))
        if candidates:
            return candidates[0]

    return None


def _read_text_if_exists(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _infer_provider_series_project_from_yaml(yaml_text: str) -> dict:
    result = {
        "provider": "",
        "project": "",
        "series": "",
        "source": "",
    }

    valid_suffixes = (
        "_ORB.SP3",
        "_SP3.SP3",
        "_CLK.CLK",
        "_OSB.BIA",
        "_BIA.BIA",
        "_ERP.ERP",
    )

    pattern = re.compile(
        r"\b([A-Z0-9]{3})0([A-Z0-9]{3})([A-Z0-9]{3})_"
        r"\d{11}_[0-9A-Z]+_[0-9A-Z]+_[A-Z0-9]+\.[A-Z0-9]+\b"
    )

    for match in pattern.finditer(yaml_text.upper()):
        filename = match.group(0)

        if filename.endswith(".SNX") or "_CRD.SNX" in filename:
            continue

        if not filename.endswith(valid_suffixes):
            continue

        result["provider"] = match.group(1)
        result["project"] = match.group(2)
        result["series"] = match.group(3)
        result["source"] = f"generated YAML PPP product filename: {filename}"

        return result

    return result


def _infer_epoch_interval_from_yaml(yaml_text: str):
    match = re.search(
        r"^\s*epoch_interval\s*:\s*['\"]?([0-9.]+)['\"]?",
        yaml_text,
        flags=re.MULTILINE,
    )

    if not match:
        return None

    value = float(match.group(1))
    if value.is_integer():
        return int(value)

    return value


def _infer_resampled_rinex_path_from_yaml(yaml_text: str) -> Path | None:
    obs_root = ""
    rnx_input = ""

    match_root = re.search(
        r"^\s*gnss_observations_root\s*:\s*(.*?)\s*(?:#.*)?$",
        yaml_text,
        flags=re.MULTILINE,
    )
    if match_root:
        obs_root = match_root.group(1).strip().strip("'\"")

    in_rnx_block = False
    for line in yaml_text.splitlines():
        if re.match(r"^\s*rnx_inputs\s*:\s*(?:#.*)?$", line):
            in_rnx_block = True
            continue

        if in_rnx_block:
            if line.strip().startswith("-"):
                rnx_input = line.strip().lstrip("-").strip().strip("'\"")
                break

            if line.strip() and not line.startswith(" "):
                break

    if obs_root and rnx_input:
        candidate = Path(obs_root) / rnx_input
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def _infer_raw_root_from_resampled_path(resampled_path: Path | None) -> tuple[str, str]:
    if resampled_path is None:
        return "", ""

    parts = list(resampled_path.parts)

    for folder_name in ("RESAMPLED", "RESAMPLED"):
        if folder_name in parts:
            idx = parts.index(folder_name)
            base = Path(*parts[:idx])

            # New layout: RAW_ROOT/RESAMPLED/<dataset>.rnx/<file>
            if base.exists() and base.is_dir():
                return (
                    str(base.resolve()),
                    f"inferred from generated YAML RINEX path and RAW_ROOT/{folder_name} layout",
                )

            # Legacy fallback: <GNSS_ROOT>/METRICA/RESAMPLED and <GNSS_ROOT>/METRICA/RAW
            candidate = base / "RAW"
            if candidate.exists() and candidate.is_dir():
                return (
                    str(candidate.resolve()),
                    f"inferred from generated YAML RINEX path and sibling {folder_name}/RAW layout",
                )

    return "", ""


def _infer_raw_root_from_ginan_process_dir(ginan_process_dir: Path) -> tuple[str, str]:
    # New layout: RAW_ROOT/GINAN_process
    candidate = ginan_process_dir.parent

    if (candidate / "GINAN_process").resolve(strict=False) == ginan_process_dir.resolve(strict=False):
        return (
            str(candidate.resolve()),
            "inferred from GINAN_process parent directory: RAW_ROOT/GINAN_process",
        )

    # Legacy fallback: <GNSS_ROOT>/METRICA/RAW
    legacy_candidate = candidate / "METRICA" / "RAW"
    if legacy_candidate.exists() and legacy_candidate.is_dir():
        return (
            str(legacy_candidate.resolve()),
            "inferred from legacy GINAN_process sibling folder: ../METRICA/RAW",
        )

    return "", ""


def _resolve_processing_metadata(
    timeseries_path: Path,
    df: pd.DataFrame,
    metadata: dict | None,
) -> dict:
    metadata = dict(metadata or {})

    default_config = _load_default_config()
    user_inputs = default_config.get("user_inputs", {})
    system = default_config.get("system", {})

    representative_yaml = _find_representative_yaml(timeseries_path, df)
    yaml_text = _read_text_if_exists(representative_yaml)

    yaml_product_info = _infer_provider_series_project_from_yaml(yaml_text)
    yaml_epoch_interval = _infer_epoch_interval_from_yaml(yaml_text)
    resampled_rinex_path = _infer_resampled_rinex_path_from_yaml(yaml_text)

    inferred_raw_root_from_rinex, inferred_raw_root_from_rinex_source = _infer_raw_root_from_resampled_path(
        resampled_rinex_path
    )

    inferred_raw_root_from_layout, inferred_raw_root_from_layout_source = _infer_raw_root_from_ginan_process_dir(
        timeseries_path.parent
    )

    provider = _metadata_value(
        metadata,
        "provider",
        yaml_product_info.get("provider") or user_inputs.get("provider", ""),
    )

    series = _metadata_value(
        metadata,
        "series",
        yaml_product_info.get("series") or user_inputs.get("series", ""),
    )

    project = _metadata_value(
        metadata,
        "project",
        yaml_product_info.get("project") or user_inputs.get("project", ""),
    )

    sampling_interval_sec = _metadata_value(metadata, "sampling_interval_sec", None)
    if sampling_interval_sec is None:
        sampling_interval_sec = yaml_epoch_interval
    if sampling_interval_sec is None:
        sampling_interval_sec = _infer_sampling_interval_sec(df)
    if sampling_interval_sec is None:
        sampling_interval_sec = user_inputs.get("requested_sample_rate_sec", "")

    raw_root_default = (
        inferred_raw_root_from_rinex
        or user_inputs.get("raw_root", "")
        or inferred_raw_root_from_layout
    )

    raw_root = _metadata_value(
        metadata,
        "raw_root",
        raw_root_default,
    )

    if metadata.get("raw_root"):
        raw_root_source = "explicit metadata"
    elif raw_root == inferred_raw_root_from_rinex and raw_root:
        raw_root_source = inferred_raw_root_from_rinex_source
    elif raw_root == user_inputs.get("raw_root", "") and raw_root:
        raw_root_source = "paths_config user_inputs.raw_root"
    elif raw_root == inferred_raw_root_from_layout and raw_root:
        raw_root_source = inferred_raw_root_from_layout_source
    else:
        raw_root_source = "not resolved"

    ginan_process_dir = _metadata_value(
        metadata,
        "ginan_process_dir",
        str(timeseries_path.parent),
    )

    pea_executable = _metadata_value(
        metadata,
        "pea_executable",
        system.get("pea_path", ""),
    )

    template_yaml_path = _metadata_value(
        metadata,
        "template_yaml_path",
        system.get("template_yaml_path", ""),
    )

    sources = {
        "provider_series_project": yaml_product_info.get("source") or "paths_config/defaults or explicit metadata",
        "sampling_interval": "generated YAML epoch_interval" if yaml_epoch_interval is not None else "run_label/paths_config or explicit metadata",
        "raw_root": raw_root_source,
        "ginan_process_dir": "timeseries.out parent directory or explicit metadata",
        "pea_executable": "paths_config system.pea_path or explicit metadata",
        "template_yaml_path": "paths_config system.template_yaml_path or explicit metadata",
        "representative_yaml": str(representative_yaml) if representative_yaml else "",
    }

    return {
        "provider": provider,
        "series": series,
        "project": project,
        "sampling_interval_sec": sampling_interval_sec,
        "raw_root": raw_root,
        "ginan_process_dir": ginan_process_dir,
        "pea_executable": pea_executable,
        "template_yaml_path": template_yaml_path,
        "sources": sources,
    }


def _qc_flag_definitions() -> list[str]:
    return [
        "OK / no flags: no QC warning flag was assigned.",
        "TRACE_WARNINGS_PRESENT: one or more warning lines were found in scanned stdout/TRACE logs.",
        "TRACE_CRITICAL_WARNINGS_PRESENT: one or more critical warning/error patterns were found in scanned logs.",
        "SHORT_CONVERGED_INTERVAL: the retained converged interval is shorter than 7200 s.",
        "NO_CONVERGENCE: convergence was not detected for the run.",
    ]


def _html_escape(value) -> str:
    return html.escape(str(value))


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("<table>")
    out.append("<thead><tr>")
    for h in headers:
        out.append(f"<th>{_html_escape(h)}</th>")
    out.append("</tr></thead>")
    out.append("<tbody>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


def _df_to_html_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [c for c in columns if c in df.columns]
    if not available:
        return "<p>No requested columns are available.</p>"

    table_df = df[available].copy()

    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            if col in {"lon_deg", "lat_deg", "lon_mean_deg", "lat_mean_deg"}:
                table_df[col] = table_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.10f}")
            elif col.endswith("_m") or col in {"X_m", "Y_m", "Z_m", "h_m"}:
                table_df[col] = table_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
            else:
                table_df[col] = table_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")

    return table_df.to_html(index=False, escape=True, border=0)


def _compact_column_definition_html(df: pd.DataFrame) -> str:
    groups = [
        (
            "Run identification",
            [
                "dataset_name",
                "run_label",
                "run_dir",
                "station_id",
                "reference_frame",
                "pos_xyz_reference_position",
                "pos_neu_reference_position",
            ],
            "Identifiers, run location, station metadata, and POS-header reference-frame information.",
        ),
        (
            "Time and convergence",
            [
                "time_mean_all_epochs_utc",
                "time_mean_converged_epochs_utc",
                "time_first_epoch_utc",
                "time_last_epoch_utc",
                "convergence_epoch_utc",
                "convergence_delay_sec",
                "convergence_found",
                "convergence_method",
                "n_epochs_total",
                "n_epochs_converged",
                "duration_total_sec",
                "duration_converged_sec",
                "converged_fraction",
            ],
            "Timing of the full run and of the retained post-convergence interval.",
        ),
        (
            "POS source selection",
            [
                "convergence_pos_files",
                "solution_pos_files",
                "convergence_pos_source",
                "solution_pos_source",
            ],
            "POS files and POS type used for convergence detection and final solution estimation.",
        ),
        (
            "Primary daily/per-file solution",
            [
                "solution_method_primary",
                "X_m",
                "Y_m",
                "Z_m",
                "lon_deg",
                "lat_deg",
                "h_m",
                "E_m",
                "N_m",
                "U_m",
            ],
            "Primary solution: median of smoothed POS epochs retained after convergence.",
        ),
        (
            "Secondary daily/per-file solution",
            [
                "X_mean_m",
                "Y_mean_m",
                "Z_mean_m",
                "lon_mean_deg",
                "lat_mean_deg",
                "h_mean_m",
                "E_mean_m",
                "N_mean_m",
                "U_mean_m",
            ],
            "Secondary solution: mean of smoothed POS epochs retained after convergence.",
        ),
        (
            "Convergence thresholds",
            [
                "threshold_E_m",
                "threshold_N_m",
                "threshold_U_m",
                "threshold_rolling_std_E_m",
                "threshold_rolling_std_N_m",
                "threshold_rolling_std_U_m",
                "threshold_slope_E_mm_per_hour",
                "threshold_slope_N_mm_per_hour",
                "threshold_slope_U_mm_per_hour",
                "persistence_window_sec",
                "rolling_window_sec",
            ],
            "Numerical thresholds used by the forward persistence convergence test.",
        ),
        (
            "Accepted convergence-window diagnostics",
            [
                "conv_window_max_abs_E_m",
                "conv_window_max_abs_N_m",
                "conv_window_max_abs_U_m",
                "conv_window_std_E_m",
                "conv_window_std_N_m",
                "conv_window_std_U_m",
                "conv_window_slope_E_mm_per_hour",
                "conv_window_slope_N_mm_per_hour",
                "conv_window_slope_U_mm_per_hour",
                "conv_window_n_epochs",
            ],
            "Diagnostics from the first forward window satisfying all convergence criteria.",
        ),
        (
            "QC and logs",
            [
                "trace_qc_status",
                "trace_warning_count",
                "trace_critical_warning_count",
                "log_files_scanned",
                "qc_flags",
            ],
            "Warnings, critical warnings, scanned logs, and assigned QC flags.",
        ),
        (
            "ENU series reference",
            [
                "series_enu_reference_run_label",
                "series_enu_reference_X_m",
                "series_enu_reference_Y_m",
                "series_enu_reference_Z_m",
            ],
            "Common ECEF reference used to express the final series in local ENU components.",
        ),
    ]

    rows = []
    columns = set(df.columns)
    grouped_columns = set()

    for title, names, description in groups:
        present = [c for c in names if c in columns]
        if not present:
            continue
        grouped_columns.update(present)
        rows.append([
            _html_escape(title),
            _html_escape(description),
            _html_escape(", ".join(present)),
        ])

    html_parts = [
        _html_table(["Group", "Definition", "Columns"], rows)
    ]

    prefixes = [
        "solution_conv",
        "qc_unsmoothed_conv",
        "full_solution",
        "full_unsmoothed",
    ]

    prefix_definitions = {
        "solution_conv": "statistics from smoothed POS epochs after convergence",
        "qc_unsmoothed_conv": "QC/scatter statistics from unsmoothed POS epochs after convergence",
        "full_solution": "statistics from the full smoothed POS time series",
        "full_unsmoothed": "statistics from the full unsmoothed POS time series",
    }

    found_prefixes = []
    for prefix in prefixes:
        if any(c.startswith(prefix + "_") for c in df.columns):
            found_prefixes.append(prefix)

    if found_prefixes:
        rows = []
        for prefix in found_prefixes:
            rows.append([_html_escape(prefix + "_*"), _html_escape(prefix_definitions[prefix])])

        html_parts.append("<h3>Repeated statistical prefixes</h3>")
        html_parts.append(_html_table(["Prefix", "Definition"], rows))

        rows = [
            ["*_std_*", "sample standard deviation"],
            ["*_min_*", "minimum"],
            ["*_max_*", "maximum"],
            ["*_range_*", "maximum minus minimum"],
        ]
        html_parts.append("<h3>Repeated statistical suffixes</h3>")
        html_parts.append(_html_table(["Suffix", "Definition"], rows))

        rows = [
            ["X_m, Y_m, Z_m", "ECEF coordinates in metres"],
            ["lon_deg, lat_deg", "geodetic longitude/latitude in degrees"],
            ["h_m", "ellipsoidal height in metres"],
            ["E_m, N_m, U_m", "local ENU components relative to the common series reference, in metres"],
        ]
        html_parts.append("<h3>Repeated components</h3>")
        html_parts.append(_html_table(["Component", "Definition"], rows))

    statistical_pattern = re.compile(
        r"^(solution_conv|qc_unsmoothed_conv|full_solution|full_unsmoothed)_(std|min|max|range)_"
    )

    ungrouped = [
        c for c in df.columns
        if c not in grouped_columns and not statistical_pattern.match(c)
    ]

    if ungrouped:
        html_parts.append("<h3>Other columns present</h3>")
        html_parts.append(f"<p>{_html_escape(', '.join(ungrouped))}</p>")

    return "\n".join(html_parts)


def _trend_time_years(df: pd.DataFrame) -> list[float] | None:
    if "time_mean_all_epochs_utc" not in df.columns:
        return None

    t = pd.to_datetime(df["time_mean_all_epochs_utc"], errors="coerce", utc=True)

    if t.isna().all():
        return None

    t0 = t.dropna().min()
    years = []

    for value in t:
        if pd.isna(value):
            years.append(math.nan)
        else:
            years.append((value - t0).total_seconds() / (365.25 * 86400.0))

    return years


def _linear_trend_native_per_year(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {
            "ok": False,
            "n": 0,
            "slope_native_per_year": math.nan,
            "slope_native_std_per_year": math.nan,
            "intercept_native": math.nan,
        }

    y_raw = pd.to_numeric(df[column], errors="coerce")
    x_raw = _trend_time_years(df)

    if x_raw is None:
        x_raw = list(range(len(y_raw)))

    pairs = []

    for x, y in zip(x_raw, y_raw):
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))

    n = len(pairs)

    if n < 2:
        return {
            "ok": False,
            "n": n,
            "slope_native_per_year": math.nan,
            "slope_native_std_per_year": math.nan,
            "intercept_native": math.nan,
        }

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    denom = sum((x - x_mean) ** 2 for x in xs)

    if denom == 0:
        return {
            "ok": False,
            "n": n,
            "slope_native_per_year": math.nan,
            "slope_native_std_per_year": math.nan,
            "intercept_native": math.nan,
        }

    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom
    intercept = y_mean - slope * x_mean

    if n >= 3:
        residual_sum = 0.0
        for x, y in pairs:
            y_hat = intercept + slope * x
            residual_sum += (y - y_hat) ** 2

        sigma2 = residual_sum / (n - 2)
        slope_std = math.sqrt(sigma2 / denom) if sigma2 >= 0 else math.nan
    else:
        slope_std = math.nan

    return {
        "ok": True,
        "n": n,
        "slope_native_per_year": float(slope),
        "slope_native_std_per_year": float(slope_std) if math.isfinite(slope_std) else math.nan,
        "intercept_native": float(intercept),
    }
def _meters_per_degree_lon_lat(mean_lat_deg: float) -> tuple[float, float]:
    # WGS84 ellipsoid.
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    phi = math.radians(mean_lat_deg)
    sin_phi = math.sin(phi)

    denom = math.sqrt(1.0 - e2 * sin_phi * sin_phi)

    prime_vertical_radius = a / denom
    meridian_radius = a * (1.0 - e2) / (denom ** 3)

    m_per_deg_lat = (math.pi / 180.0) * meridian_radius
    m_per_deg_lon = (math.pi / 180.0) * prime_vertical_radius * math.cos(phi)

    return m_per_deg_lon, m_per_deg_lat


def _equivalent_velocity_mm_per_year(df: pd.DataFrame, column: str, slope_native_per_year: float) -> tuple[float, str]:
    try:
        slope_native_per_year = float(slope_native_per_year)
    except Exception:
        return math.nan, ""

    if not math.isfinite(slope_native_per_year):
        return math.nan, ""

    if column in {"X_m", "Y_m", "Z_m", "h_m", "E_m", "N_m", "U_m"}:
        return slope_native_per_year * 1000.0, "native m/year × 1000"

    if column in {"lon_deg", "lat_deg"}:
        if "lat_deg" not in df.columns:
            return math.nan, "lat_deg unavailable for angular-to-metric conversion"

        lat_values = pd.to_numeric(df["lat_deg"], errors="coerce").dropna()

        if lat_values.empty:
            return math.nan, "mean latitude unavailable for angular-to-metric conversion"

        mean_lat = float(lat_values.mean())
        m_per_deg_lon, m_per_deg_lat = _meters_per_degree_lon_lat(mean_lat)

        if column == "lon_deg":
            return (
                slope_native_per_year * m_per_deg_lon * 1000.0,
                "longitude trend converted to local East-equivalent mm/year",
            )

        return (
            slope_native_per_year * m_per_deg_lat * 1000.0,
            "latitude trend converted to local North-equivalent mm/year",
        )

    return math.nan, ""
def _equivalent_velocity_uncertainty_mm_per_year(
    df: pd.DataFrame,
    column: str,
    slope_std_native_per_year: float,
) -> float:
    try:
        slope_std_native_per_year = float(slope_std_native_per_year)
    except Exception:
        return math.nan

    if not math.isfinite(slope_std_native_per_year):
        return math.nan

    if column in {"X_m", "Y_m", "Z_m", "h_m"}:
        return slope_std_native_per_year * 1000.0

    if column in {"lon_deg", "lat_deg"}:
        if "lat_deg" not in df.columns:
            return math.nan

        lat_values = pd.to_numeric(df["lat_deg"], errors="coerce").dropna()

        if lat_values.empty:
            return math.nan

        mean_lat = float(lat_values.mean())
        m_per_deg_lon, m_per_deg_lat = _meters_per_degree_lon_lat(mean_lat)

        if column == "lon_deg":
            return slope_std_native_per_year * m_per_deg_lon * 1000.0

        return slope_std_native_per_year * m_per_deg_lat * 1000.0

    return math.nan
def _trend_unit_for_column(column: str) -> str:
    if column in {"X_m", "Y_m", "Z_m", "h_m"}:
        return "m/year"

    if column in {"lon_deg", "lat_deg"}:
        return "deg/year"

    return "native unit/year"


def _fmt_velocity(value) -> str:
    return _fmt(value, digits=3)


def _final_station_solution(df: pd.DataFrame) -> dict:
    required = ["X_m", "Y_m", "Z_m", "lon_deg", "lat_deg", "h_m"]

    optional = ["E_m", "N_m", "U_m"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        return {
            "ok": False,
            "missing": missing,
            "rows": [],
        }

    coordinate_specs = [
        ("X_m", "X", "m", _fmt_m, _fmt_m),
        ("Y_m", "Y", "m", _fmt_m, _fmt_m),
        ("Z_m", "Z", "m", _fmt_m, _fmt_m),
        ("lon_deg", "longitude", "deg", _fmt_deg, _fmt_sci),
        ("lat_deg", "latitude", "deg", _fmt_deg, _fmt_sci),
        ("h_m", "ellipsoidal height", "m", _fmt_m, _fmt_m),
    ]

    for col, label in [
        ("E_m", "E"),
        ("N_m", "N"),
        ("U_m", "U"),
    ]:
        if col in df.columns:
            coordinate_specs.append((col, label, "m", _fmt_m, _fmt_m))

    rows = []

    for col, label, unit, fmt_mean, fmt_std in coordinate_specs:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        n = int(values.count())

        if n == 0:
            mean_value = math.nan
            std_value = math.nan
            min_value = math.nan
            max_value = math.nan
            range_value = math.nan
        else:
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if n >= 2 else math.nan
            min_value = float(values.min())
            max_value = float(values.max())
            range_value = max_value - min_value

        trend = _linear_trend_native_per_year(df, col)
        slope_native = trend["slope_native_per_year"]
        slope_native_std = trend.get("slope_native_std_per_year", math.nan)

        velocity_mm_year, velocity_note = _equivalent_velocity_mm_per_year(df, col, slope_native)
        velocity_unc_mm_year, _ = _equivalent_velocity_mm_per_year(df, col, slope_native_std)

        if col in {"lon_deg", "lat_deg"}:
            fmt_slope = _fmt_sci(slope_native)
        else:
            fmt_slope = _fmt(slope_native, digits=6)

        rows.append({
            "coordinate": label,
            "column": col,
            "mean": mean_value,
            "std": std_value,
            "min": min_value,
            "max": max_value,
            "range": range_value,
            "unit": unit,
            "n": n,
            "trend_n": trend["n"],
            "slope_native_per_year": slope_native,
            "slope_native_std_per_year": slope_native_std,
            "trend_unit": _trend_unit_for_column(col),
            "velocity_mm_per_year": velocity_mm_year,
            "velocity_uncertainty_mm_per_year": velocity_unc_mm_year,
            "velocity_note": velocity_note,
            "fmt_mean": fmt_mean(mean_value),
            "fmt_std": fmt_std(std_value),
            "fmt_min": fmt_mean(min_value),
            "fmt_max": fmt_mean(max_value),
            "fmt_range": fmt_std(range_value),
            "fmt_slope": fmt_slope,
            "fmt_velocity": _fmt_velocity(velocity_mm_year),
            "fmt_velocity_uncertainty": _fmt_velocity(velocity_unc_mm_year),
        })

    return {
        "ok": True,
        "missing": [],
        "rows": rows,
    }
def _final_station_solution_html(df: pd.DataFrame) -> str:
    solution = _final_station_solution(df)

    if not solution["ok"]:
        return (
            "<p>Final station solution could not be computed because the following columns are missing: "
            f"{_html_escape(', '.join(solution['missing']))}</p>"
        )

    rows = []
    for item in solution["rows"]:
        rows.append([
            _html_escape(item["coordinate"]),
            _html_escape(item["column"]),
            _html_escape(item["fmt_mean"]),
            _html_escape(item["fmt_std"]),
            _html_escape(item["fmt_min"]),
            _html_escape(item["fmt_max"]),
            _html_escape(item["fmt_range"]),
            _html_escape(item["unit"]),
            _html_escape(item["fmt_slope"]),
            _html_escape(item["trend_unit"]),
            _html_escape(item["fmt_velocity"]),
            _html_escape(item["fmt_velocity_uncertainty"]),
            _html_escape(item["n"]),
        ])

    text = []
    text.append(
        "<p>The final GNSS station coordinate solution is computed as the arithmetic mean of the "
        "daily/per-file primary PPP solutions. The reported standard deviations are computed from the "
        "scatter of the daily/per-file primary solutions around this mean. They are not derived from the "
        "per-file internal standard deviations.</p>"
    )
    text.append(
        "<p>The linear trend is estimated from the available daily/per-file primary PPP solutions by "
        "ordinary least-squares regression against time. For short test series, such as three daily solutions, "
        "this value is a diagnostic trend over the available interval and should not be interpreted as a "
        "long-term geodetic station velocity.</p>"
    )
    text.append(
        "<p>For longitude and latitude, the equivalent velocity in mm/year is computed by converting the "
        "angular trend to local East-equivalent and North-equivalent metric rates using the mean latitude of "
        "the available solutions and the WGS84 ellipsoid. For X, Y, Z, h, E, N and U, the equivalent velocity "
        "is the native metre-per-year trend multiplied by 1000.</p>"
    )
    text.append(_html_table(
        [
            "Coordinate",
            "Source column",
            "Final mean value",
            "Standard deviation",
            "Minimum value",
            "Maximum value",
            "Range",
            "Unit",
            "Linear trend",
            "Trend unit",
            "Equivalent velocity (mm/year)",
            "Equivalent velocity uncertainty (mm/year)",
            "Number of daily/per-file solutions",
        ],
        rows,
    ))

    return chr(10).join(text)
def _resolve_plot_columns(plot_columns) -> list[tuple[str, str]]:
    mapping = {
        "X": ("X", "X_m"),
        "Y": ("Y", "Y_m"),
        "Z": ("Z", "Z_m"),
        "lon": ("longitude", "lon_deg"),
        "lat": ("latitude", "lat_deg"),
        "h": ("ellipsoidal height", "h_m"),
        "E": ("E", "E_m"),
        "N": ("N", "N_m"),
        "U": ("U", "U_m"),
    }

    if plot_columns is None:
        requested = ["X", "Y", "Z", "h"]
    elif isinstance(plot_columns, str):
        if plot_columns.strip().lower() == "all":
            requested = ["X", "Y", "Z", "lon", "lat", "h", "E", "N", "U"]
        else:
            requested = [p.strip() for p in plot_columns.split(",") if p.strip()]
    else:
        requested = list(plot_columns)

    aliases = {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "lon": "lon",
        "longitude": "lon",
        "lat": "lat",
        "latitude": "lat",
        "h": "h",
        "height": "h",
        "ellipsoidal_height": "h",
        "e": "E",
        "east": "E",
        "n": "N",
        "north": "N",
        "u": "U",
        "up": "U",
        "all": "all",
    }

    resolved = []
    seen = set()

    for item in requested:
        key = str(item).strip()
        if not key:
            continue

        canonical = aliases.get(key.lower())

        if canonical == "all":
            expanded = ["X", "Y", "Z", "lon", "lat", "h", "E", "N", "U"]
        elif canonical:
            expanded = [canonical]
        else:
            continue

        for col_key in expanded:
            if col_key in mapping and col_key not in seen:
                resolved.append(mapping[col_key])
                seen.add(col_key)

    return resolved
def _timestamp_to_decimal_year(value) -> float:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return math.nan

    if pd.isna(ts):
        return math.nan

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    year_start = pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")
    next_year_start = pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")

    year_seconds = (next_year_start - year_start).total_seconds()
    elapsed_seconds = (ts - year_start).total_seconds()

    if year_seconds <= 0:
        return math.nan

    return float(ts.year + elapsed_seconds / year_seconds)


def _failed_dataset_decimal_year(item: dict) -> float:
    for key in ["time_mean_all_epochs_utc", "time_mean_utc"]:
        value = item.get(key, "")
        if value:
            dec = _timestamp_to_decimal_year(value)
            if math.isfinite(dec):
                return dec

    start = item.get("start_epoch", "") or item.get("start_time_utc", "")
    end = item.get("end_epoch", "") or item.get("end_time_utc", "")

    if start and end:
        try:
            ts_start = pd.Timestamp(start)
            ts_end = pd.Timestamp(end)

            if ts_start.tzinfo is None:
                ts_start = ts_start.tz_localize("UTC")
            else:
                ts_start = ts_start.tz_convert("UTC")

            if ts_end.tzinfo is None:
                ts_end = ts_end.tz_localize("UTC")
            else:
                ts_end = ts_end.tz_convert("UTC")

            midpoint = ts_start + (ts_end - ts_start) / 2
            dec = _timestamp_to_decimal_year(midpoint)
            if math.isfinite(dec):
                return dec
        except Exception:
            pass

    if start:
        dec = _timestamp_to_decimal_year(start)
        if math.isfinite(dec):
            return dec

    dataset_name = str(item.get("dataset_name", ""))
    m = re.match(r"^(\d{4})[_-](\d{3})$", dataset_name)
    if m:
        year = int(m.group(1))
        doy = int(m.group(2))
        ts = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1, hours=12)
        dec = _timestamp_to_decimal_year(ts)
        if math.isfinite(dec):
            return dec

    return math.nan


def _non_successful_datasets_html(failed_datasets: list[dict] | None = None) -> str:
    failed_datasets = failed_datasets or []

    text = []
    text.append("<h3>Non-successful datasets</h3>")

    if not failed_datasets:
        text.append("<p>None reported.</p>")
        return "\n".join(text)

    rows = []
    for item in failed_datasets:
        rows.append([
            _html_escape(item.get("dataset_name", "")),
            _html_escape(item.get("status", "")),
            _html_escape(item.get("message", "")),
            _html_escape(item.get("start_epoch", "")),
            _html_escape(item.get("end_epoch", "")),
            _html_escape(item.get("run_dir", "")),
        ])

    text.append(
        "<p>The following datasets did not complete successfully. They are not included in "
        "<code>timeseries.out</code>, but their processing status is reported here for traceability.</p>"
    )
    text.append(_html_table(
        ["Dataset", "Status", "Message", "Start epoch", "End epoch", "Run directory"],
        rows,
    ))

    return "\n".join(text)




def _default_report_analysis_config() -> dict:
    return {
        "shift_detection": {
            "enabled": True,
            "min_series_days": 100.0,
            "window_days": 28.0,
            "step_days": 1.0,
            "min_points_per_window": 10,
            "mad_sigma_floor_m": 0.005,
            "min_abs_jump_m": 0.005,
            "min_jump_sigma": 3.0,
            "min_model_improvement_percent": 20.0,
            "min_confidence": 0.70,
            "noise_method": "diff_mad",
        },
        "shift_clustering": {
            "cluster_window_days": 14.0,
            "min_components": 1,
        },
        "shift_report_selection": {
            "min_confidence": 0.90,
            "min_abs_jump_mm": 20.0,
            "min_components": 1,
        },
        "meta_clustering": {
            "enabled": True,
            "max_gap_days": 14.0,
            "enable_direction_similarity": True,
            "direction_mode": "horizontal",
            "max_direction_change_deg": 45.0,
            "enable_magnitude_compatibility": False,
            "max_magnitude_ratio": 3.0,
        },
        "velocity_windows": {
            "min_points_per_window": 5,
            "min_duration_days_for_stable_rate": 365.25,
            "apply_gaussian_smoothing": True,
            "gaussian_width_days": 28.0,
            "outlier_rejection_enabled": True,
            "sigma_floor_m": 0.001,
        },
        "rolling_velocity_diagnostics": {
            "enabled": True,
            "window_days": 182.0,
            "step_days": 7.0,
            "comparison_lag_days": 91.0,
            "min_points_per_window": 20,
            "apply_gaussian_smoothing": True,
            "gaussian_width_days": 28.0,
            "significance_z_threshold": 4.0,
            "min_abs_delta_velocity_mm_per_year": 1.0,
            "minimum_persistence_fraction_of_window": 0.5,
            "horizontal_coherence_required": True,
            "meta_association_window_days": None,
            "include_horizontal_vector": True,
        },
        "plots": {
            "show_gaussian_smoothed_enu": True,
            "gaussian_width_days": 28.0,
            "velocity_segment_plot_mode": 1,
        },
    }


def _resolve_report_analysis_config(report_analysis_config: dict | None = None) -> dict:
    cfg = _default_report_analysis_config()

    if not report_analysis_config:
        return cfg

    if not isinstance(report_analysis_config, dict):
        return cfg

    for section, values in report_analysis_config.items():
        if section not in cfg or not isinstance(values, dict):
            continue

        for key, value in values.items():
            if key in cfg[section]:
                cfg[section][key] = value

    # Keep plot smoothing width aligned with velocity smoothing width unless
    # explicitly supplied under plots.
    if "plots" not in report_analysis_config or "gaussian_width_days" not in report_analysis_config.get("plots", {}):
        cfg["plots"]["gaussian_width_days"] = cfg["velocity_windows"]["gaussian_width_days"]

    return cfg

def _compute_report_shift_clusters(df: pd.DataFrame, report_analysis_config: dict | None = None) -> pd.DataFrame:
    try:
        import timeseries_change_detection as tcd
    except Exception:
        return pd.DataFrame()

    required = ["time_mean_all_epochs_utc", "E_m", "N_m", "U_m"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    shift_cfg = analysis_cfg["shift_detection"]
    cluster_cfg = analysis_cfg["shift_clustering"]
    selection_cfg = analysis_cfg["shift_report_selection"]

    if not bool(shift_cfg.get("enabled", True)):
        return pd.DataFrame()

    cfg = tcd.ShiftDetectionConfig(
        columns=("E_m", "N_m", "U_m"),
        min_series_days=float(shift_cfg["min_series_days"]),
        window_days=float(shift_cfg["window_days"]),
        step_days=float(shift_cfg["step_days"]),
        min_points_per_window=int(shift_cfg["min_points_per_window"]),
        mad_sigma_floor_m=float(shift_cfg["mad_sigma_floor_m"]),
        min_abs_jump_m=float(shift_cfg["min_abs_jump_m"]),
        min_jump_sigma=float(shift_cfg["min_jump_sigma"]),
        min_model_improvement_percent=float(shift_cfg["min_model_improvement_percent"]),
        min_confidence=float(shift_cfg["min_confidence"]),
        noise_method=str(shift_cfg["noise_method"]),
    )

    events = tcd.detect_shifts(df, cfg)
    clusters = tcd.cluster_shift_events(
        events,
        cluster_window_days=float(cluster_cfg["cluster_window_days"]),
        min_components=int(cluster_cfg["min_components"]),
    )

    selected = tcd.select_report_shift_clusters(
        clusters,
        min_confidence=float(selection_cfg["min_confidence"]),
        min_abs_jump_mm=float(selection_cfg["min_abs_jump_mm"]),
        min_components=int(selection_cfg["min_components"]),
    )

    if selected is None or len(selected) == 0 or events is None or len(events) == 0:
        return selected

    selected = selected.copy()
    selected.attrs["report_analysis_config"] = analysis_cfg
    selected.attrs["total_accepted_shift_candidates"] = int(len(events))
    selected.attrs["total_report_cluster_candidates"] = int(selected["n_candidates"].sum()) if "n_candidates" in selected.columns else 0
    selected.attrs["total_report_clusters"] = int(len(selected))

    candidate_decimal_years = []
    candidate_utc_times = []
    candidate_counts = []

    work_events = events.copy()
    work_events["decimal_year"] = pd.to_numeric(work_events["decimal_year"], errors="coerce")

    for _, cluster in selected.iterrows():
        try:
            start_dec = float(cluster.get("cluster_start_decimal_year", math.nan))
            end_dec = float(cluster.get("cluster_end_decimal_year", math.nan))
        except Exception:
            start_dec = math.nan
            end_dec = math.nan

        components_text = str(cluster.get("components", ""))
        components = [item.strip() for item in components_text.split(",") if item.strip()]

        mask = pd.Series(True, index=work_events.index)

        if math.isfinite(start_dec) and math.isfinite(end_dec):
            lo = min(start_dec, end_dec)
            hi = max(start_dec, end_dec)
            mask = mask & (work_events["decimal_year"] >= lo) & (work_events["decimal_year"] <= hi)

        if components:
            mask = mask & work_events["component"].astype(str).isin(components)

        candidates = work_events.loc[mask].sort_values("decimal_year")

        dec_values = []
        utc_values = []

        for _, candidate in candidates.iterrows():
            try:
                dec = float(candidate.get("decimal_year", math.nan))
            except Exception:
                dec = math.nan

            if math.isfinite(dec):
                dec_values.append(dec)
                utc_values.append(str(candidate.get("time_utc", "")))

        candidate_decimal_years.append(",".join(f"{value:.4f}" for value in dec_values))
        candidate_utc_times.append(",".join(utc_values))
        candidate_counts.append(len(dec_values))

    selected["candidate_decimal_years"] = candidate_decimal_years
    selected["candidate_utc_times"] = candidate_utc_times
    selected["candidate_count"] = candidate_counts

    return selected



def _shift_cluster_duration_days(item) -> float:
    try:
        start = float(item.get("cluster_start_decimal_year", math.nan))
        end = float(item.get("cluster_end_decimal_year", math.nan))
    except Exception:
        return math.nan

    if not math.isfinite(start) or not math.isfinite(end):
        return math.nan

    return abs(end - start) * 365.25


def _shift_cluster_event_class(item) -> str:
    try:
        n_candidates = int(item.get("n_candidates", 0))
    except Exception:
        n_candidates = 0

    duration_days = _shift_cluster_duration_days(item)

    if n_candidates <= 1 or (math.isfinite(duration_days) and duration_days <= 1.0):
        return "single-candidate shift"

    if math.isfinite(duration_days) and duration_days <= 30.0:
        return "localized shift cluster"

    return "extended transition cluster"

def _shift_clusters_html(clusters: pd.DataFrame) -> str:
    text = []

    text.append(
        "<p>Report-grade shift detection is based on robust windowed step detection, "
        "first-difference MAD noise estimation, temporal clustering of adjacent candidates, "
        "and conservative report filtering. Current fixed V1 strict-cluster report thresholds are: <code>window_days = 28</code>, <code>strict_cluster_window_days = 14</code>, <code>mad_sigma_floor = 5 mm</code>, <code>min_abs_jump = 5 mm</code>, <code>min_jump_sigma = 3</code>, <code>min_model_improvement_percent = 20</code>, <code>min_report_confidence = 0.90</code>, and <code>min_report_abs_jump = 20 mm</code>.</p>"
    )

    total_candidates = ""
    report_cluster_candidates = ""
    report_clusters = ""

    if clusters is not None:
        total_candidates = clusters.attrs.get("total_accepted_shift_candidates", "")
        report_cluster_candidates = clusters.attrs.get("total_report_cluster_candidates", "")
        report_clusters = clusters.attrs.get("total_report_clusters", len(clusters))

    text.append(
        "<p>With the current detector settings, the shift detector identified "
        f"<strong>{_html_escape(total_candidates)}</strong> accepted shift candidates in total. "
        f"Of these, <strong>{_html_escape(report_cluster_candidates)}</strong> candidates belong to the "
        f"report-grade strict clusters shown in the table below, corresponding to "
        f"<strong>{_html_escape(report_clusters)}</strong> internal cluster ID(s).</p>"
    )

    if clusters is None or len(clusters) == 0:
        text.append("<p>No report-grade shift clusters were detected.</p>")
        return chr(10).join(text)

    rows = []

    for report_event_id, (_, item) in enumerate(clusters.iterrows(), start=1):
        duration_days = _shift_cluster_duration_days(item)
        event_class = _shift_cluster_event_class(item)

        rows.append([
            _html_escape(report_event_id),
            _html_escape(item.get("cluster_id", "")),
            _html_escape(event_class),
            _html_escape(_fmt(item.get("representative_decimal_year", math.nan), digits=4)),
            _html_escape(item.get("representative_time_utc", "")),
            _html_escape(_fmt(item.get("cluster_start_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(item.get("cluster_end_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(duration_days, digits=1)),
            _html_escape(item.get("components", "")),
            _html_escape(item.get("n_candidates", "")),
            _html_escape(item.get("candidate_count", "")),
            _html_escape(item.get("n_components", "")),
            _html_escape(_fmt(item.get("max_abs_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("max_jump_sigma", math.nan), digits=3)),
            _html_escape(_fmt(item.get("max_model_improvement_percent", math.nan), digits=2)),
            _html_escape(_fmt(item.get("max_confidence", math.nan), digits=3)),
            _html_escape(_fmt(item.get("E_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("N_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("U_jump_mm", math.nan), digits=3)),
        ])

    text.append(_html_table(
        [
            "Report event",
            "Internal cluster ID",
            "Event class",
            "Representative decimal year",
            "Representative UTC",
            "Cluster start decimal year",
            "Cluster end decimal year",
            "Cluster duration (days)",
            "Components",
            "Cluster candidates",
            "Annotated candidates",
            "Component count",
            "Max abs jump (mm)",
            "Max jump sigma",
            "Max model improvement (%)",
            "Max confidence",
            "E jump (mm)",
            "N jump (mm)",
            "U jump (mm)",
        ],
        rows,
    ))

    text.append(
        "<p>In the full-span ENU plots, light grey shaded intervals indicate the temporal span "
        "of each report-grade strict cluster, and red dashed vertical lines indicate the "
        "representative epoch of each cluster. Accepted shift-candidate epochs are not shown "
        "in the full-span plots; they are shown only in the shift-cluster zoom plots.</p>"
    )

    return chr(10).join(text)


def _velocity_uncertainty_mm_per_year_for_window(
    sigma_m,
    x_window: pd.Series,
    series_used_for_fit: str,
) -> float:
    if str(series_used_for_fit) == "meta_cluster_net_jump":
        return math.nan

    try:
        sigma_m = float(sigma_m)
    except Exception:
        return math.nan

    if not math.isfinite(sigma_m):
        return math.nan

    x_values = pd.to_numeric(x_window, errors="coerce").dropna()

    if len(x_values) < 3:
        return math.nan

    x_mean = float(x_values.mean())
    sxx = float(((x_values - x_mean) ** 2).sum())

    if not math.isfinite(sxx) or sxx <= 0:
        return math.nan

    return 1000.0 * sigma_m / math.sqrt(sxx)


def _draw_velocity_trends_on_fullspan_axis(
    ax,
    df: pd.DataFrame,
    column: str,
    x_series: pd.Series,
    velocity_windows: pd.DataFrame | None,
) -> bool:
    if velocity_windows is None or len(velocity_windows) == 0:
        return False

    if column not in {"E_m", "N_m", "U_m"}:
        return False

    if column not in df.columns:
        return False

    work = velocity_windows.copy()

    if "component" not in work.columns:
        return False

    work = work[work["component"].astype(str) == column].copy()

    if len(work) == 0:
        return False

    required = [
        "start_decimal_year",
        "end_decimal_year",
        "velocity_mm_per_year",
        "window_label",
        "rate_class",
        "series_used_for_fit",
        "sigma_m",
    ]

    missing = [item for item in required if item not in work.columns]
    if missing:
        return False

    y_all = pd.to_numeric(df[column], errors="coerce")
    y_finite = y_all.dropna()

    if len(y_finite) > 0:
        y_span = float(y_finite.max() - y_finite.min())
    else:
        y_span = 0.0

    if not math.isfinite(y_span) or y_span <= 0:
        y_span = 0.01

    plotted_any = False

    sort_order = {
        "before_meta_cluster": 0,
        "during_meta_cluster": 1,
        "after_meta_cluster": 2,
    }

    work["_sort_order"] = work["window_label"].astype(str).map(sort_order).fillna(99)
    work = work.sort_values(["_sort_order", "start_decimal_year", "end_decimal_year"])

    for idx, (_, item) in enumerate(work.iterrows()):
        try:
            x0 = float(item.get("start_decimal_year", math.nan))
            x1 = float(item.get("end_decimal_year", math.nan))
            velocity_mm_per_year = float(item.get("velocity_mm_per_year", math.nan))
        except Exception:
            continue

        if not math.isfinite(x0) or not math.isfinite(x1) or not math.isfinite(velocity_mm_per_year):
            continue

        if x1 <= x0:
            continue

        mask = (x_series >= x0) & (x_series <= x1)
        x_window = x_series.loc[mask]
        y_window = y_all.loc[mask].dropna()

        if len(x_window.dropna()) < 2 or len(y_window) == 0:
            continue

        x_mid = float(pd.to_numeric(x_window, errors="coerce").dropna().median())
        y_anchor = float(y_window.median())

        slope_m_per_year = velocity_mm_per_year / 1000.0

        y0 = y_anchor + slope_m_per_year * (x0 - x_mid)
        y1 = y_anchor + slope_m_per_year * (x1 - x_mid)

        label = "velocity trend" if not plotted_any else None

        ax.plot(
            [x0, x1],
            [y0, y1],
            color="green",
            linewidth=1.4,
            alpha=0.95,
            label=label,
            zorder=4,
        )

        sigma_v = _velocity_uncertainty_mm_per_year_for_window(
            item.get("sigma_m", math.nan),
            x_window,
            item.get("series_used_for_fit", ""),
        )

        if math.isfinite(sigma_v):
            sigma_text = f"{sigma_v:.1f}"
        else:
            sigma_text = "n/a"

        window_label = str(item.get("window_label", ""))
        rate_class = str(item.get("rate_class", ""))

        if window_label == "before_meta_cluster":
            prefix = "before"
        elif window_label == "during_meta_cluster":
            prefix = "during"
        elif window_label == "after_meta_cluster":
            prefix = "after"
        else:
            prefix = window_label

        text_y = y_anchor + ((idx % 3) - 1) * 0.035 * y_span

        ax.text(
            x_mid,
            text_y,
            f"{prefix}: {velocity_mm_per_year:+.1f} ± {sigma_text} mm yr⁻¹",
            color="black",
            fontsize=6,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "pad": 1.5,
            },
            zorder=5,
        )

        plotted_any = True

    return plotted_any


def _gaussian_smoothed_enu_series_for_plot(
    df: pd.DataFrame,
    column: str,
    x_series: pd.Series,
    gaussian_width_days: float = 28.0,
) -> pd.Series | None:
    if column not in {"E_m", "N_m", "U_m"}:
        return None

    if column not in df.columns:
        return None

    try:
        import timeseries_velocity_detection as tvd
    except Exception:
        return None

    required_helpers = [
        "MetaClusterVelocityWindowConfig",
        "_pre_smoothing_outlier_mask",
        "_gaussian_smooth_by_decimal_year",
    ]

    if any(not hasattr(tvd, name) for name in required_helpers):
        return None

    y_raw = pd.to_numeric(df[column], errors="coerce")
    valid = x_series.notna() & y_raw.notna()

    if valid.sum() < 3:
        return None

    x = pd.to_numeric(x_series.loc[valid], errors="coerce").to_numpy(dtype=float)
    y = y_raw.loc[valid].to_numpy(dtype=float)

    cfg = tvd.MetaClusterVelocityWindowConfig(
        columns=(column,),
        min_points_per_window=5,
        min_duration_days_for_stable_rate=365.25,
        apply_gaussian_smoothing=True,
        gaussian_width_days=float(gaussian_width_days),
        outlier_rejection_enabled=True,
        sigma_floor_m=0.001,
    )

    try:
        keep = tvd._pre_smoothing_outlier_mask(x, y, cfg)
        y_smooth = tvd._gaussian_smooth_by_decimal_year(
            x=x,
            y=y,
            valid_mask=keep,
            gaussian_width_days=float(gaussian_width_days),
        )
    except Exception:
        return None

    out = pd.Series(index=df.index, dtype="float64")
    out.loc[valid] = y_smooth

    return out


def _decimal_year_series_for_composite_plot(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "decimal_year" in df.columns:
        x = pd.to_numeric(df["decimal_year"], errors="coerce")
        return pd.Series(x, index=df.index, dtype="float64"), "decimal year"

    time_candidates = [
        "time_mean_all_epochs_utc",
        "time_utc",
        "datetime_utc",
        "time",
    ]

    for col in time_candidates:
        if col not in df.columns:
            continue

        t = pd.to_datetime(df[col], errors="coerce", utc=True)

        if t.notna().sum() == 0:
            continue

        dec = []
        for item in t:
            if pd.isna(item):
                dec.append(math.nan)
                continue

            start = pd.Timestamp(year=int(item.year), month=1, day=1, tz="UTC")
            end = pd.Timestamp(year=int(item.year) + 1, month=1, day=1, tz="UTC")
            value = int(item.year) + (item - start).total_seconds() / (end - start).total_seconds()
            dec.append(float(value))

        return pd.Series(dec, index=df.index, dtype="float64"), "decimal year"

    x = pd.Series(range(len(df)), index=df.index, dtype="float64")
    return x, "index"


def _add_external_legend(fig, axes) -> None:
    if axes is None:
        return

    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes_list = []
        for item in axes:
            if hasattr(item, "ravel"):
                axes_list.extend(list(item.ravel()))
            elif isinstance(item, (list, tuple)):
                axes_list.extend(item)
            else:
                axes_list.append(item)
    else:
        axes_list = [axes]

    handles = []
    labels = []

    for ax in axes_list:
        if not hasattr(ax, "get_legend_handles_labels"):
            continue

        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.98),
        borderaxespad=0.0,
        fontsize=8,
    )



def _select_velocity_change_segments_for_plot(
    velocity_change_diagnostics: dict | None,
    report_analysis_config: dict | None = None,
) -> pd.DataFrame:
    if velocity_change_diagnostics is None:
        return pd.DataFrame()

    classified = velocity_change_diagnostics.get("classified", pd.DataFrame())

    if classified is None or len(classified) == 0:
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    plot_cfg = analysis_cfg.get("plots", {})

    try:
        mode = int(plot_cfg.get("velocity_segment_plot_mode", 1))
    except Exception:
        mode = 1

    work = classified.copy()

    if "component" not in work.columns:
        return pd.DataFrame()

    if mode == 1:
        # Shift-related report-grade horizontal velocity-change segments only.
        required_cols = {"report_grade", "shift_related_velocity_change", "component"}
        if not required_cols.issubset(set(work.columns)):
            return pd.DataFrame()

        work = work[
            (work["report_grade"] == True)
            & (work["shift_related_velocity_change"] == True)
            & (work["component"].astype(str) == "H_magnitude")
        ].copy()

    elif mode == 2:
        # All report-grade horizontal velocity-change segments.
        required_cols = {"report_grade", "component"}
        if not required_cols.issubset(set(work.columns)):
            return pd.DataFrame()

        work = work[
            (work["report_grade"] == True)
            & (work["component"].astype(str) == "H_magnitude")
        ].copy()

    elif mode == 3:
        # All persistent horizontal velocity-change segments.
        work = work[work["component"].astype(str).isin(["H_magnitude", "E_m", "N_m"])].copy()

    elif mode == 4:
        # All persistent velocity-change segments including vertical diagnostic-only.
        work = work.copy()

    else:
        work = work[
            (work.get("report_grade", False) == True)
            & (work.get("shift_related_velocity_change", False) == True)
            & (work["component"].astype(str) == "H_magnitude")
        ].copy()

    if len(work) == 0:
        return pd.DataFrame()

    for col in [
        "cluster_start_decimal_year",
        "cluster_end_decimal_year",
        "representative_center_decimal_year",
        "representative_delta_velocity_mm_per_year",
        "max_velocity_change_z",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["cluster_start_decimal_year", "cluster_end_decimal_year"])

    return work.sort_values(["cluster_start_decimal_year", "cluster_end_decimal_year"]).reset_index(drop=True)


def _local_fit_velocity_segment_for_component(
    df: pd.DataFrame,
    column: str,
    x_series: pd.Series,
    x0: float,
    x1: float,
    gaussian_width_days: float,
) -> dict | None:
    if column not in df.columns:
        return None

    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return None

    y_series = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column=column,
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if y_series is None:
        y_series = pd.to_numeric(df[column], errors="coerce")

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")

    mask = (x >= x0) & (x <= x1) & x.notna() & y.notna()

    xw = x.loc[mask].to_numpy(dtype=float)
    yw = y.loc[mask].to_numpy(dtype=float)

    if len(xw) < 3:
        return None

    try:
        coeff = np.polyfit(xw, yw, deg=1)
    except Exception:
        return None

    slope_m_per_year = float(coeff[0])
    intercept = float(coeff[1])

    y_fit = slope_m_per_year * xw + intercept
    residuals = yw - y_fit

    dof = max(1, len(xw) - 2)
    sigma_m = float(np.sqrt(np.sum(residuals ** 2) / dof))

    x_mean = float(np.mean(xw))
    sxx = float(np.sum((xw - x_mean) ** 2))

    if sxx > 0 and math.isfinite(sigma_m):
        sigma_velocity_mm_per_year = 1000.0 * sigma_m / math.sqrt(sxx)
    else:
        sigma_velocity_mm_per_year = math.nan

    return {
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(slope_m_per_year * x0 + intercept),
        "y1": float(slope_m_per_year * x1 + intercept),
        "x_mid": float(0.5 * (x0 + x1)),
        "y_mid": float(slope_m_per_year * (0.5 * (x0 + x1)) + intercept),
        "velocity_mm_per_year": 1000.0 * slope_m_per_year,
        "sigma_velocity_mm_per_year": sigma_velocity_mm_per_year,
        "n_points": int(len(xw)),
    }


def _draw_velocity_change_segments_on_enu_axis(
    ax,
    df: pd.DataFrame,
    column: str,
    x_series: pd.Series,
    selected_segments: pd.DataFrame | None,
    gaussian_width_days: float,
) -> bool:
    if selected_segments is None or len(selected_segments) == 0:
        return False

    if column not in {"E_m", "N_m", "U_m"}:
        return False

    plotted_any = False

    y_all = pd.to_numeric(df[column], errors="coerce")
    y_finite = y_all.dropna()

    if len(y_finite) > 0:
        y_span = float(y_finite.max() - y_finite.min())
    else:
        y_span = 0.0

    if not math.isfinite(y_span) or y_span <= 0:
        y_span = 0.01

    for idx, (_, segment) in enumerate(selected_segments.iterrows()):
        try:
            x0 = float(segment.get("cluster_start_decimal_year", math.nan))
            x1 = float(segment.get("cluster_end_decimal_year", math.nan))
        except Exception:
            continue

        fit = _local_fit_velocity_segment_for_component(
            df=df,
            column=column,
            x_series=x_series,
            x0=x0,
            x1=x1,
            gaussian_width_days=gaussian_width_days,
        )

        if fit is None:
            continue

        label = "velocity-change segment trend" if not plotted_any else None

        ax.plot(
            [fit["x0"], fit["x1"]],
            [fit["y0"], fit["y1"]],
            color="green",
            linewidth=1.5,
            alpha=0.95,
            label=label,
            zorder=4,
        )

        sigma_v = fit["sigma_velocity_mm_per_year"]

        if math.isfinite(sigma_v):
            sigma_text = f"{sigma_v:.1f}"
        else:
            sigma_text = "n/a"

        try:
            max_z = float(segment.get("max_velocity_change_z", math.nan))
        except Exception:
            max_z = math.nan

        z_text = f", Z={max_z:.1f}" if math.isfinite(max_z) else ""

        text_y = fit["y_mid"] + ((idx % 3) - 1) * 0.035 * y_span

        ax.text(
            fit["x_mid"],
            text_y,
            f"v: {fit['velocity_mm_per_year']:+.1f} ± {sigma_text} mm yr⁻¹{z_text}",
            color="black",
            fontsize=6,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "pad": 1.5,
            },
            zorder=5,
        )

        plotted_any = True

    return plotted_any


def _transient_windows_from_velocity_diagnostics(
    velocity_change_diagnostics: dict | None,
) -> pd.DataFrame:
    if velocity_change_diagnostics is None:
        return pd.DataFrame()

    value = velocity_change_diagnostics.get("transient_windows", pd.DataFrame())

    if value is None or len(value) == 0:
        return pd.DataFrame()

    return value.copy()


def _component_transient_fits_from_velocity_diagnostics(
    velocity_change_diagnostics: dict | None,
) -> pd.DataFrame:
    if velocity_change_diagnostics is None:
        return pd.DataFrame()

    value = velocity_change_diagnostics.get("transient_model_fits", pd.DataFrame())

    if value is None or len(value) == 0:
        return pd.DataFrame()

    return value.copy()


def _joint_horizontal_transient_fits_from_velocity_diagnostics(
    velocity_change_diagnostics: dict | None,
) -> pd.DataFrame:
    if velocity_change_diagnostics is None:
        return pd.DataFrame()

    value = velocity_change_diagnostics.get("joint_horizontal_transient_model_fits", pd.DataFrame())

    if value is None or len(value) == 0:
        return pd.DataFrame()

    return value.copy()


def _clean_transient_windows_for_plot(transient_windows: pd.DataFrame) -> pd.DataFrame:
    if transient_windows is None or len(transient_windows) == 0:
        return pd.DataFrame()

    required = {"window_type", "start_decimal_year", "end_decimal_year"}

    if not required.issubset(set(transient_windows.columns)):
        return pd.DataFrame()

    out = transient_windows.copy()
    out["start_decimal_year"] = pd.to_numeric(out["start_decimal_year"], errors="coerce")
    out["end_decimal_year"] = pd.to_numeric(out["end_decimal_year"], errors="coerce")

    out = out[
        out["start_decimal_year"].notna()
        & out["end_decimal_year"].notna()
        & (out["end_decimal_year"] > out["start_decimal_year"])
    ].copy()

    return out.reset_index(drop=True)


def _stable_intervals_outside_transient_windows(
    x_series: pd.Series,
    transient_windows: pd.DataFrame,
) -> list[tuple[float, float]]:
    x = pd.to_numeric(x_series, errors="coerce").dropna()

    if len(x) == 0:
        return []

    x_min = float(x.min())
    x_max = float(x.max())

    tw = _clean_transient_windows_for_plot(transient_windows)

    if tw.empty:
        return [(x_min, x_max)]

    intervals = []

    for _, item in tw.iterrows():
        intervals.append((
            float(item["start_decimal_year"]),
            float(item["end_decimal_year"]),
        ))

    intervals.sort(key=lambda item: item[0])

    merged = []

    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue

        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    stable = []
    cursor = x_min

    for start, end in merged:
        if start > cursor:
            stable.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < x_max:
        stable.append((cursor, x_max))

    return [
        (float(start), float(end))
        for start, end in stable
        if math.isfinite(start) and math.isfinite(end) and end > start
    ]


def _fit_linear_curve_for_enu_plot(
    df: pd.DataFrame,
    x_series: pd.Series,
    column: str,
    start_decimal_year: float,
    end_decimal_year: float,
    gaussian_width_days: float,
    min_points: int = 20,
) -> dict | None:
    if column not in df.columns:
        return None

    if not math.isfinite(start_decimal_year) or not math.isfinite(end_decimal_year):
        return None

    if end_decimal_year <= start_decimal_year:
        return None

    y_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column=column,
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if y_smooth is None:
        return None

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_smooth, errors="coerce")

    mask = (
        x.notna()
        & y.notna()
        & (x >= start_decimal_year)
        & (x <= end_decimal_year)
    )

    if int(mask.sum()) < int(min_points):
        return None

    x_fit = x.loc[mask].to_numpy(dtype=float)
    y_fit = y.loc[mask].to_numpy(dtype=float)

    try:
        coeff = np.polyfit(x_fit, y_fit, deg=1)
    except Exception:
        return None

    y_hat = np.polyval(coeff, x_fit)

    return {
        "x": x_fit,
        "y": y_hat,
        "slope_m_per_year": float(coeff[0]),
        "intercept_m": float(coeff[1]),
        "n_points": int(len(x_fit)),
    }


def _select_joint_quadratic_fit_for_window(
    joint_fits: pd.DataFrame,
    window: pd.Series,
) -> dict | None:
    if joint_fits is None or len(joint_fits) == 0:
        return None

    required = {"window_type", "model_name"}

    if not required.issubset(set(joint_fits.columns)):
        return None

    rows = joint_fits[
        (joint_fits["window_type"].astype(str) == str(window.get("window_type", "")))
        & (joint_fits["model_name"].astype(str) == "quadratic")
    ].copy()

    if "meta_cluster_id" in joint_fits.columns and "meta_cluster_id" in window.index:
        rows = rows[
            rows["meta_cluster_id"].astype(str)
            == str(window.get("meta_cluster_id", ""))
        ].copy()

    if rows.empty:
        return None

    if "best_by_bic" in rows.columns and rows["best_by_bic"].fillna(False).any():
        rows = rows[rows["best_by_bic"].fillna(False)].copy()

    return rows.iloc[0].to_dict()


def _select_component_quadratic_fit_for_window(
    component_fits: pd.DataFrame,
    window: pd.Series,
    component: str,
) -> dict | None:
    if component_fits is None or len(component_fits) == 0:
        return None

    required = {"window_type", "component", "model_name"}

    if not required.issubset(set(component_fits.columns)):
        return None

    rows = component_fits[
        (component_fits["window_type"].astype(str) == str(window.get("window_type", "")))
        & (component_fits["component"].astype(str) == str(component))
        & (component_fits["model_name"].astype(str) == "quadratic")
    ].copy()

    if "meta_cluster_id" in component_fits.columns and "meta_cluster_id" in window.index:
        rows = rows[
            rows["meta_cluster_id"].astype(str)
            == str(window.get("meta_cluster_id", ""))
        ].copy()

    if rows.empty:
        return None

    if "best_by_bic" in rows.columns and rows["best_by_bic"].fillna(False).any():
        rows = rows[rows["best_by_bic"].fillna(False)].copy()

    return rows.iloc[0].to_dict()


def _quadratic_curve_from_fit_row(
    x_values: np.ndarray,
    start_decimal_year: float,
    fit_row: dict,
    column: str,
) -> np.ndarray | None:
    t_days = (x_values - float(start_decimal_year)) * 365.25

    if column in {"E_m", "N_m"}:
        prefix = "E" if column == "E_m" else "N"

        intercept_key = f"{prefix}_intercept_mm"
        linear_key = f"{prefix}_linear_mm_per_day"
        quadratic_key = f"{prefix}_quadratic_mm_per_day2"

        if not all(key in fit_row for key in [intercept_key, linear_key, quadratic_key]):
            return None

        intercept_mm = float(fit_row[intercept_key])
        linear_mm_per_day = float(fit_row[linear_key])
        quadratic_mm_per_day2 = float(fit_row[quadratic_key])

    else:
        if not all(key in fit_row for key in ["intercept_mm", "linear_mm_per_day", "quadratic_mm_per_day2"]):
            return None

        intercept_mm = float(fit_row["intercept_mm"])
        linear_mm_per_day = float(fit_row["linear_mm_per_day"])
        quadratic_mm_per_day2 = float(fit_row["quadratic_mm_per_day2"])

    y_mm = (
        intercept_mm
        + linear_mm_per_day * t_days
        + quadratic_mm_per_day2 * (t_days ** 2)
    )

    return y_mm / 1000.0



def _velocity_label_from_smoothed_polynomial_fit(
    df: pd.DataFrame,
    x_series: pd.Series,
    column: str,
    start_decimal_year: float,
    end_decimal_year: float,
    gaussian_width_days: float,
    degree: int,
) -> dict | None:
    if column not in df.columns:
        return None

    if degree not in {1, 2}:
        return None

    if not math.isfinite(float(start_decimal_year)) or not math.isfinite(float(end_decimal_year)):
        return None

    if float(end_decimal_year) <= float(start_decimal_year):
        return None

    y_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column=column,
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if y_smooth is None:
        return None

    x = pd.to_numeric(x_series, errors="coerce")
    y_mm = pd.to_numeric(y_smooth, errors="coerce") * 1000.0

    mask = (
        x.notna()
        & y_mm.notna()
        & (x >= float(start_decimal_year))
        & (x <= float(end_decimal_year))
    )

    if int(mask.sum()) <= degree + 2:
        return None

    xw = x.loc[mask].to_numpy(dtype=float)
    yw = y_mm.loc[mask].to_numpy(dtype=float)

    t_days = (xw - float(start_decimal_year)) * 365.25

    try:
        coeff, cov = np.polyfit(t_days, yw, deg=degree, cov=True)
    except Exception:
        return None

    if cov is None:
        return None

    t_mid = 0.5 * (float(t_days.min()) + float(t_days.max()))
    x_mid = float(start_decimal_year) + t_mid / 365.25

    try:
        y_mid_mm = float(np.polyval(coeff, t_mid))
    except Exception:
        return None

    if degree == 1:
        velocity_mm_per_day = float(coeff[0])
        try:
            sigma_velocity_mm_per_day = float(math.sqrt(max(float(cov[0, 0]), 0.0)))
        except Exception:
            sigma_velocity_mm_per_day = math.nan
        label_prefix = "v"
    else:
        # coeff = [c, b, a] for y = c*t² + b*t + a
        c = float(coeff[0])
        b = float(coeff[1])

        velocity_mm_per_day = b + 2.0 * c * t_mid

        gradient = np.array([2.0 * t_mid, 1.0, 0.0], dtype=float)

        try:
            variance = float(gradient @ cov @ gradient.T)
            sigma_velocity_mm_per_day = float(math.sqrt(max(variance, 0.0)))
        except Exception:
            sigma_velocity_mm_per_day = math.nan

        label_prefix = "v_mid"

    velocity_mm_per_year = velocity_mm_per_day * 365.25
    sigma_velocity_mm_per_year = sigma_velocity_mm_per_day * 365.25

    if math.isfinite(sigma_velocity_mm_per_year):
        label = f"{label_prefix}: {velocity_mm_per_year:+.1f} ± {sigma_velocity_mm_per_year:.1f} mm yr⁻¹"
    else:
        label = f"{label_prefix}: {velocity_mm_per_year:+.1f} mm yr⁻¹"

    return {
        "x": x_mid,
        "y": y_mid_mm / 1000.0,
        "velocity_mm_per_year": velocity_mm_per_year,
        "sigma_velocity_mm_per_year": sigma_velocity_mm_per_year,
        "slope_m_per_year": velocity_mm_per_year / 1000.0,
        "label": label,
    }


def _enu_rotation_degrees_from_slope(
    ax,
    x: float,
    y: float,
    slope_m_per_year: float,
) -> float:
    try:
        x0_lim, x1_lim = ax.get_xlim()
        dx = 0.030 * abs(float(x1_lim) - float(x0_lim))
        if not math.isfinite(dx) or dx <= 0:
            dx = 0.10
    except Exception:
        dx = 0.10

    try:
        p0 = ax.transData.transform((x - dx, y - slope_m_per_year * dx))
        p1 = ax.transData.transform((x + dx, y + slope_m_per_year * dx))
        angle = math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))
    except Exception:
        angle = 0.0

    if not math.isfinite(angle):
        return 0.0

    return float(angle)


def _add_enu_velocity_label_to_axis(
    ax,
    label_info: dict | None,
) -> None:
    if label_info is None:
        return

    try:
        x = float(label_info["x"])
        y = float(label_info["y"])
        label = str(label_info["label"])
    except Exception:
        return

    if not math.isfinite(x) or not math.isfinite(y):
        return

    try:
        slope_m_per_year = float(label_info.get("slope_m_per_year", math.nan))
    except Exception:
        slope_m_per_year = math.nan

    if math.isfinite(slope_m_per_year):
        rotation = _enu_rotation_degrees_from_slope(
            ax=ax,
            x=x,
            y=y,
            slope_m_per_year=slope_m_per_year,
        )
    else:
        rotation = 0.0

    try:
        y0, y1 = ax.get_ylim()
        dy = 0.020 * abs(float(y1) - float(y0))
    except Exception:
        dy = 0.0

    ax.text(
        x,
        y + dy,
        label,
        color="black",
        fontsize=6.5,
        ha="center",
        va="bottom",
        rotation=rotation,
        rotation_mode="anchor",
        zorder=8,
        clip_on=True,
        bbox={
            "facecolor": "white",
            "alpha": 0.68,
            "edgecolor": "none",
            "pad": 1.3,
        },
    )




def _transient_boundary_markers_for_plot(
    transient_windows: pd.DataFrame,
) -> list[dict]:
    tw = _clean_transient_windows_for_plot(transient_windows)

    if tw.empty:
        return []

    markers = []

    # One marker at the start of each pre-event transient window.
    pre = tw[tw["window_type"].astype(str) == "pre_event_transient"].copy()
    for _, item in pre.iterrows():
        try:
            x = float(item["start_decimal_year"])
        except Exception:
            continue

        if math.isfinite(x):
            markers.append({
                "x": x,
                "label": "pre-event transient",
            })

    # One marker at the end of each post-event transient window.
    post = tw[tw["window_type"].astype(str) == "post_event_transient"].copy()
    for _, item in post.iterrows():
        try:
            x = float(item["end_decimal_year"])
        except Exception:
            continue

        if math.isfinite(x):
            markers.append({
                "x": x,
                "label": "post-event transient",
            })

    # Remove exact duplicates while preserving order.
    unique = []
    seen = set()

    for item in markers:
        key = (round(float(item["x"]), 10), str(item["label"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def _draw_transient_boundary_markers_on_axis(
    ax,
    transient_windows: pd.DataFrame,
    draw_labels: bool = False,
) -> None:
    markers = _transient_boundary_markers_for_plot(transient_windows)

    if not markers:
        return

    for item in markers:
        x = float(item["x"])
        label = str(item["label"])

        ax.axvline(
            x,
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.80,
            zorder=2,
        )

        if not draw_labels:
            continue

        ax.text(
            x,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="top",
            color="gray",
            fontsize=7.0,
            zorder=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.65,
                "edgecolor": "none",
                "pad": 1.2,
            },
            clip_on=True,
        )

def _draw_quadratic_transient_and_stable_linear_enu_overlay(
    ax,
    df: pd.DataFrame,
    x_series: pd.Series,
    column: str,
    transient_windows: pd.DataFrame,
    component_fits: pd.DataFrame,
    joint_fits: pd.DataFrame,
    gaussian_width_days: float,
    label_once: bool,
) -> bool:
    if column not in {"E_m", "N_m", "U_m"}:
        return label_once

    tw = _clean_transient_windows_for_plot(transient_windows)
    x = pd.to_numeric(x_series, errors="coerce")

    # A. Quadratic curves in pre/post transient windows.
    for _, window in tw.iterrows():
        window_type = str(window.get("window_type", ""))

        if window_type not in {"pre_event_transient", "post_event_transient"}:
            continue

        start_dec = float(window["start_decimal_year"])
        end_dec = float(window["end_decimal_year"])

        mask = (
            x.notna()
            & (x >= start_dec)
            & (x <= end_dec)
        )

        if int(mask.sum()) < 5:
            continue

        x_values = x.loc[mask].to_numpy(dtype=float)

        if column in {"E_m", "N_m"}:
            fit_row = _select_joint_quadratic_fit_for_window(joint_fits, window)
        else:
            fit_row = _select_component_quadratic_fit_for_window(component_fits, window, column)

        if fit_row is None:
            continue

        y_curve = _quadratic_curve_from_fit_row(
            x_values=x_values,
            start_decimal_year=start_dec,
            fit_row=fit_row,
            column=column,
        )

        if y_curve is None:
            continue

        ax.plot(
            x_values,
            y_curve,
            color="orange",
            linestyle="--",
            linewidth=0.9,
            alpha=0.95,
            zorder=5,
            label="quadratic transient fit" if not label_once else None,
        )

        label_info = _velocity_label_from_smoothed_polynomial_fit(
            df=df,
            x_series=x_series,
            column=column,
            start_decimal_year=start_dec,
            end_decimal_year=end_dec,
            gaussian_width_days=gaussian_width_days,
            degree=2,
        )
        _add_enu_velocity_label_to_axis(ax, label_info)

        label_once = True

    # B. Linear models in stable intervals outside pre/event/post windows.
    for start_dec, end_dec in _stable_intervals_outside_transient_windows(x_series, tw):
        fit = _fit_linear_curve_for_enu_plot(
            df=df,
            x_series=x_series,
            column=column,
            start_decimal_year=start_dec,
            end_decimal_year=end_dec,
            gaussian_width_days=gaussian_width_days,
            min_points=20,
        )

        if fit is None:
            continue

        ax.plot(
            fit["x"],
            fit["y"],
            color="green",
            linestyle="-",
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
            label="linear stable fit" if not label_once else None,
        )

        label_info = _velocity_label_from_smoothed_polynomial_fit(
            df=df,
            x_series=x_series,
            column=column,
            start_decimal_year=start_dec,
            end_decimal_year=end_dec,
            gaussian_width_days=gaussian_width_days,
            degree=1,
        )
        _add_enu_velocity_label_to_axis(ax, label_info)

        label_once = True

    return label_once

def _enu_composite_fullspan_plot_html(
    df: pd.DataFrame,
    shift_clusters: pd.DataFrame | None = None,
    meta_cluster_velocity_windows: pd.DataFrame | None = None,
    velocity_change_diagnostics: dict | None = None,
    report_analysis_config: dict | None = None,
) -> str:
    available = [col for col in ("E_m", "N_m", "U_m") if col in df.columns]

    if not available:
        return ""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except Exception:
        return ""

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    plot_cfg = analysis_cfg.get("plots", {})
    show_smoothed = bool(plot_cfg.get("show_gaussian_smoothed_enu", True))
    gaussian_width_days = float(plot_cfg.get("gaussian_width_days", 28.0))

    x_series, x_label = _decimal_year_series_for_composite_plot(df)
    x = x_series.to_numpy(dtype=float)

    selected_velocity_change_segments = _select_velocity_change_segments_for_plot(
        velocity_change_diagnostics=velocity_change_diagnostics,
        report_analysis_config=analysis_cfg,
    )

    shift_items = []
    if shift_clusters is not None and len(shift_clusters) > 0:
        for _, item in shift_clusters.iterrows():
            try:
                representative = float(item.get("representative_decimal_year", math.nan))
            except Exception:
                representative = math.nan

            try:
                start_dec = float(item.get("cluster_start_decimal_year", math.nan))
                end_dec = float(item.get("cluster_end_decimal_year", math.nan))
            except Exception:
                start_dec = math.nan
                end_dec = math.nan

            if not math.isfinite(start_dec):
                start_dec = representative
            if not math.isfinite(end_dec):
                end_dec = representative

            shift_items.append({
                "representative": representative,
                "start": start_dec,
                "end": end_dec,
            })

    fig, axes = plt.subplots(
        len(available),
        1,
        sharex=True,
        figsize=(10.8, 2.55 * len(available)),
        dpi=140,
    )

    if len(available) == 1:
        axes = [axes]

    for ax_index, (ax, column) in enumerate(zip(axes, available)):
        y = pd.to_numeric(df[column], errors="coerce")

        ax.plot(
            x,
            y,
            linestyle="-",
            color="blue",
            alpha=0.18,
            linewidth=0.8,
            label="raw ENU series",
            zorder=1,
        )

        if show_smoothed:
            y_smooth = _gaussian_smoothed_enu_series_for_plot(
                df=df,
                column=column,
                x_series=x_series,
                gaussian_width_days=gaussian_width_days,
            )

            if y_smooth is not None:
                ax.plot(
                    x,
                    y_smooth,
                    linewidth=1.6,
                    alpha=0.95,
                    label=f"Gaussian-smoothed ENU series, {gaussian_width_days:g} d",
                    zorder=3,
                )

        transient_windows_for_plot = _transient_windows_from_velocity_diagnostics(
            velocity_change_diagnostics
        )
        component_transient_fits_for_plot = _component_transient_fits_from_velocity_diagnostics(
            velocity_change_diagnostics
        )
        joint_horizontal_fits_for_plot = _joint_horizontal_transient_fits_from_velocity_diagnostics(
            velocity_change_diagnostics
        )

        if "_enu_fit_label_used" not in locals():
            _enu_fit_label_used = False

        _enu_fit_label_used = _draw_quadratic_transient_and_stable_linear_enu_overlay(
            ax=ax,
            df=df,
            x_series=x_series,
            column=column,
            transient_windows=transient_windows_for_plot,
            component_fits=component_transient_fits_for_plot,
            joint_fits=joint_horizontal_fits_for_plot,
            gaussian_width_days=gaussian_width_days,
            label_once=_enu_fit_label_used,
        )

        _draw_transient_boundary_markers_on_axis(
            ax=ax,
            transient_windows=transient_windows_for_plot,
            draw_labels=(ax_index == 0),
        )

        for k, shift in enumerate(shift_items):
            start_sx = shift["start"]
            end_sx = shift["end"]
            sx = shift["representative"]

            if math.isfinite(start_sx) and math.isfinite(end_sx):
                ax.axvspan(
                    min(start_sx, end_sx),
                    max(start_sx, end_sx),
                    color="gray",
                    alpha=0.16,
                    label="shift-cluster interval" if k == 0 else None,
                    zorder=0,
                )

            if math.isfinite(sx):
                ax.axvline(
                    sx,
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    label="representative shift epoch" if k == 0 else None,
                    zorder=4,
                )

        ax.set_ylabel(column)
        ax.grid(True)

        if x_label == "decimal year":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel(x_label)
    axes[0].set_title("ENU composite time series")

    _add_external_legend(fig, axes)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return (
        "<h3>ENU composite full-span plot</h3>"
        f'<img class="plot" src="data:image/png;base64,{encoded}" alt="ENU composite full-span plot">'
    )

def _plot_html(
    df: pd.DataFrame,
    plot_columns=None,
    failed_datasets: list[dict] | None = None,
    shift_clusters: pd.DataFrame | None = None,
    meta_cluster_velocity_windows: pd.DataFrame | None = None,
    velocity_change_diagnostics: dict | None = None,
    report_analysis_config: dict | None = None,
) -> str:
    resolved = _resolve_plot_columns(plot_columns)

    enu_composite_requested = any(column in ("E_m", "N_m", "U_m") for _, column in resolved)
    resolved = [(label, column) for label, column in resolved if column not in ("E_m", "N_m", "U_m")]

    if not resolved and not enu_composite_requested:
        return "<p>No plots requested.</p>"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
    except Exception as exc:
        return f"<p>Plots could not be generated because matplotlib is unavailable: {_html_escape(exc)}</p>"

    class _PlainOffsetScalarFormatter(ScalarFormatter):
        def get_offset(self):
            offset = getattr(self, "offset", 0)

            try:
                offset = float(offset)
            except Exception:
                return ""

            if offset == 0:
                return ""

            label = f"{offset:+.12f}".rstrip("0").rstrip(".")

            if label in {"+0", "-0"}:
                return ""

            return label

    def _apply_plain_y_offset_formatter(ax, column: str) -> None:
        if column not in {"X_m", "Y_m", "Z_m", "lon_deg", "lat_deg", "h_m"}:
            return

        formatter = _PlainOffsetScalarFormatter(useOffset=True, useMathText=False)
        formatter.set_useOffset(True)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

    failed_datasets = failed_datasets or []
    failed_x = []

    for item in failed_datasets:
        dec = _failed_dataset_decimal_year(item)
        if math.isfinite(dec):
            failed_x.append(dec)

    shift_x = []
    if shift_clusters is not None and len(shift_clusters) > 0:
        for report_event_id, (_, item) in enumerate(shift_clusters.iterrows(), start=1):
            try:
                dec = float(item.get("representative_decimal_year", math.nan))
            except Exception:
                dec = math.nan

            try:
                start_dec = float(item.get("cluster_start_decimal_year", math.nan))
                end_dec = float(item.get("cluster_end_decimal_year", math.nan))
            except Exception:
                start_dec = math.nan
                end_dec = math.nan

            if not math.isfinite(start_dec):
                start_dec = dec
            if not math.isfinite(end_dec):
                end_dec = dec

            candidates = []
            raw_candidates = item.get("candidate_decimal_years", "")

            if isinstance(raw_candidates, str):
                raw_candidate_values = [value.strip() for value in raw_candidates.split(",") if value.strip()]
            elif isinstance(raw_candidates, (list, tuple)):
                raw_candidate_values = raw_candidates
            else:
                raw_candidate_values = []

            for value in raw_candidate_values:
                try:
                    candidate_dec = float(value)
                except Exception:
                    candidate_dec = math.nan

                if math.isfinite(candidate_dec):
                    candidates.append(candidate_dec)

            if math.isfinite(dec):
                shift_x.append({
                    "x": dec,
                    "start": min(start_dec, end_dec),
                    "end": max(start_dec, end_dec),
                    "label": f"{dec:.4f}",
                    "report_event_id": report_event_id,
                    "candidates": candidates,
                })

    if "time_mean_all_epochs_utc" in df.columns:
        times = pd.to_datetime(df["time_mean_all_epochs_utc"], errors="coerce", utc=True)
        x = [_timestamp_to_decimal_year(t) for t in times]
        x_label = "decimal year"
    else:
        x = list(range(1, len(df) + 1))
        x_label = "solution index"
        failed_x = []
        shift_x = []

    x_series = pd.Series(x, index=df.index, dtype="float64")

    html_parts = []

    if enu_composite_requested:
        enu_composite_html = _enu_composite_fullspan_plot_html(
            df=df,
            shift_clusters=shift_clusters,
            meta_cluster_velocity_windows=meta_cluster_velocity_windows,
            velocity_change_diagnostics=velocity_change_diagnostics,
            report_analysis_config=report_analysis_config,
        )

        if enu_composite_html:
            html_parts.append(enu_composite_html)

    html_parts.append(
        "<p>The following plots are generated from the daily/per-file primary solutions stored in "
        "<code>timeseries.out</code>. The x-axis is shown as decimal year where time metadata are available.</p>"
    )

    if failed_x:
        html_parts.append(
            "<p>Red dashed vertical lines indicate datasets that did not complete successfully and are therefore "
            "not included in <code>timeseries.out</code>.</p>"
        )

    if shift_x:
        html_parts.append(
            "<p>Shift annotations are shown only on the E_m, N_m, and U_m full-span plots. "
            "Grey intervals indicate report-grade strict-cluster spans, and red dashed vertical "
            "lines indicate representative strict-cluster epochs. Accepted shift-candidate epochs "
            "are shown only in the zoom plots.</p>"
        )

    for label, column in resolved:
        if column not in df.columns:
            html_parts.append(f"<p>Requested plot skipped; missing column: {_html_escape(column)}</p>")
            continue

        y = pd.to_numeric(df[column], errors="coerce")

        if y.dropna().empty:
            html_parts.append(f"<p>Requested plot skipped; no numeric values in column: {_html_escape(column)}</p>")
            continue

        fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)

        if column in ("E_m", "N_m", "U_m"):
            ax.plot(
                x,
                y,
                linestyle="-",
                color="blue",
                alpha=0.18,
                linewidth=0.8,
                label="raw ENU series",
                zorder=1,
            )
        else:
            ax.plot(x, y, linestyle="-")
        ax.set_title(f"{column} time series")
        ax.set_xlabel(x_label)
        ax.set_ylabel(column)
        _apply_plain_y_offset_formatter(ax, column)
        ax.grid(True)

        if x_label == "decimal year":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        for k, fx in enumerate(failed_x):
            ax.axvline(
                fx,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
                label="non-successful dataset" if k == 0 else None,
            )

        analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
        plot_cfg = analysis_cfg["plots"]

        if column in ("E_m", "N_m", "U_m") and bool(plot_cfg.get("show_gaussian_smoothed_enu", True)):
            plot_gaussian_width_days = float(plot_cfg.get("gaussian_width_days", 28.0))

            y_smooth = _gaussian_smoothed_enu_series_for_plot(
                df=df,
                column=column,
                x_series=x_series,
                gaussian_width_days=plot_gaussian_width_days,
            )

            if y_smooth is not None:
                ax.plot(
                    x,
                    y_smooth,
                    linewidth=1.6,
                    alpha=0.95,
                    label=f"Gaussian-smoothed ENU series, {plot_gaussian_width_days:g} d",
                    zorder=3,
                )

        velocity_trends_drawn = _draw_velocity_trends_on_fullspan_axis(
            ax=ax,
            df=df,
            column=column,
            x_series=x_series,
            velocity_windows=meta_cluster_velocity_windows,
        )

        show_shift_annotations = column in ("E_m", "N_m", "U_m")

        if show_shift_annotations:
            for k, item in enumerate(shift_x):
                sx = item["x"]
                start_sx = item.get("start", sx)
                end_sx = item.get("end", sx)

                if (
                    math.isfinite(start_sx)
                    and math.isfinite(end_sx)
                    and abs(end_sx - start_sx) > 0
                ):
                    ax.axvspan(
                        start_sx,
                        end_sx,
                        color="gray",
                        alpha=0.12,
                        label="shift-cluster interval" if k == 0 else None,
                    )

                ax.axvline(
                    sx,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.95,
                    label="representative shift epoch" if k == 0 else None,
                )
                ax.text(
                    sx,
                    -0.13,
                    item["label"],
                    transform=ax.get_xaxis_transform(),
                    color="red",
                    fontsize=7,
                    rotation=90,
                    ha="center",
                    va="top",
                    clip_on=False,
                )

        if failed_x or (shift_x and column in ("E_m", "N_m", "U_m")) or velocity_trends_drawn:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )

        if shift_x and column in ("E_m", "N_m", "U_m"):
            fig.tight_layout(rect=[0, 0.08, 0.80, 1])
        elif failed_x or velocity_trends_drawn:
            fig.tight_layout(rect=[0, 0, 0.80, 1])
        else:
            fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        html_parts.append(f"<h3>{_html_escape(column)}</h3>")
        html_parts.append(
            f'<img class="plot" src="data:image/png;base64,{encoded}" alt="{_html_escape(column)} plot">'
        )

    return "\n".join(html_parts)
def _daily_summary_html(df: pd.DataFrame) -> str:
    columns = [
        "run_label",
        "dataset_name",
        "station_id",
        "time_mean_all_epochs_utc",
        "convergence_epoch_utc",
        "convergence_delay_sec",
        "X_m",
        "Y_m",
        "Z_m",
        "lon_deg",
        "lat_deg",
        "h_m",
        "E_m",
        "N_m",
        "U_m",
        "qc_unsmoothed_conv_std_E_m",
        "qc_unsmoothed_conv_std_N_m",
        "qc_unsmoothed_conv_std_U_m",
        "trace_warning_count",
        "trace_critical_warning_count",
        "qc_flags",
    ]

    return _df_to_html_table(df, columns)


def _parse_candidate_decimal_years(value) -> list[float]:
    if value is None:
        return []

    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return []

    out = []

    for item in parts:
        try:
            dec = float(item)
        except Exception:
            dec = math.nan

        if math.isfinite(dec):
            out.append(dec)

    return out


def _decimal_year_series_from_times(df: pd.DataFrame) -> pd.Series:
    if "time_mean_all_epochs_utc" not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index)

    times = pd.to_datetime(df["time_mean_all_epochs_utc"], errors="coerce", utc=True)
    values = []

    for t in times:
        if pd.isna(t):
            values.append(math.nan)
            continue

        start = pd.Timestamp(year=int(t.year), month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=int(t.year) + 1, month=1, day=1, tz="UTC")
        values.append(float(int(t.year) + (t - start).total_seconds() / (end - start).total_seconds()))

    return pd.Series(values, index=df.index)


def _shift_cluster_zoom_plots_html(df: pd.DataFrame, shift_clusters: pd.DataFrame | None = None) -> str:
    if shift_clusters is None or len(shift_clusters) == 0:
        return "<p>No report-grade shift clusters available for zoom plotting.</p>"

    required_columns = ["E_m", "N_m", "U_m"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        return "<p>Shift-cluster zoom plots could not be generated because required ENU columns are missing.</p>"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        return f"<p>Shift-cluster zoom plots could not be generated because matplotlib is unavailable: {_html_escape(exc)}</p>"

    x = _decimal_year_series_from_times(df)
    html_parts = []

    html_parts.append(
        "<p>The following plots zoom into the temporal span of each report-grade shift cluster. "
        "They show the ENU components only, with accepted shift candidates marked in pale purple "
        "and the representative shift epoch marked in red.</p>"
    )

    for report_event_id, (_, cluster) in enumerate(shift_clusters.iterrows(), start=1):
        try:
            start_dec = float(cluster.get("cluster_start_decimal_year", math.nan))
            end_dec = float(cluster.get("cluster_end_decimal_year", math.nan))
            rep_dec = float(cluster.get("representative_decimal_year", math.nan))
        except Exception:
            start_dec = math.nan
            end_dec = math.nan
            rep_dec = math.nan

        if not math.isfinite(start_dec) or not math.isfinite(end_dec):
            continue

        lo = min(start_dec, end_dec)
        hi = max(start_dec, end_dec)

        if not math.isfinite(rep_dec):
            rep_dec = (lo + hi) / 2.0

        candidates = _parse_candidate_decimal_years(cluster.get("candidate_decimal_years", ""))

        mask = (x >= lo) & (x <= hi)
        sub = df.loc[mask].copy()
        sub_x = x.loc[mask]

        if len(sub) == 0:
            continue

        internal_cluster_id = cluster.get("cluster_id", "")
        event_class = _shift_cluster_event_class(cluster)
        components = cluster.get("components", "")

        html_parts.append(
            f"<h3>Shift-cluster zoom plots — report event {report_event_id}, "
            f"internal cluster { _html_escape(internal_cluster_id) }</h3>"
        )
        html_parts.append(
            "<p>"
            f"Event class: <code>{_html_escape(event_class)}</code>; "
            f"components: <code>{_html_escape(components)}</code>; "
            f"cluster interval: <code>{_fmt(lo, digits=4)}–{_fmt(hi, digits=4)}</code>; "
            f"representative epoch: <code>{_fmt(rep_dec, digits=4)}</code>."
            "</p>"
        )

        for column in required_columns:
            y = pd.to_numeric(sub[column], errors="coerce")

            fig, ax = plt.subplots(figsize=(10.5, 4.2))
            ax.plot(sub_x, y, linewidth=1.3)

            ax.axvspan(
                lo,
                hi,
                color="gray",
                alpha=0.12,
                label="shift-cluster interval",
            )

            for idx, candidate_x in enumerate(candidates):
                if candidate_x < lo or candidate_x > hi:
                    continue

                ax.axvline(
                    candidate_x,
                    color="purple",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.35,
                    label="accepted shift candidate" if idx == 0 else None,
                )
                ax.text(
                    candidate_x,
                    -0.26,
                    f"{candidate_x:.4f}",
                    transform=ax.get_xaxis_transform(),
                    color="purple",
                    fontsize=5,
                    rotation=90,
                    ha="center",
                    va="top",
                    alpha=0.50,
                    clip_on=False,
                )

            ax.axvline(
                rep_dec,
                color="red",
                linestyle="--",
                linewidth=1.3,
                alpha=0.95,
                label="representative shift epoch",
            )
            ax.text(
                rep_dec,
                -0.12,
                f"{rep_dec:.4f}",
                transform=ax.get_xaxis_transform(),
                color="red",
                fontsize=7,
                rotation=90,
                ha="center",
                va="top",
                clip_on=False,
            )

            ax.set_title(f"{column} zoom — report event {report_event_id}")
            ax.set_xlabel("decimal year")
            ax.set_ylabel(column)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            ax.grid(True)
            ax.legend(loc="best")
            fig.tight_layout(rect=[0, 0.18, 1, 1])

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=140)
            plt.close(fig)

            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            html_parts.append(
                f'<img class="plot" src="data:image/png;base64,{encoded}" alt="{_html_escape(column)} zoom plot">'
            )

    if len(html_parts) == 1:
        html_parts.append("<p>No valid shift-cluster zoom plots were generated.</p>")

    return chr(10).join(html_parts)


def _compute_report_meta_clusters(strict_clusters: pd.DataFrame, report_analysis_config: dict | None = None) -> pd.DataFrame:
    if strict_clusters is None or len(strict_clusters) == 0:
        return pd.DataFrame()

    try:
        import timeseries_change_detection as tcd
    except Exception:
        return pd.DataFrame()

    if not hasattr(tcd, "MetaClusteringConfig") or not hasattr(tcd, "create_meta_clusters"):
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    meta_cfg = analysis_cfg["meta_clustering"]

    cfg = tcd.MetaClusteringConfig(
        enabled=bool(meta_cfg["enabled"]),
        max_gap_days=float(meta_cfg["max_gap_days"]),
        enable_direction_similarity=bool(meta_cfg["enable_direction_similarity"]),
        direction_mode=str(meta_cfg["direction_mode"]),
        max_direction_change_deg=float(meta_cfg["max_direction_change_deg"]),
        enable_magnitude_compatibility=bool(meta_cfg["enable_magnitude_compatibility"]),
        max_magnitude_ratio=float(meta_cfg["max_magnitude_ratio"]),
    )

    try:
        return tcd.create_meta_clusters(strict_clusters, cfg)
    except Exception:
        return pd.DataFrame()


def _meta_clusters_html(meta_clusters: pd.DataFrame) -> str:
    text = []

    text.append(
        "<p>Level-2 meta-clustering groups neighbouring report-grade strict clusters. "
        "Current fixed V1 meta-clustering defaults are: "
        "<code>max_gap_days = 14</code>, "
        "<code>direction_mode = horizontal</code>, "
        "<code>maximum adjacent direction-change tolerance = 45 degrees</code>, and "
        "<code>magnitude compatibility = disabled</code>.</p>"
    )

    if meta_clusters is None or len(meta_clusters) == 0:
        text.append("<p>No report-grade meta-clusters were created.</p>")
        return chr(10).join(text)

    rows = []

    for _, item in meta_clusters.iterrows():
        rows.append([
            _html_escape(item.get("meta_cluster_id", "")),
            _html_escape(item.get("strict_cluster_ids", "")),
            _html_escape(item.get("n_strict_clusters", "")),
            _html_escape(_fmt(item.get("meta_start_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(item.get("meta_end_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(item.get("meta_duration_days", math.nan), digits=3)),
            _html_escape(_fmt(item.get("representative_decimal_year", math.nan), digits=4)),
            _html_escape(item.get("representative_time_utc", "")),
            _html_escape(item.get("components", "")),
            _html_escape(_fmt(item.get("E_net_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("N_net_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("U_net_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("horizontal_net_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("total_net_jump_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("max_gap_days", math.nan), digits=3)),
            _html_escape(_fmt(item.get("max_adjacent_direction_change_deg", math.nan), digits=3)),
            _html_escape(_fmt(item.get("cumulative_rotation_deg", math.nan), digits=3)),
            _html_escape(item.get("direction_behaviour", "")),
        ])

    text.append(_html_table(
        [
            "Meta event",
            "Strict cluster IDs",
            "Strict cluster count",
            "Meta start decimal year",
            "Meta end decimal year",
            "Meta duration (days)",
            "Representative decimal year",
            "Representative UTC",
            "Components",
            "E net jump (mm)",
            "N net jump (mm)",
            "U net jump (mm)",
            "Horizontal net jump (mm)",
            "Total net jump (mm)",
            "Max gap (days)",
            "Max adjacent direction change (deg)",
            "Cumulative rotation (deg)",
            "Direction behaviour",
        ],
        rows,
    ))

    text.append(
        "<p>Net jumps are computed from the strict-cluster representative component jumps. "
        "A missing component in a strict cluster means that no accepted component candidate "
        "was assigned to that strict cluster, not that the physical displacement is proven zero.</p>"
    )

    return chr(10).join(text)


def _compute_meta_cluster_velocity_windows(df: pd.DataFrame, meta_clusters: pd.DataFrame, report_analysis_config: dict | None = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if meta_clusters is None or len(meta_clusters) == 0:
        return pd.DataFrame()

    try:
        import timeseries_velocity_detection as tvd
    except Exception:
        return pd.DataFrame()

    if not hasattr(tvd, "MetaClusterVelocityWindowConfig"):
        return pd.DataFrame()

    if not hasattr(tvd, "fit_velocity_windows_around_meta_clusters"):
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    vel_cfg = analysis_cfg["velocity_windows"]

    cfg = tvd.MetaClusterVelocityWindowConfig(
        columns=("E_m", "N_m", "U_m"),
        min_points_per_window=int(vel_cfg["min_points_per_window"]),
        min_duration_days_for_stable_rate=float(vel_cfg["min_duration_days_for_stable_rate"]),
        apply_gaussian_smoothing=bool(vel_cfg["apply_gaussian_smoothing"]),
        gaussian_width_days=float(vel_cfg["gaussian_width_days"]),
        outlier_rejection_enabled=bool(vel_cfg["outlier_rejection_enabled"]),
        sigma_floor_m=float(vel_cfg["sigma_floor_m"]),
    )

    try:
        return tvd.fit_velocity_windows_around_meta_clusters(
            df=df,
            meta_clusters=meta_clusters,
            cfg=cfg,
        )
    except Exception:
        return pd.DataFrame()


def _meta_cluster_velocity_windows_html(velocity_windows: pd.DataFrame) -> str:
    text = []

    text.append(
        "<p>This section estimates component-wise rates around each report-grade meta-cluster. "
        "For <code>before_meta_cluster</code> and <code>after_meta_cluster</code>, the reported value is a "
        "stable-window velocity estimated from Gaussian-smoothed ENU series. Current fixed V1 defaults are: "
        "<code>minimum stable velocity period = 365.25 days</code>, "
        "<code>Gaussian smoothing width = 28 days</code>, and robust pre-smoothing outlier rejection. "
        "For <code>during_meta_cluster</code>, the reported value is a transition rate derived from the "
        "meta-cluster net displacement divided by the meta-cluster duration; it is not treated as a secular "
        "geodetic velocity.</p>"
    )

    if velocity_windows is None or len(velocity_windows) == 0:
        text.append("<p>No meta-cluster velocity-window estimates were generated.</p>")
        return chr(10).join(text)

    rows = []

    for _, item in velocity_windows.iterrows():
        rows.append([
            _html_escape(item.get("meta_cluster_id", "")),
            _html_escape(item.get("strict_cluster_ids", "")),
            _html_escape(item.get("component", "")),
            _html_escape(item.get("window_label", "")),
            _html_escape(item.get("rate_class", "")),
            _html_escape(_fmt(item.get("start_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(item.get("end_decimal_year", math.nan), digits=4)),
            _html_escape(_fmt(item.get("duration_days", math.nan), digits=3)),
            _html_escape(item.get("n_points", "")),
            _html_escape(item.get("n_points_used_for_smoothing", "")),
            _html_escape(item.get("series_used_for_fit", "")),
            _html_escape(_fmt(item.get("gaussian_width_days", math.nan), digits=3)),
            _html_escape(_fmt(item.get("velocity_mm_per_year", math.nan), digits=3)),
            _html_escape(_fmt(item.get("estimated_displacement_mm", math.nan), digits=3)),
            _html_escape(_fmt(item.get("sigma_m", math.nan), digits=6)),
            _html_escape(item.get("quality_flag", "")),
        ])

    text.append(_html_table(
        [
            "Meta event",
            "Strict cluster IDs",
            "Component",
            "Window",
            "Rate class",
            "Start decimal year",
            "End decimal year",
            "Duration (days)",
            "Points",
            "Points used for smoothing",
            "Series used",
            "Gaussian width (days)",
            "Rate (mm/yr)",
            "Estimated displacement (mm)",
            "Sigma (m)",
            "Quality flag",
        ],
        rows,
    ))

    text.append(
        "<p><strong>Interpretation note:</strong> rows marked <code>transition_rate</code> describe the "
        "mean transition rate implied by the detected meta-cluster displacement. They should not be interpreted "
        "as long-term station velocities. Rows marked <code>stable_window_velocity</code> are velocity estimates "
        "for the pre- and post-event windows, provided the window duration passes the minimum stable-period criterion.</p>"
    )

    return chr(10).join(text)



def _decimal_year_to_calendar_date_label(decimal_year) -> str:
    try:
        dec = float(decimal_year)
    except Exception:
        return ""

    if not math.isfinite(dec):
        return ""

    year = int(math.floor(dec))
    frac = dec - year

    try:
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        t = start + pd.to_timedelta(frac * (end - start).total_seconds(), unit="s")
        return t.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _component_window_displacement_summary(
    df: pd.DataFrame,
    component: str,
    x_series: pd.Series,
    x0: float,
    x1: float,
    gaussian_width_days: float,
) -> dict:
    empty = {
        f"{component}_start_mm": math.nan,
        f"{component}_end_mm": math.nan,
        f"{component}_net_mm": math.nan,
        f"{component}_range_mm": math.nan,
    }

    if component not in df.columns:
        return empty

    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return empty

    y_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column=component,
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if y_smooth is None:
        y_smooth = pd.to_numeric(df[component], errors="coerce")

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_smooth, errors="coerce")

    mask = (x >= x0) & (x <= x1) & x.notna() & y.notna()

    xw = x.loc[mask]
    yw = y.loc[mask]

    if len(yw) < 2:
        return empty

    start_mm = float(yw.iloc[0]) * 1000.0
    end_mm = float(yw.iloc[-1]) * 1000.0
    net_mm = end_mm - start_mm
    range_mm = float((yw.max() - yw.min()) * 1000.0)

    return {
        f"{component}_start_mm": start_mm,
        f"{component}_end_mm": end_mm,
        f"{component}_net_mm": net_mm,
        f"{component}_range_mm": range_mm,
    }


def _compute_transient_windows_from_velocity_change_clusters(
    df: pd.DataFrame,
    meta_clusters: pd.DataFrame,
    velocity_change_diagnostics: dict,
    report_analysis_config: dict | None = None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if meta_clusters is None or len(meta_clusters) == 0:
        return pd.DataFrame()

    classified = velocity_change_diagnostics.get("classified", pd.DataFrame())

    if classified is None or len(classified) == 0:
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    plot_cfg = analysis_cfg.get("plots", {})
    gaussian_width_days = float(plot_cfg.get("gaussian_width_days", 28.0))

    x_series, _ = _decimal_year_series_for_composite_plot(df)

    related_h = classified.copy()

    required = {
        "component",
        "report_grade",
        "shift_related_velocity_change",
        "cluster_start_decimal_year",
        "cluster_end_decimal_year",
    }

    if not required.issubset(set(related_h.columns)):
        return pd.DataFrame()

    related_h = related_h[
        (related_h["component"].astype(str) == "H_magnitude")
        & (related_h["report_grade"] == True)
        & (related_h["shift_related_velocity_change"] == True)
    ].copy()

    if len(related_h) == 0:
        return pd.DataFrame()

    for col in ["cluster_start_decimal_year", "cluster_end_decimal_year"]:
        related_h[col] = pd.to_numeric(related_h[col], errors="coerce")

    related_h = related_h.dropna(subset=["cluster_start_decimal_year", "cluster_end_decimal_year"])

    rows = []

    for _, meta in meta_clusters.iterrows():
        try:
            meta_id = meta.get("meta_cluster_id", meta.get("cluster_id", ""))
        except Exception:
            meta_id = ""

        try:
            meta_start = float(meta.get("meta_start_decimal_year", math.nan))
            meta_end = float(meta.get("meta_end_decimal_year", math.nan))
        except Exception:
            meta_start = math.nan
            meta_end = math.nan

        if not math.isfinite(meta_start) or not math.isfinite(meta_end):
            continue

        if meta_end < meta_start:
            meta_start, meta_end = meta_end, meta_start

        meta_duration_days = float((meta_end - meta_start) * 365.25)

        pre_candidates = related_h[related_h["cluster_start_decimal_year"] < meta_start]
        post_candidates = related_h[related_h["cluster_end_decimal_year"] > meta_end]

        pre_start = math.nan
        pre_end = math.nan

        if len(pre_candidates) > 0:
            pre_start = float(pre_candidates["cluster_start_decimal_year"].min())
            pre_end = meta_start

        post_start = math.nan
        post_end = math.nan

        if len(post_candidates) > 0:
            post_start = meta_end
            post_end = float(post_candidates["cluster_end_decimal_year"].max())

        # Event/incidence row from meta-cluster net displacement.
        event_row = {
            "meta_cluster_id": meta_id,
            "window_type": "event_incidence",
            "interpretation": "net displacement and deformation rate during incidence; no velocity fit",
            "start_decimal_year": meta_start,
            "end_decimal_year": meta_end,
            "start_date": _decimal_year_to_calendar_date_label(meta_start),
            "end_date": _decimal_year_to_calendar_date_label(meta_end),
            "duration_days": meta_duration_days,
            "source": "meta_cluster_net_displacement",
        }

        for comp, meta_col in [
            ("E_m", "E_net_jump_mm"),
            ("N_m", "N_net_jump_mm"),
            ("U_m", "U_net_jump_mm"),
        ]:
            try:
                net_mm = float(meta.get(meta_col, math.nan))
            except Exception:
                net_mm = math.nan

            event_row[f"{comp}_net_mm"] = net_mm
            event_row[f"{comp}_range_mm"] = math.nan

            if math.isfinite(net_mm) and meta_duration_days > 0:
                event_row[f"{comp}_rate_mm_per_day"] = net_mm / meta_duration_days
            else:
                event_row[f"{comp}_rate_mm_per_day"] = math.nan

        try:
            h_net = float(meta.get("horizontal_net_jump_mm", math.nan))
        except Exception:
            h_net = math.nan

        event_row["H_net_mm"] = h_net
        event_row["H_range_mm"] = math.nan
        event_row["H_rate_mm_per_day"] = h_net / meta_duration_days if math.isfinite(h_net) and meta_duration_days > 0 else math.nan

        rows.append(event_row)

        # Pre-event transient row.
        if math.isfinite(pre_start) and math.isfinite(pre_end) and pre_end > pre_start:
            pre_row = {
                "meta_cluster_id": meta_id,
                "window_type": "pre_event_transient",
                "interpretation": "automatic transient window inferred from shift-related report-grade velocity-change clusters; excluded from stable rolling velocity interpretation",
                "start_decimal_year": pre_start,
                "end_decimal_year": pre_end,
                "start_date": _decimal_year_to_calendar_date_label(pre_start),
                "end_date": _decimal_year_to_calendar_date_label(pre_end),
                "duration_days": float((pre_end - pre_start) * 365.25),
                "source": "velocity_change_clusters",
            }

            for comp in ["E_m", "N_m", "U_m"]:
                pre_row.update(_component_window_displacement_summary(
                    df=df,
                    component=comp,
                    x_series=x_series,
                    x0=pre_start,
                    x1=pre_end,
                    gaussian_width_days=gaussian_width_days,
                ))

            e_net = pre_row.get("E_m_net_mm", math.nan)
            n_net = pre_row.get("N_m_net_mm", math.nan)

            if math.isfinite(e_net) and math.isfinite(n_net):
                pre_row["H_net_mm"] = math.sqrt(e_net ** 2 + n_net ** 2)
            else:
                pre_row["H_net_mm"] = math.nan

            rows.append(pre_row)

        # Post-event transient row.
        if math.isfinite(post_start) and math.isfinite(post_end) and post_end > post_start:
            post_row = {
                "meta_cluster_id": meta_id,
                "window_type": "post_event_transient",
                "interpretation": "automatic transient window inferred from shift-related report-grade velocity-change clusters; excluded from stable rolling velocity interpretation",
                "start_decimal_year": post_start,
                "end_decimal_year": post_end,
                "start_date": _decimal_year_to_calendar_date_label(post_start),
                "end_date": _decimal_year_to_calendar_date_label(post_end),
                "duration_days": float((post_end - post_start) * 365.25),
                "source": "velocity_change_clusters",
            }

            for comp in ["E_m", "N_m", "U_m"]:
                post_row.update(_component_window_displacement_summary(
                    df=df,
                    component=comp,
                    x_series=x_series,
                    x0=post_start,
                    x1=post_end,
                    gaussian_width_days=gaussian_width_days,
                ))

            e_net = post_row.get("E_m_net_mm", math.nan)
            n_net = post_row.get("N_m_net_mm", math.nan)

            if math.isfinite(e_net) and math.isfinite(n_net):
                post_row["H_net_mm"] = math.sqrt(e_net ** 2 + n_net ** 2)
            else:
                post_row["H_net_mm"] = math.nan

            rows.append(post_row)

    return pd.DataFrame(rows)



def _transient_fit_design_matrix(
    t_days: np.ndarray,
    model_name: str,
    tau_days: float | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    if model_name == "linear":
        return np.column_stack([np.ones_like(t_days), t_days]), [
            "intercept_mm",
            "linear_mm_per_day",
        ]

    if model_name == "quadratic":
        return np.column_stack([np.ones_like(t_days), t_days, t_days ** 2]), [
            "intercept_mm",
            "linear_mm_per_day",
            "quadratic_mm_per_day2",
        ]

    if model_name == "exponential":
        if tau_days is None or not math.isfinite(float(tau_days)) or float(tau_days) <= 0:
            return None, []

        basis = 1.0 - np.exp(-t_days / float(tau_days))

        return np.column_stack([np.ones_like(t_days), t_days, basis]), [
            "intercept_mm",
            "linear_mm_per_day",
            "amplitude_mm",
        ]

    if model_name == "logarithmic":
        if tau_days is None or not math.isfinite(float(tau_days)) or float(tau_days) <= 0:
            return None, []

        basis = np.log1p(t_days / float(tau_days))

        return np.column_stack([np.ones_like(t_days), t_days, basis]), [
            "intercept_mm",
            "linear_mm_per_day",
            "amplitude_mm",
        ]

    return None, []


def _fit_transient_model_least_squares(
    t_days: np.ndarray,
    y_mm: np.ndarray,
    model_name: str,
    tau_days: float | None = None,
) -> dict:
    X, names = _transient_fit_design_matrix(t_days, model_name, tau_days=tau_days)

    if X is None or len(names) == 0:
        return {"ok": False, "reason": "invalid_design_matrix"}

    n = int(len(y_mm))
    k = int(X.shape[1])

    if n <= k + 1:
        return {"ok": False, "reason": "insufficient_points"}

    try:
        beta, residuals, rank, singular_values = np.linalg.lstsq(X, y_mm, rcond=None)
    except Exception as exc:
        return {"ok": False, "reason": f"least_squares_failed: {exc}"}

    if rank < k:
        return {"ok": False, "reason": "rank_deficient"}

    y_hat = X @ beta
    residual = y_mm - y_hat
    rss = float(np.sum(residual ** 2))

    if not math.isfinite(rss):
        return {"ok": False, "reason": "invalid_rss"}

    rss_for_ic = max(rss, 1.0e-12)

    rms_mm = float(math.sqrt(rss / n))
    aic = float(n * math.log(rss_for_ic / n) + 2 * k)
    bic = float(n * math.log(rss_for_ic / n) + k * math.log(n))

    out = {
        "ok": True,
        "model_name": model_name,
        "tau_days": float(tau_days) if tau_days is not None else math.nan,
        "n_points": n,
        "n_parameters": k,
        "rank": int(rank),
        "rss_mm2": rss,
        "rms_mm": rms_mm,
        "aic": aic,
        "bic": bic,
        "reason": "",
    }

    for name, value in zip(names, beta):
        out[name] = float(value)

    for name in [
        "intercept_mm",
        "linear_mm_per_day",
        "quadratic_mm_per_day2",
        "amplitude_mm",
    ]:
        if name not in out:
            out[name] = math.nan

    return out


def _tau_grid_for_transient_fit(duration_days: float) -> list[float]:
    if not math.isfinite(duration_days) or duration_days <= 0:
        return []

    lo = max(3.0, 0.03 * duration_days)
    hi = max(lo * 1.01, 3.0 * duration_days)

    try:
        grid = np.geomspace(lo, hi, 24)
    except Exception:
        return []

    return [float(x) for x in grid if math.isfinite(float(x)) and float(x) > 0]


def _fit_transient_models_for_component(
    df: pd.DataFrame,
    component: str,
    x_series: pd.Series,
    x0: float,
    x1: float,
    gaussian_width_days: float,
) -> pd.DataFrame:
    if component not in df.columns:
        return pd.DataFrame()

    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return pd.DataFrame()

    y_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column=component,
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if y_smooth is None:
        y_smooth = pd.to_numeric(df[component], errors="coerce")

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_smooth, errors="coerce") * 1000.0

    mask = (x >= x0) & (x <= x1) & x.notna() & y.notna()

    xw = x.loc[mask].to_numpy(dtype=float)
    yw = y.loc[mask].to_numpy(dtype=float)

    if len(xw) < 10:
        return pd.DataFrame()

    t_days = (xw - float(x0)) * 365.25
    duration_days = float((float(x1) - float(x0)) * 365.25)

    rows = []

    for model_name in ["linear", "quadratic"]:
        fit = _fit_transient_model_least_squares(
            t_days=t_days,
            y_mm=yw,
            model_name=model_name,
            tau_days=None,
        )

        if fit.get("ok", False):
            rows.append(fit)

    for model_name in []:  # exponential/logarithmic disabled; quadratic-only transient model
        best = None

        for tau in _tau_grid_for_transient_fit(duration_days):
            fit = _fit_transient_model_least_squares(
                t_days=t_days,
                y_mm=yw,
                model_name=model_name,
                tau_days=tau,
            )

            if not fit.get("ok", False):
                continue

            if best is None or float(fit["bic"]) < float(best["bic"]):
                best = fit

        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["component"] = component
    out["duration_days"] = duration_days

    best_bic = float(out["bic"].min())

    out["delta_bic_from_best"] = out["bic"] - best_bic
    out["best_by_bic"] = out["bic"] == best_bic

    return out.sort_values(["best_by_bic", "bic"], ascending=[False, True]).reset_index(drop=True)


def _compute_transient_model_fits(
    df: pd.DataFrame,
    transient_windows: pd.DataFrame,
    report_analysis_config: dict | None = None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if transient_windows is None or len(transient_windows) == 0:
        return pd.DataFrame()

    required = {
        "window_type",
        "start_decimal_year",
        "end_decimal_year",
    }

    if not required.issubset(set(transient_windows.columns)):
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    plot_cfg = analysis_cfg.get("plots", {})
    gaussian_width_days = float(plot_cfg.get("gaussian_width_days", 28.0))

    x_series, _ = _decimal_year_series_for_composite_plot(df)

    rows = []

    work = transient_windows.copy()
    work = work[work["window_type"].astype(str).isin(["pre_event_transient", "post_event_transient"])].copy()

    if len(work) == 0:
        return pd.DataFrame()

    for _, window in work.iterrows():
        try:
            x0 = float(window.get("start_decimal_year", math.nan))
            x1 = float(window.get("end_decimal_year", math.nan))
        except Exception:
            continue

        for component in ["E_m", "N_m", "U_m"]:
            fits = _fit_transient_models_for_component(
                df=df,
                component=component,
                x_series=x_series,
                x0=x0,
                x1=x1,
                gaussian_width_days=gaussian_width_days,
            )

            if fits is None or len(fits) == 0:
                continue

            for _, fit in fits.iterrows():
                row = fit.to_dict()

                row["meta_cluster_id"] = window.get("meta_cluster_id", "")
                row["window_type"] = window.get("window_type", "")
                row["start_decimal_year"] = x0
                row["end_decimal_year"] = x1
                row["start_date"] = window.get("start_date", "")
                row["end_date"] = window.get("end_date", "")
                row["gaussian_width_days"] = gaussian_width_days

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    order_cols = [
        "meta_cluster_id",
        "window_type",
        "component",
        "best_by_bic",
        "bic",
    ]

    existing_order = [col for col in order_cols if col in out.columns]

    if existing_order:
        out = out.sort_values(existing_order, ascending=[True, True, True, False, True]).reset_index(drop=True)

    return out



def _fit_joint_horizontal_transient_model(
    t_days: np.ndarray,
    e_mm: np.ndarray,
    n_mm: np.ndarray,
    model_name: str,
    tau_days: float | None = None,
) -> dict:
    X, names = _transient_fit_design_matrix(t_days, model_name, tau_days=tau_days)

    if X is None or len(names) == 0:
        return {"ok": False, "reason": "invalid_design_matrix"}

    n_obs = int(len(t_days))
    p = int(X.shape[1])

    if n_obs <= p + 1:
        return {"ok": False, "reason": "insufficient_points"}

    try:
        beta_e, _, rank_e, _ = np.linalg.lstsq(X, e_mm, rcond=None)
        beta_n, _, rank_n, _ = np.linalg.lstsq(X, n_mm, rcond=None)
    except Exception as exc:
        return {"ok": False, "reason": f"least_squares_failed: {exc}"}

    if rank_e < p or rank_n < p:
        return {"ok": False, "reason": "rank_deficient"}

    e_hat = X @ beta_e
    n_hat = X @ beta_n

    res_e = e_mm - e_hat
    res_n = n_mm - n_hat

    rss_e = float(np.sum(res_e ** 2))
    rss_n = float(np.sum(res_n ** 2))
    rss_total = rss_e + rss_n

    n_total = int(2 * n_obs)

    # Separate E and N coefficient vectors plus one effective tau parameter for nonlinear grid models.
    k_total = int(2 * p + (1 if model_name in {"exponential", "logarithmic"} else 0))

    rss_for_ic = max(rss_total, 1.0e-12)

    rms_e = float(math.sqrt(rss_e / n_obs))
    rms_n = float(math.sqrt(rss_n / n_obs))
    rms_h = float(math.sqrt(rss_total / n_total))

    aic = float(n_total * math.log(rss_for_ic / n_total) + 2 * k_total)
    bic = float(n_total * math.log(rss_for_ic / n_total) + k_total * math.log(n_total))

    out = {
        "ok": True,
        "model_name": model_name,
        "tau_days": float(tau_days) if tau_days is not None else math.nan,
        "n_points_per_component": n_obs,
        "n_total_observations": n_total,
        "n_parameters_total": k_total,
        "rss_E_mm2": rss_e,
        "rss_N_mm2": rss_n,
        "rss_horizontal_total_mm2": rss_total,
        "rms_E_mm": rms_e,
        "rms_N_mm": rms_n,
        "rms_horizontal_mm": rms_h,
        "aic_joint": aic,
        "bic_joint": bic,
        "reason": "",
    }

    for name, value in zip(names, beta_e):
        out[f"E_{name}"] = float(value)

    for name, value in zip(names, beta_n):
        out[f"N_{name}"] = float(value)

    for prefix in ["E", "N"]:
        for name in [
            "intercept_mm",
            "linear_mm_per_day",
            "quadratic_mm_per_day2",
            "amplitude_mm",
        ]:
            key = f"{prefix}_{name}"
            if key not in out:
                out[key] = math.nan

    return out


def _fit_joint_horizontal_transient_models_for_window(
    df: pd.DataFrame,
    x_series: pd.Series,
    x0: float,
    x1: float,
    gaussian_width_days: float,
) -> pd.DataFrame:
    if "E_m" not in df.columns or "N_m" not in df.columns:
        return pd.DataFrame()

    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return pd.DataFrame()

    e_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column="E_m",
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    n_smooth = _gaussian_smoothed_enu_series_for_plot(
        df=df,
        column="N_m",
        x_series=x_series,
        gaussian_width_days=gaussian_width_days,
    )

    if e_smooth is None:
        e_smooth = pd.to_numeric(df["E_m"], errors="coerce")

    if n_smooth is None:
        n_smooth = pd.to_numeric(df["N_m"], errors="coerce")

    x = pd.to_numeric(x_series, errors="coerce")
    e = pd.to_numeric(e_smooth, errors="coerce") * 1000.0
    n = pd.to_numeric(n_smooth, errors="coerce") * 1000.0

    mask = (
        (x >= x0)
        & (x <= x1)
        & x.notna()
        & e.notna()
        & n.notna()
    )

    xw = x.loc[mask].to_numpy(dtype=float)
    ew = e.loc[mask].to_numpy(dtype=float)
    nw = n.loc[mask].to_numpy(dtype=float)

    if len(xw) < 10:
        return pd.DataFrame()

    t_days = (xw - float(x0)) * 365.25
    duration_days = float((float(x1) - float(x0)) * 365.25)

    rows = []

    for model_name in ["linear", "quadratic"]:
        fit = _fit_joint_horizontal_transient_model(
            t_days=t_days,
            e_mm=ew,
            n_mm=nw,
            model_name=model_name,
            tau_days=None,
        )

        if fit.get("ok", False):
            rows.append(fit)

    for model_name in []:  # exponential/logarithmic disabled; quadratic-only transient model
        best = None

        for tau in _tau_grid_for_transient_fit(duration_days):
            fit = _fit_joint_horizontal_transient_model(
                t_days=t_days,
                e_mm=ew,
                n_mm=nw,
                model_name=model_name,
                tau_days=tau,
            )

            if not fit.get("ok", False):
                continue

            if best is None or float(fit["bic_joint"]) < float(best["bic_joint"]):
                best = fit

        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["duration_days"] = duration_days

    best_bic = float(out["bic_joint"].min())

    out["delta_bic_from_best"] = out["bic_joint"] - best_bic
    out["best_by_bic"] = out["bic_joint"] == best_bic

    return out.sort_values(["best_by_bic", "bic_joint"], ascending=[False, True]).reset_index(drop=True)


def _compute_joint_horizontal_transient_model_fits(
    df: pd.DataFrame,
    transient_windows: pd.DataFrame,
    report_analysis_config: dict | None = None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if transient_windows is None or len(transient_windows) == 0:
        return pd.DataFrame()

    required = {
        "window_type",
        "start_decimal_year",
        "end_decimal_year",
    }

    if not required.issubset(set(transient_windows.columns)):
        return pd.DataFrame()

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    plot_cfg = analysis_cfg.get("plots", {})
    gaussian_width_days = float(plot_cfg.get("gaussian_width_days", 28.0))

    x_series, _ = _decimal_year_series_for_composite_plot(df)

    rows = []

    work = transient_windows.copy()
    work = work[work["window_type"].astype(str).isin(["pre_event_transient", "post_event_transient"])].copy()

    if len(work) == 0:
        return pd.DataFrame()

    for _, window in work.iterrows():
        try:
            x0 = float(window.get("start_decimal_year", math.nan))
            x1 = float(window.get("end_decimal_year", math.nan))
        except Exception:
            continue

        fits = _fit_joint_horizontal_transient_models_for_window(
            df=df,
            x_series=x_series,
            x0=x0,
            x1=x1,
            gaussian_width_days=gaussian_width_days,
        )

        if fits is None or len(fits) == 0:
            continue

        for _, fit in fits.iterrows():
            row = fit.to_dict()

            row["meta_cluster_id"] = window.get("meta_cluster_id", "")
            row["window_type"] = window.get("window_type", "")
            row["start_decimal_year"] = x0
            row["end_decimal_year"] = x1
            row["start_date"] = window.get("start_date", "")
            row["end_date"] = window.get("end_date", "")
            row["gaussian_width_days"] = gaussian_width_days

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    order_cols = [
        "meta_cluster_id",
        "window_type",
        "best_by_bic",
        "bic_joint",
    ]

    existing_order = [col for col in order_cols if col in out.columns]

    out = _add_joint_horizontal_interpretation_flags(out)

    if existing_order:
        out = out.sort_values(existing_order, ascending=[True, True, False, True]).reset_index(drop=True)

    return out



def _bic_preference_strength(delta_bic: float) -> str:
    try:
        value = float(delta_bic)
    except Exception:
        return "undefined"

    if not math.isfinite(value):
        return "undefined"

    if value >= 10.0:
        return "strong"

    if value >= 6.0:
        return "moderate"

    if value >= 2.0:
        return "weak"

    return "negligible"


def _joint_horizontal_interpretation_class(window_type: str, model_name: str) -> str:
    window_type = str(window_type)
    model_name = str(model_name)

    if window_type == "pre_event_transient":
        if model_name == "quadratic":
            return "pre_event_horizontal_acceleration"
        if model_name in {"exponential", "logarithmic"}:
            return "pre_event_horizontal_nonlinear_transient"
        if model_name == "linear":
            return "pre_event_linear_background"
        return "pre_event_unclassified"

    if window_type == "post_event_transient":
        if model_name in {"exponential", "logarithmic"}:
            return "post_event_horizontal_relaxation"
        if model_name == "quadratic":
            return "post_event_horizontal_curvature"
        if model_name == "linear":
            return "post_event_linear_background"
        return "post_event_unclassified"

    return "unclassified"


def _tau_resolution_flag(
    model_name: str,
    tau_days,
    gaussian_width_days,
    duration_days,
) -> str:
    model_name = str(model_name)

    if model_name not in {"exponential", "logarithmic"}:
        return "not_applicable"

    try:
        tau = float(tau_days)
    except Exception:
        return "undefined"

    try:
        gw = float(gaussian_width_days)
    except Exception:
        gw = math.nan

    try:
        duration = float(duration_days)
    except Exception:
        duration = math.nan

    if not math.isfinite(tau) or tau <= 0:
        return "invalid_tau"

    if math.isfinite(gw) and tau < gw:
        return "tau_shorter_than_gaussian_width"

    if math.isfinite(duration) and tau > duration:
        return "tau_longer_than_window"

    return "tau_resolved"


def _add_joint_horizontal_interpretation_flags(joint_fits: pd.DataFrame) -> pd.DataFrame:
    if joint_fits is None or len(joint_fits) == 0:
        return pd.DataFrame()

    out = joint_fits.copy()

    for col in [
        "bic_preference_margin_to_second",
        "model_preference_strength",
        "horizontal_interpretation_class",
        "physical_interpretation_flag",
        "tau_resolution_flag",
    ]:
        if col not in out.columns:
            out[col] = ""

    required = {"meta_cluster_id", "window_type", "model_name", "bic_joint"}

    if not required.issubset(set(out.columns)):
        return out

    for _, group in out.groupby(["meta_cluster_id", "window_type"], dropna=False):
        if len(group) == 0:
            continue

        sorted_group = group.sort_values("bic_joint")

        best_idx = sorted_group.index[0]
        best_bic = float(sorted_group.iloc[0]["bic_joint"])

        if len(sorted_group) >= 2:
            second_bic = float(sorted_group.iloc[1]["bic_joint"])
            margin = second_bic - best_bic
        else:
            margin = math.nan

        strength = _bic_preference_strength(margin)

        for idx, row in group.iterrows():
            is_best = bool(row.get("best_by_bic", False))

            model_name = str(row.get("model_name", ""))
            window_type = str(row.get("window_type", ""))

            interpretation_class = _joint_horizontal_interpretation_class(
                window_type=window_type,
                model_name=model_name,
            )

            tau_flag = _tau_resolution_flag(
                model_name=model_name,
                tau_days=row.get("tau_days", math.nan),
                gaussian_width_days=row.get("gaussian_width_days", math.nan),
                duration_days=row.get("duration_days", math.nan),
            )

            out.loc[idx, "bic_preference_margin_to_second"] = margin if is_best else math.nan
            out.loc[idx, "model_preference_strength"] = strength if is_best else "not_selected"
            out.loc[idx, "horizontal_interpretation_class"] = interpretation_class

            if not is_best:
                out.loc[idx, "physical_interpretation_flag"] = "not_selected"
            elif strength in {"strong", "moderate"} and tau_flag not in {"tau_shorter_than_gaussian_width", "invalid_tau"}:
                out.loc[idx, "physical_interpretation_flag"] = "primary_horizontal_interpretation"
            elif strength in {"strong", "moderate"} and tau_flag == "tau_shorter_than_gaussian_width":
                out.loc[idx, "physical_interpretation_flag"] = "primary_model_but_tau_weakly_resolved"
            elif strength == "weak":
                out.loc[idx, "physical_interpretation_flag"] = "weak_model_preference"
            else:
                out.loc[idx, "physical_interpretation_flag"] = "diagnostic_only"

            out.loc[idx, "tau_resolution_flag"] = tau_flag

    return out

def _joint_horizontal_transient_model_fits_html(joint_fits: pd.DataFrame) -> str:
    text = ["<h3>Joint horizontal transient model fit diagnostics</h3>"]

    if joint_fits is None or len(joint_fits) == 0:
        text.append("<p>No joint horizontal transient model fits were generated.</p>")
        return chr(10).join(text)

    text.append(
        "<p>Joint horizontal transient fits use E and N together over the same automatically inferred transient windows. "
        "Only two models are evaluated: a linear reference model and a quadratic transient diagnostic model. "
        "Model comparison is based on joint horizontal BIC using RSS<sub>total</sub> = RSS<sub>E</sub> + RSS<sub>N</sub>. "
        "This table is the primary source for horizontal transient model selection. "
        "No exponential/logarithmic time-scale parameter is estimated in the current method.</p>"
    )

    best = joint_fits[joint_fits.get("best_by_bic", False) == True].copy()

    if len(best) > 0:
        text.append("<h4>Primary horizontal transient model selection</h4>")

        best_cols = [
            "meta_cluster_id",
            "window_type",
            "model_name",
            "horizontal_interpretation_class",
            "physical_interpretation_flag",
            "model_preference_strength",
            "duration_days",
            "rms_horizontal_mm",
            "bic_joint",
            "bic_preference_margin_to_second",
            "E_linear_mm_per_day",
            "N_linear_mm_per_day",
            "E_quadratic_mm_per_day2",
            "N_quadratic_mm_per_day2",
        ]

        best_existing = [col for col in best_cols if col in best.columns]

        best_headers = [
            "Meta ID",
            "Window type",
            "Best model",
            "Interpretation class",
            "Physical interpretation flag",
            "Preference strength",
            "Duration (days)",
            "RMS horizontal (mm)",
            "BIC joint",
            "BIC margin to second",
            "E linear (mm/day)",
            "N linear (mm/day)",
            "E quadratic (mm/day²)",
            "N quadratic (mm/day²)",
        ][:len(best_existing)]

        best_rows = []

        for _, item in best.iterrows():
            row = []

            for col in best_existing:
                value = item.get(col, "")

                if col in {
                    "duration_days",
                    "rms_horizontal_mm",
                    "bic_joint",
                    "delta_bic_from_best",
                    "E_linear_mm_per_day",
                    "N_linear_mm_per_day",
                    "E_quadratic_mm_per_day2",
                    "N_quadratic_mm_per_day2",
                    "E_amplitude_mm",
                    "N_amplitude_mm",
                    "tau_days",
                }:
                    value = _fmt(value, digits=4)

                row.append(_html_escape(value))

            best_rows.append(row)

        text.append(_html_table(best_headers, best_rows))

    cols = [
        "meta_cluster_id",
        "window_type",
        "model_name",
        "best_by_bic",
        "start_decimal_year",
        "end_decimal_year",
        "start_date",
        "end_date",
        "duration_days",
        "n_points_per_component",
        "n_parameters_total",
        "rms_E_mm",
        "rms_N_mm",
        "rms_horizontal_mm",
        "aic_joint",
        "bic_joint",
        "delta_bic_from_best",
        "E_linear_mm_per_day",
        "N_linear_mm_per_day",
        "E_quadratic_mm_per_day2",
        "N_quadratic_mm_per_day2",
        "gaussian_width_days",
    ]

    existing = [col for col in cols if col in joint_fits.columns]

    headers = [
        "Meta ID",
        "Window type",
        "Model",
        "Best by BIC",
        "Start decimal year",
        "End decimal year",
        "Start date",
        "End date",
        "Duration (days)",
        "N/component",
        "k total",
        "RMS E (mm)",
        "RMS N (mm)",
        "RMS horizontal (mm)",
        "AIC joint",
        "BIC joint",
        "ΔBIC",
        "E linear (mm/day)",
        "N linear (mm/day)",
        "E quadratic (mm/day²)",
        "N quadratic (mm/day²)",
        "Gaussian width (days)",
    ][:len(existing)]

    rows = []

    for _, item in joint_fits.iterrows():
        row = []

        for col in existing:
            value = item.get(col, "")

            if col in {
                "start_decimal_year",
                "end_decimal_year",
            }:
                value = _fmt(value, digits=6)
            elif col in {
                "duration_days",
                "rms_E_mm",
                "rms_N_mm",
                "rms_horizontal_mm",
                "aic_joint",
                "bic_joint",
                "delta_bic_from_best",
                "E_linear_mm_per_day",
                "N_linear_mm_per_day",
                "E_quadratic_mm_per_day2",
                "N_quadratic_mm_per_day2",
                "E_amplitude_mm",
                "N_amplitude_mm",
                "tau_days",
                "gaussian_width_days",
            }:
                value = _fmt(value, digits=4)

            row.append(_html_escape(value))

        rows.append(row)

    text.append(_html_table(headers, rows))

    return chr(10).join(text)

def _transient_model_fits_html(transient_model_fits: pd.DataFrame) -> str:
    text = ["<h3>Component-wise transient model fit diagnostics (secondary)</h3>"]

    if transient_model_fits is None or len(transient_model_fits) == 0:
        text.append("<p>No transient model fits were generated.</p>")
        return chr(10).join(text)

    text.append(
        "<p>These component-wise transient fits are secondary diagnostics. "
        "For E/N horizontal transient model selection, use the joint horizontal transient model table above. "
        "Only a linear reference model and a quadratic transient diagnostic model are evaluated. "
        "The component-wise E/N fits are retained only to inspect residual behaviour by component. "
        "U fits are diagnostic-only. No fitted transient model is drawn in the ENU composite plot at this stage.</p>"
    )

    cols = [
        "meta_cluster_id",
        "window_type",
        "component",
        "model_name",
        "best_by_bic",
        "start_decimal_year",
        "end_decimal_year",
        "start_date",
        "end_date",
        "duration_days",
        "n_points",
        "n_parameters",
        "rms_mm",
        "aic",
        "bic",
        "delta_bic_from_best",
        "intercept_mm",
        "linear_mm_per_day",
        "quadratic_mm_per_day2",
        "gaussian_width_days",
    ]

    existing = [col for col in cols if col in transient_model_fits.columns]

    headers = [
        "Meta ID",
        "Window type",
        "Component",
        "Model",
        "Best by BIC",
        "Start decimal year",
        "End decimal year",
        "Start date",
        "End date",
        "Duration (days)",
        "N",
        "k",
        "RMS (mm)",
        "AIC",
        "BIC",
        "ΔBIC",
        "Intercept (mm)",
        "Linear (mm/day)",
        "Quadratic (mm/day²)",
        "Gaussian width (days)",
    ][:len(existing)]

    rows = []

    for _, item in transient_model_fits.iterrows():
        row = []

        for col in existing:
            value = item.get(col, "")

            if col in {
                "start_decimal_year",
                "end_decimal_year",
            }:
                value = _fmt(value, digits=6)
            elif col in {
                "duration_days",
                "rms_mm",
                "aic",
                "bic",
                "delta_bic_from_best",
                "intercept_mm",
                "linear_mm_per_day",
                "quadratic_mm_per_day2",
                "gaussian_width_days",
            }:
                value = _fmt(value, digits=4)

            row.append(_html_escape(value))

        rows.append(row)

    text.append(_html_table(headers, rows))

    return chr(10).join(text)

def _transient_windows_html(transient_windows: pd.DataFrame) -> str:
    text = ["<h3>Automatic pre/event/post transient windows</h3>"]

    if transient_windows is None or len(transient_windows) == 0:
        text.append("<p>No automatic transient windows were generated.</p>")
        return chr(10).join(text)

    text.append(
        "<p>Event/incidence intervals are not treated as stable velocity intervals. "
        "They are reported as net displacement and deformation rate during incidence. "
        "Pre- and post-event transient windows are inferred automatically from shift-related report-grade horizontal velocity-change clusters "
        "and are excluded from stable rolling/sliding velocity interpretation.</p>"
    )

    cols = [
        "meta_cluster_id",
        "window_type",
        "source",
        "start_decimal_year",
        "end_decimal_year",
        "start_date",
        "end_date",
        "duration_days",
        "E_m_net_mm",
        "N_m_net_mm",
        "U_m_net_mm",
        "H_net_mm",
        "E_m_range_mm",
        "N_m_range_mm",
        "U_m_range_mm",
        "E_m_rate_mm_per_day",
        "N_m_rate_mm_per_day",
        "U_m_rate_mm_per_day",
        "H_rate_mm_per_day",
        "interpretation",
    ]

    existing = [col for col in cols if col in transient_windows.columns]

    headers = [
        "Meta ID",
        "Window type",
        "Source",
        "Start decimal year",
        "End decimal year",
        "Start date",
        "End date",
        "Duration (days)",
        "E net (mm)",
        "N net (mm)",
        "U net (mm)",
        "H net (mm)",
        "E range (mm)",
        "N range (mm)",
        "U range (mm)",
        "E incidence rate (mm/day)",
        "N incidence rate (mm/day)",
        "U incidence rate (mm/day)",
        "H incidence rate (mm/day)",
        "Interpretation",
    ][:len(existing)]

    rows = []

    for _, item in transient_windows.iterrows():
        row = []

        for col in existing:
            value = item.get(col, "")

            if col in {
                "start_decimal_year",
                "end_decimal_year",
            }:
                value = _fmt(value, digits=6)
            elif col in {
                "duration_days",
                "E_m_net_mm",
                "N_m_net_mm",
                "U_m_net_mm",
                "H_net_mm",
                "E_m_range_mm",
                "N_m_range_mm",
                "U_m_range_mm",
                "E_m_rate_mm_per_day",
                "N_m_rate_mm_per_day",
                "U_m_rate_mm_per_day",
                "H_rate_mm_per_day",
            }:
                value = _fmt(value, digits=3)

            row.append(_html_escape(value))

        rows.append(row)

    text.append(_html_table(headers, rows))

    return chr(10).join(text)

def _compute_velocity_change_diagnostics(
    df: pd.DataFrame,
    meta_clusters: pd.DataFrame,
    report_analysis_config: dict | None = None,
) -> dict:
    empty = {
        "rolling": pd.DataFrame(),
        "changes": pd.DataFrame(),
        "clusters": pd.DataFrame(),
        "classified": pd.DataFrame(),
        "enabled": False,
        "error": "",
    }

    if df is None or len(df) == 0:
        return empty

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)
    vc_cfg = analysis_cfg.get("rolling_velocity_diagnostics", {})

    if not bool(vc_cfg.get("enabled", True)):
        return empty

    try:
        import timeseries_velocity_detection as tvd
    except Exception as exc:
        out = empty.copy()
        out["error"] = f"timeseries_velocity_detection import failed: {exc}"
        return out

    required_names = [
        "RollingVelocityDiagnosticConfig",
        "compute_rolling_velocity_diagnostics",
        "cluster_persistent_velocity_changes_v1",
        "classify_velocity_change_clusters_against_meta_clusters_v1",
    ]

    for name in required_names:
        if not hasattr(tvd, name):
            out = empty.copy()
            out["error"] = f"timeseries_velocity_detection missing required function/class: {name}"
            return out

    try:
        cfg = tvd.RollingVelocityDiagnosticConfig(
            columns=("E_m", "N_m", "U_m"),
            window_days=float(vc_cfg.get("window_days", 182.0)),
            step_days=float(vc_cfg.get("step_days", 7.0)),
            comparison_lag_days=float(vc_cfg.get("comparison_lag_days", 91.0)),
            min_points_per_window=int(vc_cfg.get("min_points_per_window", 20)),
            apply_gaussian_smoothing=bool(vc_cfg.get("apply_gaussian_smoothing", True)),
            gaussian_width_days=float(vc_cfg.get("gaussian_width_days", 28.0)),
            significance_z_threshold=float(vc_cfg.get("significance_z_threshold", 4.0)),
            min_abs_delta_velocity_mm_per_year=float(vc_cfg.get("min_abs_delta_velocity_mm_per_year", 1.0)),
            minimum_persistence_fraction_of_window=float(vc_cfg.get("minimum_persistence_fraction_of_window", 0.5)),
            horizontal_coherence_required=bool(vc_cfg.get("horizontal_coherence_required", True)),
            meta_association_window_days=vc_cfg.get("meta_association_window_days", None),
            include_horizontal_vector=bool(vc_cfg.get("include_horizontal_vector", True)),
        )

        rolling, changes = tvd.compute_rolling_velocity_diagnostics(df, cfg)
        clusters = tvd.cluster_persistent_velocity_changes_v1(changes, cfg)
        classified = tvd.classify_velocity_change_clusters_against_meta_clusters_v1(
            velocity_clusters=clusters,
            meta_clusters=meta_clusters,
            cfg=cfg,
        )

        transient_windows = _compute_transient_windows_from_velocity_change_clusters(
            df=df,
            meta_clusters=meta_clusters,
            velocity_change_diagnostics={"classified": classified},
            report_analysis_config=analysis_cfg,
        )

        transient_model_fits = _compute_transient_model_fits(
            df=df,
            transient_windows=transient_windows,
            report_analysis_config=analysis_cfg,
        )

        joint_horizontal_transient_model_fits = _compute_joint_horizontal_transient_model_fits(
            df=df,
            transient_windows=transient_windows,
            report_analysis_config=analysis_cfg,
        )

        return {
            "rolling": rolling,
            "changes": changes,
            "clusters": clusters,
            "classified": classified,
            "transient_windows": transient_windows,
            "transient_model_fits": transient_model_fits,
            "joint_horizontal_transient_model_fits": joint_horizontal_transient_model_fits,
            "enabled": True,
            "error": "",
            "config": {
                "window_days": float(cfg.window_days),
                "step_days": float(cfg.step_days),
                "comparison_lag_days": float(cfg.comparison_lag_days),
                "significance_z_threshold": float(cfg.significance_z_threshold),
                "min_abs_delta_velocity_mm_per_year": float(cfg.min_abs_delta_velocity_mm_per_year),
                "minimum_persistence_fraction_of_window": float(cfg.minimum_persistence_fraction_of_window),
                "minimum_persistence_days": float(cfg.minimum_persistence_fraction_of_window) * float(cfg.window_days),
                "gaussian_width_days": float(cfg.gaussian_width_days),
                "horizontal_coherence_required": bool(cfg.horizontal_coherence_required),
            },
        }

    except Exception as exc:
        out = empty.copy()
        out["enabled"] = True
        out["error"] = str(exc)
        return out


def _velocity_change_table_html(
    df: pd.DataFrame,
    title: str,
    max_rows: int = 50,
) -> str:
    text = [f"<h3>{_html_escape(title)}</h3>"]

    if df is None or len(df) == 0:
        text.append("<p>No rows.</p>")
        return chr(10).join(text)

    cols = [
        "velocity_change_cluster_id",
        "component",
        "shift_context_class",
        "report_grade",
        "shift_related_velocity_change",
        "nearest_meta_relation",
        "nearest_meta_gap_days",
        "nearest_meta_overlap_days",
        "cluster_start_decimal_year",
        "cluster_end_decimal_year",
        "cluster_duration_days",
        "n_consecutive_centers",
        "minimum_required_centers",
        "representative_center_decimal_year",
        "representative_delta_velocity_mm_per_year",
        "representative_sigma_delta_velocity_mm_per_year",
        "max_abs_delta_velocity_mm_per_year",
        "max_velocity_change_z",
        "horizontal_support_components",
    ]

    existing = [col for col in cols if col in df.columns]

    headers = [
        "Cluster",
        "Component",
        "Context class",
        "Report-grade",
        "Shift-related",
        "Meta relation",
        "Meta gap (days)",
        "Meta overlap (days)",
        "Start decimal year",
        "End decimal year",
        "Duration (days)",
        "Consecutive centers",
        "Required centers",
        "Representative center",
        "Representative Δv (mm/yr)",
        "σΔv (mm/yr)",
        "Max |Δv| (mm/yr)",
        "Max Z",
        "Horizontal support",
    ][:len(existing)]

    rows = []

    for _, item in df.head(max_rows).iterrows():
        row = []

        for col in existing:
            value = item.get(col, "")

            if col in {
                "cluster_start_decimal_year",
                "cluster_end_decimal_year",
                "representative_center_decimal_year",
            }:
                value = _fmt(value, digits=4)
            elif col in {
                "nearest_meta_gap_days",
                "nearest_meta_overlap_days",
                "cluster_duration_days",
                "representative_delta_velocity_mm_per_year",
                "representative_sigma_delta_velocity_mm_per_year",
                "max_abs_delta_velocity_mm_per_year",
                "max_velocity_change_z",
            }:
                value = _fmt(value, digits=3)

            row.append(_html_escape(value))

        rows.append(row)

    text.append(_html_table(headers, rows))

    if len(df) > max_rows:
        text.append(f"<p>Table truncated to first {max_rows} rows out of {len(df)}.</p>")

    return chr(10).join(text)


def _velocity_change_diagnostics_html(result: dict) -> str:
    if result is None:
        return "<p>No velocity-change diagnostic result.</p>"

    if not result.get("enabled", False):
        return "<p>Rolling velocity-change diagnostics are disabled.</p>"

    if result.get("error"):
        return f"<p>Velocity-change diagnostics failed: {_html_escape(result.get('error'))}</p>"

    rolling = result.get("rolling", pd.DataFrame())
    changes = result.get("changes", pd.DataFrame())
    classified = result.get("classified", pd.DataFrame())
    cfg = result.get("config", {})

    text = []

    text.append(
        "<p>This section reports rolling velocity-change diagnostics. "
        "The diagnostic first estimates local velocities in moving windows, compares each window against a previous reference window, "
        "keeps only persistent changes, and then classifies persistent velocity-change clusters relative to the detected displacement meta-clusters. "
        "U-only changes are retained as diagnostic-only and are not treated as report-grade horizontal velocity changes.</p>"
    )

    parameter_rows = [
        ["rolling_window_days", _fmt(cfg.get("window_days", math.nan), digits=3)],
        ["rolling_step_days", _fmt(cfg.get("step_days", math.nan), digits=3)],
        ["comparison_lag_days", _fmt(cfg.get("comparison_lag_days", math.nan), digits=3)],
        ["Z_threshold", _fmt(cfg.get("significance_z_threshold", math.nan), digits=3)],
        ["min_abs_delta_velocity_mm_per_year", _fmt(cfg.get("min_abs_delta_velocity_mm_per_year", math.nan), digits=3)],
        ["minimum_persistence_fraction_of_window", _fmt(cfg.get("minimum_persistence_fraction_of_window", math.nan), digits=3)],
        ["minimum_persistence_days", _fmt(cfg.get("minimum_persistence_days", math.nan), digits=3)],
        ["gaussian_width_days", _fmt(cfg.get("gaussian_width_days", math.nan), digits=3)],
        ["horizontal_coherence_required", _html_escape(cfg.get("horizontal_coherence_required", ""))],
        ["rolling_velocity_rows", _html_escape(len(rolling))],
        ["velocity_change_rows", _html_escape(len(changes))],
        ["persistent_velocity_change_clusters", _html_escape(len(classified))],
    ]

    if len(classified) > 0 and "report_grade" in classified.columns:
        parameter_rows.append(["report_grade_clusters", _html_escape(int(classified["report_grade"].sum()))])

    if len(classified) > 0 and "shift_related_velocity_change" in classified.columns:
        parameter_rows.append(["shift_related_clusters", _html_escape(int(classified["shift_related_velocity_change"].sum()))])

    text.append("<h3>Rolling velocity-change diagnostic parameters and counts</h3>")
    text.append(_html_table(["Item", "Value"], parameter_rows))

    transient_windows = result.get("transient_windows", pd.DataFrame())
    text.append(_transient_windows_html(transient_windows))

    joint_horizontal_transient_model_fits = result.get("joint_horizontal_transient_model_fits", pd.DataFrame())
    text.append(_joint_horizontal_transient_model_fits_html(joint_horizontal_transient_model_fits))

    transient_model_fits = result.get("transient_model_fits", pd.DataFrame())
    text.append(_transient_model_fits_html(transient_model_fits))

    if classified is None or len(classified) == 0:
        text.append("<p>No persistent velocity-change clusters passed the current criteria.</p>")
        return chr(10).join(text)

    shift_related_report_grade = classified[
        (classified.get("report_grade", False) == True)
        & (classified.get("shift_related_velocity_change", False) == True)
        & (classified.get("component", "") == "H_magnitude")
    ].copy()

    background_report_grade = classified[
        (classified.get("report_grade", False) == True)
        & (classified.get("shift_related_velocity_change", False) == False)
        & (classified.get("component", "") == "H_magnitude")
    ].copy()

    vertical_diag = classified[
        classified.get("component", "").astype(str) == "U_m"
    ].copy()

    text.append(_velocity_change_table_html(
        shift_related_report_grade,
        "Shift-related report-grade horizontal velocity changes",
    ))

    text.append(_velocity_change_table_html(
        background_report_grade,
        "Background report-grade horizontal velocity-change diagnostics",
    ))

    text.append(_velocity_change_table_html(
        vertical_diag,
        "Vertical diagnostic-only velocity changes",
    ))

    text.append(_velocity_change_table_html(
        classified,
        "All persistent velocity-change clusters",
        max_rows=80,
    ))

    return chr(10).join(text)

def build_timeseries_report(
    timeseries_path: str | Path,
    report_path: str | Path | None = None,
    metadata: dict | None = None,
    plot_columns=None,
    failed_datasets: list[dict] | None = None,
    report_analysis_config: dict | None = None,
) -> dict:
    ts_path = Path(timeseries_path).expanduser().resolve()

    if not ts_path.exists():
        raise FileNotFoundError(f"timeseries.out not found: {ts_path}")

    if report_path is None:
        report_path = ts_path.with_name("timeseries.report")

    rpt_path = Path(report_path).expanduser().resolve()
    rpt_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ts_path, sep="\t")
    failed_datasets = failed_datasets or []

    if len(df) == 0:
        raise ValueError(f"timeseries.out has no data rows: {ts_path}")

    conv_cfg = _load_default_convergence_config()
    meta = _resolve_processing_metadata(ts_path, df, metadata)

    final_ref_hours = conv_cfg.get("internal_reference_last_hours", "")
    min_window_fraction = conv_cfg.get("min_window_fraction", "")

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    threshold_rows = []
    threshold_columns = [
        "threshold_E_m",
        "threshold_N_m",
        "threshold_U_m",
        "threshold_rolling_std_E_m",
        "threshold_rolling_std_N_m",
        "threshold_rolling_std_U_m",
        "threshold_slope_E_mm_per_hour",
        "threshold_slope_N_mm_per_hour",
        "threshold_slope_U_mm_per_hour",
        "persistence_window_sec",
        "rolling_window_sec",
    ]

    for col in threshold_columns:
        if col in df.columns:
            threshold_rows.append([_html_escape(col), _html_escape(_fmt(_value_from_row(df, col)))])
        elif col in conv_cfg:
            threshold_rows.append([_html_escape(col), _html_escape(_fmt(conv_cfg[col]))])

    if min_window_fraction != "":
        threshold_rows.append(["min_window_fraction", _html_escape(_fmt(min_window_fraction))])

    qc_rows = [[_html_escape(item.split(":", 1)[0]), _html_escape(item)] for item in _qc_flag_definitions()]

    unique_flags = []
    if "qc_flags" in df.columns:
        for value in df["qc_flags"].dropna().astype(str):
            for flag in value.split(";"):
                flag = flag.strip()
                if flag and flag not in unique_flags:
                    unique_flags.append(flag)

    if unique_flags:
        flags_html = "<ul>" + "".join(f"<li>{_html_escape(flag)}</li>" for flag in unique_flags) + "</ul>"
    else:
        flags_html = "<p>none</p>"

    metadata_rows = [
        ["PPP provider", _html_escape(meta["provider"])],
        ["PPP series", _html_escape(meta["series"])],
        ["PPP project", _html_escape(meta["project"])],
        ["Sampling interval", _html_escape(str(meta["sampling_interval_sec"])) + " s"],
        ["RINEX RAW root", _html_escape(meta["raw_root"])],
        ["GINAN process directory", _html_escape(meta["ginan_process_dir"])],
        ["PEA executable", _html_escape(meta["pea_executable"])],
        ["Template YAML path", _html_escape(meta["template_yaml_path"])],
    ]

    source_rows = []
    for key, value in meta["sources"].items():
        if value:
            source_rows.append([_html_escape(key), _html_escape(value)])

    analysis_cfg = _resolve_report_analysis_config(report_analysis_config)

    non_successful_html = _non_successful_datasets_html(failed_datasets)
    shift_clusters = _compute_report_shift_clusters(df, analysis_cfg)
    shift_clusters_html = _shift_clusters_html(shift_clusters)
    meta_clusters = _compute_report_meta_clusters(shift_clusters, analysis_cfg)
    meta_clusters_html = _meta_clusters_html(meta_clusters)
    meta_cluster_velocity_windows = _compute_meta_cluster_velocity_windows(df, meta_clusters, analysis_cfg)
    meta_cluster_velocity_windows_html = _meta_cluster_velocity_windows_html(meta_cluster_velocity_windows)
    velocity_change_diagnostics = _compute_velocity_change_diagnostics(df, meta_clusters, analysis_cfg)
    velocity_change_diagnostics_html = _velocity_change_diagnostics_html(velocity_change_diagnostics)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PPP Batch Orchestrator - Timeseries Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 28px;
    line-height: 1.42;
    color: #222;
}}
h1 {{
    font-size: 24px;
    border-bottom: 2px solid #444;
    padding-bottom: 8px;
}}
h2 {{
    font-size: 19px;
    margin-top: 28px;
    border-bottom: 1px solid #aaa;
    padding-bottom: 4px;
}}
h3 {{
    font-size: 16px;
    margin-top: 18px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 18px 0;
    font-size: 13px;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 5px 7px;
    vertical-align: top;
}}
th {{
    background: #f0f0f0;
    text-align: left;
}}
code {{
    background: #f4f4f4;
    padding: 1px 4px;
}}
pre {{
    background: #f6f6f6;
    padding: 10px;
    overflow-x: auto;
}}
img.plot {{
    max-width: 100%;
    border: 1px solid #ccc;
    margin: 6px 0 18px 0;
}}
.small {{
    font-size: 12px;
    color: #555;
}}
</style>
</head>
<body>

<h1>PPP Batch Orchestrator - Timeseries Report</h1>

<table>
<tr><th>Generated UTC</th><td>{_html_escape(generated_utc)}</td></tr>
<tr><th>Input timeseries</th><td>{_html_escape(ts_path)}</td></tr>
<tr><th>Output report</th><td>{_html_escape(rpt_path)}</td></tr>
</table>

<h2>1. Processing strategy</h2>
<p>The processing strategy is daily/per-file single-station PPP. Each raw RINEX dataset is processed as an independent Ginan/PEA PPP run. The output of each run is reduced to one daily/per-file position solution.</p>

<pre>raw daily/per-file RINEX
-&gt; optional GFZRNX resampling
-&gt; effective time-window determination
-&gt; PPP product resolution/download
-&gt; deterministic YAML generation
-&gt; Ginan/PEA execution
-&gt; POS-based convergence analysis
-&gt; daily/per-file solution written to timeseries.out</pre>

<h2>2. PPP setup and paths</h2>
{_html_table(["Item", "Value"], metadata_rows)}

<h3>Metadata sources</h3>
{_html_table(["Item", "Source"], source_rows)}

<h2>3. Convergence detection method</h2>
<p>Convergence is detected using the unsmoothed POS time series. The internal convergence reference is the robust median position of the final {_html_escape(final_ref_hours)} hours of the unsmoothed POS time series. The algorithm scans forward in time and accepts the first epoch for which the following forward persistence window satisfies absolute residual, standard-deviation, and linear-slope thresholds.</p>

<h3>Thresholds</h3>
{_html_table(["Parameter", "Value"], threshold_rows)}

<h2>4. Smoothed versus unsmoothed POS usage</h2>
<p>Unsmoothed POS is used for convergence detection and for realistic post-convergence QC/scatter statistics. Smoothed POS is used for the final daily/per-file position estimate after the convergence epoch.</p>

<table>
<tr><th>Prefix</th><th>Interpretation</th></tr>
<tr><td><code>solution_conv_*</code></td><td>statistics from smoothed POS after convergence</td></tr>
<tr><td><code>qc_unsmoothed_conv_*</code></td><td>realistic QC/scatter statistics from unsmoothed POS after convergence</td></tr>
<tr><td><code>full_solution_*</code></td><td>statistics from the full smoothed POS time series</td></tr>
<tr><td><code>full_unsmoothed_*</code></td><td>statistics from the full unsmoothed POS time series</td></tr>
<tr><td><code>conv_window_*</code></td><td>diagnostics from the accepted one-hour convergence window</td></tr>
</table>

<h2>5. Daily/per-file position solution definition</h2>
<p>The primary daily/per-file solution is the median XYZ position of smoothed POS epochs retained after the convergence epoch. The secondary solution is the corresponding mean XYZ position over the same post-convergence smoothed interval.</p>

<table>
<tr><th>Solution</th><th>Columns</th></tr>
<tr><td>Primary</td><td><code>X_m, Y_m, Z_m, lon_deg, lat_deg, h_m, E_m, N_m, U_m</code></td></tr>
<tr><td>Secondary</td><td><code>X_mean_m, Y_mean_m, Z_mean_m, lon_mean_deg, lat_mean_deg, h_mean_m, E_mean_m, N_mean_m, U_mean_m</code></td></tr>
</table>

<h2>6. ENU series reference</h2>
<p>The common ENU reference for the series is the primary median solution of the first successful converged daily/per-file PPP solution, unless an external reference is introduced in a later workflow.</p>

<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>series_enu_reference_run_label</td><td>{_html_escape(_fmt(_value_from_row(df, "series_enu_reference_run_label")))}</td></tr>
<tr><td>series_enu_reference_X_m</td><td>{_html_escape(_fmt_m(_value_from_row(df, "series_enu_reference_X_m")))}</td></tr>
<tr><td>series_enu_reference_Y_m</td><td>{_html_escape(_fmt_m(_value_from_row(df, "series_enu_reference_Y_m")))}</td></tr>
<tr><td>series_enu_reference_Z_m</td><td>{_html_escape(_fmt_m(_value_from_row(df, "series_enu_reference_Z_m")))}</td></tr>
</table>

<h2>7. QC flags</h2>
{_html_table(["Flag", "Definition"], qc_rows)}

<h3>QC flags present in this timeseries</h3>
{flags_html}

<h2>8. Column definitions of timeseries.out</h2>
{_compact_column_definition_html(df)}

<h2>9. Summary of daily/per-file solutions</h2>
{_daily_summary_html(df)}
{non_successful_html}

<h2>10. Final GNSS station coordinate solution from daily/per-file PPP solutions</h2>
{_final_station_solution_html(df)}

<h2>11. Automatic shift detection</h2>
{shift_clusters_html}

<h2>12. Automatic meta-clustering of report-grade strict clusters</h2>
{meta_clusters_html}

<h2>13. Meta-cluster velocity windows</h2>
{meta_cluster_velocity_windows_html}

<h2>14. Velocity change diagnostics</h2>
{velocity_change_diagnostics_html}

<h2>15. Plots requested by user</h2>
{_plot_html(df, plot_columns=plot_columns, failed_datasets=failed_datasets, shift_clusters=shift_clusters, meta_cluster_velocity_windows=meta_cluster_velocity_windows, velocity_change_diagnostics=velocity_change_diagnostics, report_analysis_config=analysis_cfg)}

<h2>16. Shift-cluster zoom plots</h2>
{_shift_cluster_zoom_plots_html(df, shift_clusters)}

</body>
</html>
"""

    rpt_path.write_text(html_text, encoding="utf-8")

    return {
        "ok": True,
        "timeseries_path": str(ts_path),
        "report_path": str(rpt_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_report_lines": int(len(html_text.splitlines())),
        "report_format": "html",
        "plot_columns": [column for _, column in _resolve_plot_columns(plot_columns)],
        "n_failed_datasets": int(len(failed_datasets)),
        "n_shift_clusters": int(len(shift_clusters)) if shift_clusters is not None else 0,
        "n_meta_clusters": int(len(meta_clusters)) if meta_clusters is not None else 0,
        "n_meta_cluster_velocity_windows": int(len(meta_cluster_velocity_windows)) if meta_cluster_velocity_windows is not None else 0,
        "n_velocity_change_clusters": int(len(velocity_change_diagnostics.get("classified", pd.DataFrame()))) if velocity_change_diagnostics else 0,
        "n_transient_windows": int(len(velocity_change_diagnostics.get("transient_windows", pd.DataFrame()))) if velocity_change_diagnostics else 0,
        "n_transient_model_fits": int(len(velocity_change_diagnostics.get("transient_model_fits", pd.DataFrame()))) if velocity_change_diagnostics else 0,
        "n_joint_horizontal_transient_model_fits": int(len(velocity_change_diagnostics.get("joint_horizontal_transient_model_fits", pd.DataFrame()))) if velocity_change_diagnostics else 0,
        "n_shift_related_report_grade_velocity_changes": int(
            (
                (velocity_change_diagnostics.get("classified", pd.DataFrame()).get("report_grade", pd.Series(dtype=bool)) == True)
                & (velocity_change_diagnostics.get("classified", pd.DataFrame()).get("shift_related_velocity_change", pd.Series(dtype=bool)) == True)
                & (velocity_change_diagnostics.get("classified", pd.DataFrame()).get("component", pd.Series(dtype=str)).astype(str) == "H_magnitude")
            ).sum()
        ) if velocity_change_diagnostics and len(velocity_change_diagnostics.get("classified", pd.DataFrame())) > 0 else 0,
    }


if __name__ == "__main__":
    default_timeseries = Path.cwd() / "timeseries.out"
    result = build_timeseries_report(default_timeseries)
    print(result)
