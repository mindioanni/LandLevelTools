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


def _plot_html(df: pd.DataFrame, plot_columns=None, failed_datasets: list[dict] | None = None) -> str:
    resolved = _resolve_plot_columns(plot_columns)

    if not resolved:
        return "<p>No plots requested.</p>"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        return f"<p>Plots could not be generated because matplotlib is unavailable: {_html_escape(exc)}</p>"

    failed_datasets = failed_datasets or []
    failed_x = []

    for item in failed_datasets:
        dec = _failed_dataset_decimal_year(item)
        if math.isfinite(dec):
            failed_x.append(dec)

    if "time_mean_all_epochs_utc" in df.columns:
        times = pd.to_datetime(df["time_mean_all_epochs_utc"], errors="coerce", utc=True)
        x = [_timestamp_to_decimal_year(t) for t in times]
        x_label = "decimal year"
    else:
        x = list(range(1, len(df) + 1))
        x_label = "solution index"
        failed_x = []

    html_parts = []
    html_parts.append(
        "<p>The following plots are generated from the daily/per-file primary solutions stored in "
        "<code>timeseries.out</code>. The x-axis is shown as decimal year where time metadata are available.</p>"
    )

    if failed_x:
        html_parts.append(
            "<p>Red dashed vertical lines indicate datasets that did not complete successfully and are therefore "
            "not included in <code>timeseries.out</code>.</p>"
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
        ax.plot(x, y, linestyle="-")
        ax.set_title(f"{column} time series")
        ax.set_xlabel(x_label)
        ax.set_ylabel(column)
        ax.grid(True)

        if x_label == "decimal year":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        for k, fx in enumerate(failed_x):
            ax.axvline(
                fx,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label="non-successful dataset" if k == 0 else None,
            )

        if failed_x:
            ax.legend(loc="best")

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
def build_timeseries_report(
    timeseries_path: str | Path,
    report_path: str | Path | None = None,
    metadata: dict | None = None,
    plot_columns=None,
    failed_datasets: list[dict] | None = None,
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

    non_successful_html = _non_successful_datasets_html(failed_datasets)

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

<h2>11. Plots requested by user</h2>
{_plot_html(df, plot_columns=plot_columns, failed_datasets=failed_datasets)}

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
    }


if __name__ == "__main__":
    default_timeseries = Path.cwd() / "timeseries.out"
    result = build_timeseries_report(default_timeseries)
    print(result)
