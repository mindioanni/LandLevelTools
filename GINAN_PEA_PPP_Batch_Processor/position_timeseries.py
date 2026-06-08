from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np
import pandas as pd


POS_COLUMNS = [
    "epoch_utc",
    "decimal_year",
    "X_m",
    "Y_m",
    "Z_m",
    "Sx_m",
    "Sy_m",
    "Sz_m",
    "Rxy",
    "Rxz",
    "Ryz",
    "lat_deg",
    "lon_deg",
    "h_m",
    "pos_dN_m",
    "pos_dE_m",
    "pos_dU_m",
    "Sn_m",
    "Se_m",
    "Su_m",
    "Rne",
    "Rnu",
    "Reu",
    "soln",
]


DEFAULT_CONVERGENCE_CONFIG = {
    "convergence_use_smoothed_pos": False,
    "solution_use_smoothed_pos": True,

    "internal_reference_last_hours": 4.0,

    "persistence_window_sec": 3600.0,
    "rolling_window_sec": 3600.0,
    "min_window_fraction": 0.80,

    "threshold_E_m": 0.03,
    "threshold_N_m": 0.03,
    "threshold_U_m": 0.05,

    "threshold_rolling_std_E_m": 0.02,
    "threshold_rolling_std_N_m": 0.02,
    "threshold_rolling_std_U_m": 0.02,

    "threshold_slope_E_mm_per_hour": 2.0,
    "threshold_slope_N_mm_per_hour": 2.0,
    "threshold_slope_U_mm_per_hour": 5.0,

    "scan_trace_files": False,
}


def _as_iso_z(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean_time_utc(times: pd.Series) -> pd.Timestamp | None:
    if len(times) == 0:
        return None

    t = pd.to_datetime(times, utc=True)
    ns = t.astype("int64").to_numpy(dtype=np.int64)

    if len(ns) == 0:
        return None

    return pd.to_datetime(int(np.mean(ns)), utc=True)


def _read_pos_reference_info(pos_path: Path) -> dict:
    info = {
        "station_id": "",
        "reference_frame": "",
        "xyz_reference_position": "",
        "neu_reference_position": "",
    }

    with pos_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s.startswith("PBO Station Position Time Series. Reference Frame"):
                info["reference_frame"] = s.split(":")[-1].strip()

            elif s.startswith("4-character ID:"):
                info["station_id"] = s.split(":", 1)[1].strip()

            elif s.startswith("XYZ Reference position"):
                info["xyz_reference_position"] = s.split(":", 1)[1].strip()

            elif s.startswith("NEU Reference position"):
                info["neu_reference_position"] = s.split(":", 1)[1].strip()

            elif s.startswith("*YYYY-MM-DD"):
                break

    return info


def read_pos_file(pos_path: str | Path) -> pd.DataFrame:
    p = Path(pos_path).expanduser().resolve()

    rows = []

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()

            if len(parts) < len(POS_COLUMNS):
                continue

            if not re.match(r"^\d{4}-\d{2}-\d{2}T", parts[0]):
                continue

            row = {}
            for name, value in zip(POS_COLUMNS, parts[:len(POS_COLUMNS)]):
                if name in ("epoch_utc", "soln"):
                    row[name] = value
                else:
                    row[name] = float(value)

            rows.append(row)

    if not rows:
        raise ValueError(f"No POS data rows found in: {p}")

    df = pd.DataFrame(rows)
    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], utc=True)

    return df


def find_pos_files(run_dir: str | Path, use_smoothed_pos: bool) -> list[Path]:
    rd = Path(run_dir).expanduser().resolve()

    if not rd.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {rd}")

    pos_files = sorted(rd.glob("*.POS"))

    if use_smoothed_pos:
        selected = [p for p in pos_files if p.name.endswith("_smoothed.POS")]
    else:
        selected = [p for p in pos_files if not p.name.endswith("_smoothed.POS")]

    if not selected:
        kind = "smoothed" if use_smoothed_pos else "unsmoothed"
        raise FileNotFoundError(f"No {kind} POS files found in run directory: {rd}")

    return selected


def read_run_pos_timeseries(run_dir: str | Path, use_smoothed_pos: bool) -> tuple[pd.DataFrame, list[Path], dict]:
    pos_files = find_pos_files(run_dir, use_smoothed_pos=use_smoothed_pos)

    dfs = []
    for p in pos_files:
        df = read_pos_file(p)
        df["source_pos_file"] = str(p)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("epoch_utc")
    out = out.drop_duplicates(subset=["epoch_utc"], keep="last").reset_index(drop=True)

    ref_info = _read_pos_reference_info(pos_files[0])

    return out, pos_files, ref_info


def ecef_to_geodetic_wgs84(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p == 0:
        lat = math.copysign(math.pi / 2.0, z)
        h = abs(z) - a * math.sqrt(1.0 - e2)
        return math.degrees(lat), math.degrees(lon), h

    lat = math.atan2(z, p * (1.0 - e2))

    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(z, p * (1.0 - e2 * N / (N + h)))

        if abs(lat_new - lat) < 1e-14:
            lat = lat_new
            break

        lat = lat_new

    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), h


def ecef_to_enu_series(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ref_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, z0 = ref_xyz
    lat0_deg, lon0_deg, _ = ecef_to_geodetic_wgs84(x0, y0, z0)

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    dz = np.asarray(z, dtype=float) - z0

    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def _estimate_median_epoch_interval_sec(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return np.nan

    dt = df["epoch_utc"].diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        return np.nan

    return float(dt.median())


def _select_internal_reference_segment(df: pd.DataFrame, last_hours: float) -> pd.DataFrame:
    last_time = df["epoch_utc"].max()
    cutoff = last_time - pd.Timedelta(hours=float(last_hours))

    ref_df = df[df["epoch_utc"] >= cutoff].copy()

    min_rows = max(10, int(0.05 * len(df)))

    if len(ref_df) < min_rows:
        start_idx = int(0.75 * len(df))
        ref_df = df.iloc[start_idx:].copy()

    return ref_df


def _linear_slope_mm_per_hour(times: pd.Series, values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan

    t_hours = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    y = values.to_numpy(dtype=float)

    ok = np.isfinite(t_hours) & np.isfinite(y)
    t_hours = t_hours[ok]
    y = y[ok]

    if len(y) < 2:
        return np.nan

    if np.nanmax(t_hours) - np.nanmin(t_hours) <= 0:
        return np.nan

    slope_m_per_hour = np.polyfit(t_hours, y, 1)[0]
    return float(slope_m_per_hour * 1000.0)


def detect_convergence(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    c = dict(DEFAULT_CONVERGENCE_CONFIG)
    if cfg:
        c.update(cfg)

    work = df.copy().sort_values("epoch_utc").reset_index(drop=True)

    ref_df = _select_internal_reference_segment(work, c["internal_reference_last_hours"])

    ref_dE = float(ref_df["pos_dE_m"].median())
    ref_dN = float(ref_df["pos_dN_m"].median())
    ref_dU = float(ref_df["pos_dU_m"].median())

    work["conv_E_m"] = work["pos_dE_m"] - ref_dE
    work["conv_N_m"] = work["pos_dN_m"] - ref_dN
    work["conv_U_m"] = work["pos_dU_m"] - ref_dU

    median_dt = _estimate_median_epoch_interval_sec(work)
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 15.0

    expected_n = int(round(float(c["persistence_window_sec"]) / median_dt)) + 1
    min_n = max(3, int(float(c["min_window_fraction"]) * expected_n))

    convergence_row = None
    convergence_diag = {}

    for i in range(len(work)):
        t0 = work.loc[i, "epoch_utc"]
        t1 = t0 + pd.Timedelta(seconds=float(c["persistence_window_sec"]))

        w = work[(work["epoch_utc"] >= t0) & (work["epoch_utc"] <= t1)].copy()

        if len(w) < min_n:
            continue

        max_abs_E = float(w["conv_E_m"].abs().max())
        max_abs_N = float(w["conv_N_m"].abs().max())
        max_abs_U = float(w["conv_U_m"].abs().max())

        std_E = float(w["conv_E_m"].std(ddof=1))
        std_N = float(w["conv_N_m"].std(ddof=1))
        std_U = float(w["conv_U_m"].std(ddof=1))

        slope_E = _linear_slope_mm_per_hour(w["epoch_utc"], w["conv_E_m"])
        slope_N = _linear_slope_mm_per_hour(w["epoch_utc"], w["conv_N_m"])
        slope_U = _linear_slope_mm_per_hour(w["epoch_utc"], w["conv_U_m"])

        tests = [
            max_abs_E <= float(c["threshold_E_m"]),
            max_abs_N <= float(c["threshold_N_m"]),
            max_abs_U <= float(c["threshold_U_m"]),
            std_E <= float(c["threshold_rolling_std_E_m"]),
            std_N <= float(c["threshold_rolling_std_N_m"]),
            std_U <= float(c["threshold_rolling_std_U_m"]),
            abs(slope_E) <= float(c["threshold_slope_E_mm_per_hour"]),
            abs(slope_N) <= float(c["threshold_slope_N_mm_per_hour"]),
            abs(slope_U) <= float(c["threshold_slope_U_mm_per_hour"]),
        ]

        if all(tests):
            convergence_row = i
            convergence_diag = {
                "conv_window_max_abs_E_m": max_abs_E,
                "conv_window_max_abs_N_m": max_abs_N,
                "conv_window_max_abs_U_m": max_abs_U,
                "conv_window_std_E_m": std_E,
                "conv_window_std_N_m": std_N,
                "conv_window_std_U_m": std_U,
                "conv_window_slope_E_mm_per_hour": slope_E,
                "conv_window_slope_N_mm_per_hour": slope_N,
                "conv_window_slope_U_mm_per_hour": slope_U,
                "conv_window_n_epochs": int(len(w)),
            }
            break

    threshold_fields = {
        k: c[k] for k in c
        if k.startswith("threshold_") or k.endswith("_sec")
    }

    if convergence_row is None:
        return {
            "convergence_found": False,
            "convergence_epoch_utc": "",
            "convergence_delay_sec": np.nan,
            "internal_reference_last_hours": c["internal_reference_last_hours"],
            "internal_ref_dE_m": ref_dE,
            "internal_ref_dN_m": ref_dN,
            "internal_ref_dU_m": ref_dU,
            "flags": ["NO_CONVERGENCE"],
            **threshold_fields,
            "conv_window_max_abs_E_m": np.nan,
            "conv_window_max_abs_N_m": np.nan,
            "conv_window_max_abs_U_m": np.nan,
            "conv_window_std_E_m": np.nan,
            "conv_window_std_N_m": np.nan,
            "conv_window_std_U_m": np.nan,
            "conv_window_slope_E_mm_per_hour": np.nan,
            "conv_window_slope_N_mm_per_hour": np.nan,
            "conv_window_slope_U_mm_per_hour": np.nan,
            "conv_window_n_epochs": 0,
        }

    convergence_time = work.loc[convergence_row, "epoch_utc"]
    first_time = work["epoch_utc"].iloc[0]
    delay_sec = float((convergence_time - first_time).total_seconds())

    return {
        "convergence_found": True,
        "convergence_epoch_utc": _as_iso_z(convergence_time),
        "convergence_delay_sec": delay_sec,
        "internal_reference_last_hours": c["internal_reference_last_hours"],
        "internal_ref_dE_m": ref_dE,
        "internal_ref_dN_m": ref_dN,
        "internal_ref_dU_m": ref_dU,
        "flags": [],
        **threshold_fields,
        **convergence_diag,
    }


def _stats(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()

    if len(v) == 0:
        return {
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }

    return {
        "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
    }


def _add_stats(row: dict, df: pd.DataFrame, prefix: str, components: dict[str, str]) -> None:
    for out_name, col in components.items():
        if col in df.columns:
            s = _stats(df[col])
        else:
            s = {
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
            }

        row[f"{prefix}_std_{out_name}"] = s["std"]
        row[f"{prefix}_min_{out_name}"] = s["min"]
        row[f"{prefix}_max_{out_name}"] = s["max"]
        row[f"{prefix}_range_{out_name}"] = s["range"]


def _xyz_primary_secondary(conv_df: pd.DataFrame) -> dict:
    if len(conv_df) == 0:
        return {
            "primary_X_m": np.nan,
            "primary_Y_m": np.nan,
            "primary_Z_m": np.nan,
            "primary_lat_deg": np.nan,
            "primary_lon_deg": np.nan,
            "primary_h_m": np.nan,
            "mean_X_m": np.nan,
            "mean_Y_m": np.nan,
            "mean_Z_m": np.nan,
            "mean_lat_deg": np.nan,
            "mean_lon_deg": np.nan,
            "mean_h_m": np.nan,
        }

    X_med = float(conv_df["X_m"].median())
    Y_med = float(conv_df["Y_m"].median())
    Z_med = float(conv_df["Z_m"].median())
    lat_med, lon_med, h_med = ecef_to_geodetic_wgs84(X_med, Y_med, Z_med)

    X_mean = float(conv_df["X_m"].mean())
    Y_mean = float(conv_df["Y_m"].mean())
    Z_mean = float(conv_df["Z_m"].mean())
    lat_mean, lon_mean, h_mean = ecef_to_geodetic_wgs84(X_mean, Y_mean, Z_mean)

    return {
        "primary_X_m": X_med,
        "primary_Y_m": Y_med,
        "primary_Z_m": Z_med,
        "primary_lat_deg": lat_med,
        "primary_lon_deg": lon_med,
        "primary_h_m": h_med,
        "mean_X_m": X_mean,
        "mean_Y_m": Y_mean,
        "mean_Z_m": Z_mean,
        "mean_lat_deg": lat_mean,
        "mean_lon_deg": lon_mean,
        "mean_h_m": h_mean,
    }


def collect_log_qc(run_dir: str | Path, scan_trace_files: bool = False) -> dict:
    rd = Path(run_dir).expanduser().resolve()

    files = list(rd.glob("stdout_*.txt"))

    if scan_trace_files:
        files += list(rd.glob("*.TRACE"))

    warning_count = 0
    critical_count = 0

    critical_patterns = [
        "inputs finished at epoch #1",
        "segmentation fault",
        "traceback",
        "exception",
        "fatal",
        "aborting",
        "pea execution failed",
    ]

    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        low_text = text.lower()

        warning_count += low_text.count("warning:")

        if "no more data available" in low_text and "processed epoch #" not in low_text:
            critical_count += 1

        for pattern in critical_patterns:
            if pattern in low_text:
                critical_count += 1

    if critical_count > 0:
        status = "CRITICAL_WARNINGS_PRESENT"
    elif warning_count > 0:
        status = "WARNINGS_PRESENT"
    else:
        status = "OK"

    return {
        "trace_qc_status": status,
        "trace_warning_count": int(warning_count),
        "trace_critical_warning_count": int(critical_count),
        "log_files_scanned": int(len(files)),
    }


def _prepare_run_analysis(run_dir: str | Path, cfg: dict) -> dict:
    rd = Path(run_dir).expanduser().resolve()

    convergence_df, convergence_pos_files, conv_ref_info = read_run_pos_timeseries(
        rd,
        use_smoothed_pos=bool(cfg["convergence_use_smoothed_pos"]),
    )

    solution_df, solution_pos_files, sol_ref_info = read_run_pos_timeseries(
        rd,
        use_smoothed_pos=bool(cfg["solution_use_smoothed_pos"]),
    )

    conv = detect_convergence(convergence_df, cfg=cfg)

    if conv["convergence_found"]:
        conv_time = pd.to_datetime(conv["convergence_epoch_utc"], utc=True)
        solution_conv_df = solution_df[solution_df["epoch_utc"] >= conv_time].copy()
    else:
        solution_conv_df = solution_df.iloc[0:0].copy()

    xyz = _xyz_primary_secondary(solution_conv_df)

    return {
        "run_dir": rd,
        "run_label": rd.name,
        "dataset_name": rd.name.split("_test_")[0],
        "convergence_df": convergence_df,
        "solution_df": solution_df,
        "solution_conv_df": solution_conv_df,
        "convergence_pos_files": convergence_pos_files,
        "solution_pos_files": solution_pos_files,
        "ref_info": sol_ref_info,
        "conv_ref_info": conv_ref_info,
        "conv": conv,
        "xyz": xyz,
    }


def _finalize_run_row(pre: dict, series_ref_xyz: tuple[float, float, float], cfg: dict) -> dict:
    solution_df = pre["solution_df"].copy()
    solution_conv_df = pre["solution_conv_df"].copy()
    convergence_df = pre["convergence_df"].copy()

    e_all, n_all, u_all = ecef_to_enu_series(
        solution_df["X_m"].to_numpy(),
        solution_df["Y_m"].to_numpy(),
        solution_df["Z_m"].to_numpy(),
        series_ref_xyz,
    )
    solution_df["E_m"] = e_all
    solution_df["N_m"] = n_all
    solution_df["U_m"] = u_all

    if len(solution_conv_df) > 0:
        e_conv, n_conv, u_conv = ecef_to_enu_series(
            solution_conv_df["X_m"].to_numpy(),
            solution_conv_df["Y_m"].to_numpy(),
            solution_conv_df["Z_m"].to_numpy(),
            series_ref_xyz,
        )
        solution_conv_df["E_m"] = e_conv
        solution_conv_df["N_m"] = n_conv
        solution_conv_df["U_m"] = u_conv

    e_uns, n_uns, u_uns = ecef_to_enu_series(
        convergence_df["X_m"].to_numpy(),
        convergence_df["Y_m"].to_numpy(),
        convergence_df["Z_m"].to_numpy(),
        series_ref_xyz,
    )
    convergence_df["E_m"] = e_uns
    convergence_df["N_m"] = n_uns
    convergence_df["U_m"] = u_uns

    conv = pre["conv"]
    if conv["convergence_found"]:
        conv_time = pd.to_datetime(conv["convergence_epoch_utc"], utc=True)
        qc_unsmoothed_conv_df = convergence_df[convergence_df["epoch_utc"] >= conv_time].copy()
    else:
        qc_unsmoothed_conv_df = convergence_df.iloc[0:0].copy()

    xyz = pre["xyz"]

    if np.isfinite(xyz["primary_X_m"]):
        E_primary, N_primary, U_primary = ecef_to_enu_series(
            np.array([xyz["primary_X_m"]]),
            np.array([xyz["primary_Y_m"]]),
            np.array([xyz["primary_Z_m"]]),
            series_ref_xyz,
        )
        E_primary = float(E_primary[0])
        N_primary = float(N_primary[0])
        U_primary = float(U_primary[0])
    else:
        E_primary = N_primary = U_primary = np.nan

    if np.isfinite(xyz["mean_X_m"]):
        E_mean, N_mean, U_mean = ecef_to_enu_series(
            np.array([xyz["mean_X_m"]]),
            np.array([xyz["mean_Y_m"]]),
            np.array([xyz["mean_Z_m"]]),
            series_ref_xyz,
        )
        E_mean = float(E_mean[0])
        N_mean = float(N_mean[0])
        U_mean = float(U_mean[0])
    else:
        E_mean = N_mean = U_mean = np.nan

    first_epoch = solution_df["epoch_utc"].min()
    last_epoch = solution_df["epoch_utc"].max()

    duration_total_sec = float((last_epoch - first_epoch).total_seconds()) if len(solution_df) > 1 else 0.0

    if len(solution_conv_df) > 1:
        duration_converged_sec = float(
            (solution_conv_df["epoch_utc"].max() - solution_conv_df["epoch_utc"].min()).total_seconds()
        )
    else:
        duration_converged_sec = 0.0

    flags = list(conv.get("flags", []))

    if conv["convergence_found"] and duration_converged_sec < 7200:
        flags.append("SHORT_CONVERGED_INTERVAL")

    qc = collect_log_qc(
        pre["run_dir"],
        scan_trace_files=bool(cfg["scan_trace_files"]),
    )

    if qc["trace_warning_count"] > 0:
        flags.append("TRACE_WARNINGS_PRESENT")

    if qc["trace_critical_warning_count"] > 0:
        flags.append("TRACE_CRITICAL_WARNINGS_PRESENT")

    if not flags:
        flags = ["OK"]

    row = {
        "dataset_name": pre["dataset_name"],
        "run_label": pre["run_label"],
        "run_dir": str(pre["run_dir"]),

        "convergence_pos_files": ";".join(str(p) for p in pre["convergence_pos_files"]),
        "solution_pos_files": ";".join(str(p) for p in pre["solution_pos_files"]),
        "convergence_pos_source": "smoothed" if bool(cfg["convergence_use_smoothed_pos"]) else "unsmoothed",
        "solution_pos_source": "smoothed" if bool(cfg["solution_use_smoothed_pos"]) else "unsmoothed",

        "station_id": pre["ref_info"].get("station_id", ""),
        "reference_frame": pre["ref_info"].get("reference_frame", ""),
        "pos_xyz_reference_position": pre["ref_info"].get("xyz_reference_position", ""),
        "pos_neu_reference_position": pre["ref_info"].get("neu_reference_position", ""),

        "time_mean_all_epochs_utc": _as_iso_z(_mean_time_utc(solution_df["epoch_utc"])),
        "time_mean_converged_epochs_utc": _as_iso_z(_mean_time_utc(solution_conv_df["epoch_utc"])),
        "time_first_epoch_utc": _as_iso_z(first_epoch),
        "time_last_epoch_utc": _as_iso_z(last_epoch),
        "convergence_epoch_utc": conv["convergence_epoch_utc"],
        "convergence_delay_sec": conv["convergence_delay_sec"],

        "n_epochs_total": int(len(solution_df)),
        "n_epochs_converged": int(len(solution_conv_df)),
        "duration_total_sec": duration_total_sec,
        "duration_converged_sec": duration_converged_sec,
        "converged_fraction": float(len(solution_conv_df) / len(solution_df)) if len(solution_df) else np.nan,

        "solution_method_primary": "median_converged_smoothed_after_unsmoothed_convergence",
        "X_m": xyz["primary_X_m"],
        "Y_m": xyz["primary_Y_m"],
        "Z_m": xyz["primary_Z_m"],
        "lon_deg": xyz["primary_lon_deg"],
        "lat_deg": xyz["primary_lat_deg"],
        "h_m": xyz["primary_h_m"],
        "E_m": E_primary,
        "N_m": N_primary,
        "U_m": U_primary,

        "X_mean_m": xyz["mean_X_m"],
        "Y_mean_m": xyz["mean_Y_m"],
        "Z_mean_m": xyz["mean_Z_m"],
        "lon_mean_deg": xyz["mean_lon_deg"],
        "lat_mean_deg": xyz["mean_lat_deg"],
        "h_mean_m": xyz["mean_h_m"],
        "E_mean_m": E_mean,
        "N_mean_m": N_mean,
        "U_mean_m": U_mean,

        "convergence_found": bool(conv["convergence_found"]),
        "convergence_method": "unsmoothed_pos_internal_final_window_reference__threshold_persistence_std_slope",
        "threshold_E_m": conv["threshold_E_m"],
        "threshold_N_m": conv["threshold_N_m"],
        "threshold_U_m": conv["threshold_U_m"],
        "threshold_rolling_std_E_m": conv["threshold_rolling_std_E_m"],
        "threshold_rolling_std_N_m": conv["threshold_rolling_std_N_m"],
        "threshold_rolling_std_U_m": conv["threshold_rolling_std_U_m"],
        "threshold_slope_E_mm_per_hour": conv["threshold_slope_E_mm_per_hour"],
        "threshold_slope_N_mm_per_hour": conv["threshold_slope_N_mm_per_hour"],
        "threshold_slope_U_mm_per_hour": conv["threshold_slope_U_mm_per_hour"],
        "persistence_window_sec": conv["persistence_window_sec"],
        "rolling_window_sec": conv["rolling_window_sec"],

        "conv_window_max_abs_E_m": conv["conv_window_max_abs_E_m"],
        "conv_window_max_abs_N_m": conv["conv_window_max_abs_N_m"],
        "conv_window_max_abs_U_m": conv["conv_window_max_abs_U_m"],
        "conv_window_std_E_m": conv["conv_window_std_E_m"],
        "conv_window_std_N_m": conv["conv_window_std_N_m"],
        "conv_window_std_U_m": conv["conv_window_std_U_m"],
        "conv_window_slope_E_mm_per_hour": conv["conv_window_slope_E_mm_per_hour"],
        "conv_window_slope_N_mm_per_hour": conv["conv_window_slope_N_mm_per_hour"],
        "conv_window_slope_U_mm_per_hour": conv["conv_window_slope_U_mm_per_hour"],
        "conv_window_n_epochs": conv["conv_window_n_epochs"],

        "trace_qc_status": qc["trace_qc_status"],
        "trace_warning_count": qc["trace_warning_count"],
        "trace_critical_warning_count": qc["trace_critical_warning_count"],
        "log_files_scanned": qc["log_files_scanned"],
        "qc_flags": ";".join(flags),
    }

    components = {
        "X_m": "X_m",
        "Y_m": "Y_m",
        "Z_m": "Z_m",
        "lon_deg": "lon_deg",
        "lat_deg": "lat_deg",
        "h_m": "h_m",
        "E_m": "E_m",
        "N_m": "N_m",
        "U_m": "U_m",
    }

    _add_stats(row, solution_conv_df, "solution_conv", components)
    _add_stats(row, qc_unsmoothed_conv_df, "qc_unsmoothed_conv", components)
    _add_stats(row, solution_df, "full_solution", components)
    _add_stats(row, convergence_df, "full_unsmoothed", components)

    return row


def build_timeseries_out(
    run_dirs: list[str | Path],
    output_path: str | Path,
    convergence_config: dict | None = None,
    overwrite: bool = True,
) -> dict:
    cfg = dict(DEFAULT_CONVERGENCE_CONFIG)
    if convergence_config:
        cfg.update(convergence_config)

    prepared = []

    for run_dir in run_dirs:
        prepared.append(_prepare_run_analysis(run_dir, cfg))

    ref_xyz = None
    ref_label = ""

    for pre in prepared:
        xyz = pre["xyz"]
        if pre["conv"]["convergence_found"] and np.isfinite(xyz["primary_X_m"]):
            ref_xyz = (
                float(xyz["primary_X_m"]),
                float(xyz["primary_Y_m"]),
                float(xyz["primary_Z_m"]),
            )
            ref_label = pre["run_label"]
            break

    if ref_xyz is None:
        raise RuntimeError("Could not define series ENU reference: no converged daily solution found.")

    rows = []

    for pre in prepared:
        row = _finalize_run_row(pre, ref_xyz, cfg)
        row["series_enu_reference_run_label"] = ref_label
        row["series_enu_reference_X_m"] = ref_xyz[0]
        row["series_enu_reference_Y_m"] = ref_xyz[1]
        row["series_enu_reference_Z_m"] = ref_xyz[2]
        rows.append(row)

    out_df = pd.DataFrame(rows)

    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        old = pd.read_csv(p, sep="\t")
        out_df = pd.concat([old, out_df], ignore_index=True)

    out_df.to_csv(p, sep="\t", index=False, float_format="%.12g")

    return {
        "ok": True,
        "output_path": str(p),
        "n_rows": int(len(out_df)),
        "series_enu_reference_run_label": ref_label,
        "series_enu_reference_xyz": ref_xyz,
        "dataframe": out_df,
    }
