
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
from models import ParsedPos, BaselinePair, CorsSolution, UserInputs, BaselineSolution


def _mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.mean())


def _std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(s.std(ddof=1))


def _min(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.min())


def _max(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.max())


def ecef_to_geodetic(x: float, y: float, z_in: float) -> tuple[float, float, float]:
    """
    RTKLIB 2.4.2 p13 equivalent of ecef2pos().

    Source convention:
        RE_WGS84 = 6378137.0
        FE_WGS84 = 1.0 / 298.257223563

    Returns:
        lat_rad, lon_rad, h_m
    """
    RE_WGS84 = 6378137.0
    FE_WGS84 = 1.0 / 298.257223563

    e2 = FE_WGS84 * (2.0 - FE_WGS84)
    r2 = x * x + y * y

    z = z_in
    zk = 0.0
    v = RE_WGS84

    while abs(z - zk) >= 1.0e-4:
        zk = z
        sinp = z / math.sqrt(r2 + z * z)
        v = RE_WGS84 / math.sqrt(1.0 - e2 * sinp * sinp)
        z = z_in + v * e2 * sinp

    if r2 > 1.0e-12:
        lat = math.atan(z / math.sqrt(r2))
        lon = math.atan2(y, x)
    else:
        lat = math.pi / 2.0 if z_in > 0.0 else -math.pi / 2.0
        lon = 0.0

    h = math.sqrt(r2 + z * z) - v

    return lat, lon, h

def ecef_diff_to_enu(dx, dy, dz, lat_deg, lon_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def _q_counts(df):
    if "Q" not in df.columns or df.empty:
        return {}
    return {int(k): int(v) for k, v in df["Q"].value_counts().to_dict().items()}


def _fixed_segments(df: pd.DataFrame) -> tuple[list[dict], float | None]:
    """
    Identify continuous Q=1 fixed segments.

    Continuity is determined from the median epoch spacing of the fixed epochs.
    A new segment starts when the time gap exceeds 2.5 times the median spacing.
    """
    if df.empty or "time" not in df.columns:
        return [], None

    tmp = df.sort_values("time").copy()
    times = pd.to_datetime(tmp["time"])

    if len(times) < 2:
        if len(times) == 1:
            return [{
                "start": times.iloc[0],
                "end": times.iloc[0],
                "duration_sec": 0.0,
                "epochs": 1,
            }], None
        return [], None

    dt_sec = times.diff().dt.total_seconds().dropna()
    dt_sec = dt_sec[dt_sec > 0]

    median_dt = float(dt_sec.median()) if not dt_sec.empty else None
    gap_threshold = max(2.5 * median_dt, 2.0) if median_dt else 2.0

    groups = []
    current_start = 0

    gaps = times.diff().dt.total_seconds().fillna(0.0)
    for i in range(1, len(tmp)):
        if float(gaps.iloc[i]) > gap_threshold:
            groups.append((current_start, i - 1))
            current_start = i
    groups.append((current_start, len(tmp) - 1))

    segments = []
    for a, b in groups:
        start = times.iloc[a]
        end = times.iloc[b]
        duration_sec = float((end - start).total_seconds())
        if median_dt is not None and b > a:
            duration_sec += median_dt
        segments.append({
            "start": start,
            "end": end,
            "duration_sec": duration_sec,
            "epochs": int(b - a + 1),
        })

    return segments, median_dt


def _segments_total_duration_min(segments: list[dict]) -> float | None:
    if not segments:
        return None
    return sum(float(s["duration_sec"]) for s in segments) / 60.0


def _longest_segment(segments: list[dict]) -> dict | None:
    if not segments:
        return None
    return max(segments, key=lambda s: (float(s["duration_sec"]), int(s["epochs"])))


def build_pos_qc_table(parsed: ParsedPos) -> dict:
    df = parsed.dataframe
    if df.empty:
        return {
            "total_epochs": 0,
            "q_counts": {},
        }

    return {
        "total_epochs": int(len(df)),
        "first_epoch": df["time"].min(),
        "last_epoch": df["time"].max(),
        "q_counts": _q_counts(df),
        "ratio_min": _min(df["ratio"]),
        "ratio_mean": _mean(df["ratio"]),
        "ratio_max": _max(df["ratio"]),
        "ns_mean": _mean(df["ns"]),
        "all_X_mean": _mean(df["X"]),
        "all_Y_mean": _mean(df["Y"]),
        "all_Z_mean": _mean(df["Z"]),
        "all_X_std": _std(df["X"]),
        "all_Y_std": _std(df["Y"]),
        "all_Z_std": _std(df["Z"]),
    }


def compute_baseline_solution(
    parsed: ParsedPos,
    pair: BaselinePair,
    cors: CorsSolution,
    inputs: UserInputs,
) -> BaselineSolution:
    df = parsed.dataframe.copy()
    qc_flags = []

    if df.empty:
        return BaselineSolution(
            run_label=pair.run_label,
            benchmark_id=pair.rover.marker_name or pair.rover.filename,
            rover_file=pair.rover.path,
            base_file=pair.base.path,
            solution_method="NO_SOLUTION",
            final_window_minutes=inputs.final_window_minutes,
            X_m=None, Y_m=None, Z_m=None,
            lon_deg=None, lat_deg=None, h_m=None,
            std_X_m=None, std_Y_m=None, std_Z_m=None,
            std_lon_m=None, std_lat_m=None, std_h_m=None,
            baseline_dX_m=None, baseline_dY_m=None, baseline_dZ_m=None,
            baseline_E_m=None, baseline_N_m=None, baseline_U_m=None,
            baseline_length_m=None,
            q1_fixed_percent=None,
            ratio_min=None, ratio_mean=None, ratio_max=None,
            n_fixed_epochs_used=None,
            fixed_time_start=None,
            fixed_time_end=None,
            fixed_total_duration_min=None,
            longest_fixed_segment_start=None,
            longest_fixed_segment_end=None,
            longest_fixed_segment_duration_min=None,
            longest_fixed_segment_epochs=None,
            qc_flags=["EMPTY_POS"],
        )

    q_counts = _q_counts(df)
    q1 = q_counts.get(1, 0)
    q1_percent = 100.0 * q1 / len(df) if len(df) else 0.0
    if q1_percent < inputs.min_fixed_percent:
        qc_flags.append("LOW_FIXED_PERCENT")

    fixed_df = df[df["Q"] == 1].copy() if "Q" in df.columns else pd.DataFrame()
    fixed_segments, fixed_median_dt_sec = _fixed_segments(fixed_df)
    longest = _longest_segment(fixed_segments)

    n_fixed_epochs_used = int(len(fixed_df))
    fixed_time_start = None
    fixed_time_end = None
    fixed_total_duration_min = _segments_total_duration_min(fixed_segments)
    longest_fixed_segment_start = None
    longest_fixed_segment_end = None
    longest_fixed_segment_duration_min = None
    longest_fixed_segment_epochs = None

    if not fixed_df.empty:
        fixed_time_start = pd.to_datetime(fixed_df["time"]).min()
        fixed_time_end = pd.to_datetime(fixed_df["time"]).max()

    if longest is not None:
        longest_fixed_segment_start = longest["start"]
        longest_fixed_segment_end = longest["end"]
        longest_fixed_segment_duration_min = float(longest["duration_sec"]) / 60.0
        longest_fixed_segment_epochs = int(longest["epochs"])

        if float(longest["duration_sec"]) < float(inputs.min_continuous_fixed_duration_sec):
            qc_flags.append("SHORT_LONGEST_FIXED_SEGMENT")

    if len(fixed_segments) > 1:
        qc_flags.append("FRAGMENTED_FIXED_SOLUTION")

    if inputs.q_fixed_only_for_final:
        final_df = fixed_df
        solution_method = "all Q=1 fixed POS epochs"
        if final_df.empty:
            qc_flags.append("NO_FIXED_EPOCHS")
    else:
        final_df = df.copy()
        solution_method = "all POS epochs, Q not filtered"
        qc_flags.append("ALL_Q_SOLUTION")

    if final_df.empty:
        X = Y = Z = sx = sy = sz = None
        rover_lon_deg = rover_lat_deg = rover_h = None
        std_lon_m = std_lat_m = std_h_m = None
    else:
        X = _mean(final_df["X"])
        Y = _mean(final_df["Y"])
        Z = _mean(final_df["Z"])
        rover_lat_rad, rover_lon_rad, rover_h = ecef_to_geodetic(X, Y, Z)
        rover_lat_deg = math.degrees(rover_lat_rad)
        rover_lon_deg = math.degrees(rover_lon_rad)

        # Local scatter of selected epochs around the final rover solution.
        # std_lon_m = East scatter, std_lat_m = North scatter, std_h_m = Up scatter.
        e_res = []
        n_res = []
        u_res = []
        for _, row in final_df.iterrows():
            e_i, n_i, u_i = ecef_diff_to_enu(
                float(row["X"]) - X,
                float(row["Y"]) - Y,
                float(row["Z"]) - Z,
                rover_lat_deg,
                rover_lon_deg,
            )
            e_res.append(e_i)
            n_res.append(n_i)
            u_res.append(u_i)

        sx = _std(final_df["X"])
        sy = _std(final_df["Y"])
        sz = _std(final_df["Z"])
        std_lon_m = _std(pd.Series(e_res))
        std_lat_m = _std(pd.Series(n_res))
        std_h_m = _std(pd.Series(u_res))

    if X is None or Y is None or Z is None:
        dX = dY = dZ = E = N = U = length = None
    else:
        dX = X - cors.X_m
        dY = Y - cors.Y_m
        dZ = Z - cors.Z_m
        cors_lat_rad, cors_lon_rad, _ = ecef_to_geodetic(cors.X_m, cors.Y_m, cors.Z_m)
        E, N, U = ecef_diff_to_enu(
            dX,
            dY,
            dZ,
            math.degrees(cors_lat_rad),
            math.degrees(cors_lon_rad),
        )
        length = math.sqrt(dX*dX + dY*dY + dZ*dZ)

    ratio_series = pd.to_numeric(final_df["ratio"], errors="coerce") if not final_df.empty else pd.Series(dtype=float)

    return BaselineSolution(
        run_label=pair.run_label,
        benchmark_id=pair.rover.marker_name or pair.rover.filename,
        rover_file=pair.rover.path,
        base_file=pair.base.path,
        solution_method=solution_method,
        final_window_minutes=inputs.final_window_minutes,
        X_m=X, Y_m=Y, Z_m=Z,
        lon_deg=rover_lon_deg, lat_deg=rover_lat_deg, h_m=rover_h,
        std_X_m=sx, std_Y_m=sy, std_Z_m=sz,
        std_lon_m=std_lon_m, std_lat_m=std_lat_m, std_h_m=std_h_m,
        baseline_dX_m=dX, baseline_dY_m=dY, baseline_dZ_m=dZ,
        baseline_E_m=E, baseline_N_m=N, baseline_U_m=U,
        baseline_length_m=length,
        q1_fixed_percent=q1_percent,
        ratio_min=_min(ratio_series),
        ratio_mean=_mean(ratio_series),
        ratio_max=_max(ratio_series),
        n_fixed_epochs_used=n_fixed_epochs_used,
        fixed_time_start=fixed_time_start,
        fixed_time_end=fixed_time_end,
        fixed_total_duration_min=fixed_total_duration_min,
        longest_fixed_segment_start=longest_fixed_segment_start,
        longest_fixed_segment_end=longest_fixed_segment_end,
        longest_fixed_segment_duration_min=longest_fixed_segment_duration_min,
        longest_fixed_segment_epochs=longest_fixed_segment_epochs,
        qc_flags=qc_flags,
    )
