
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
            qc_flags=["EMPTY_POS"],
        )

    q_counts = _q_counts(df)
    q1 = q_counts.get(1, 0)
    q1_percent = 100.0 * q1 / len(df) if len(df) else 0.0
    if q1_percent < inputs.min_fixed_percent:
        qc_flags.append("LOW_FIXED_PERCENT")

    window_start = df["time"].max() - pd.Timedelta(minutes=float(inputs.final_window_minutes))
    final_df = df[df["time"] >= window_start]

    if inputs.final_window_minutes < inputs.recommended_min_final_window_minutes:
        qc_flags.append("FINAL_WINDOW_BELOW_RECOMMENDED_MINIMUM")

    if inputs.q_fixed_only_for_final:
        final_df = final_df[final_df["Q"] == 1]
        solution_method = f"Q=1 fixed only, last {inputs.final_window_minutes:g} min"
    else:
        solution_method = f"all Q, last {inputs.final_window_minutes:g} min"

    if final_df.empty:
        qc_flags.append("NO_EPOCHS_IN_FINAL_WINDOW")
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

        # Local scatter of final-window epochs around the final rover solution.
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
        qc_flags=qc_flags,
    )
