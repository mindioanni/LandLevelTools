
from __future__ import annotations

from pathlib import Path
import base64
import html
import io
import math
import pandas as pd

from models import UserInputs, CorsSolution, RinexObsFile, BaselinePair, ResolvedProducts, RunResult, BaselineSolution
from baseline_qc import ecef_to_geodetic, ecef_diff_to_enu


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value, digits=4):
    if value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"



def _rinex_duration_minutes(item) -> float | None:
    first_obs = getattr(item, "first_obs", None)
    last_obs = getattr(item, "last_obs", None)

    if first_obs is None or last_obs is None:
        return None

    try:
        return (last_obs - first_obs).total_seconds() / 60.0
    except Exception:
        return None



def _html_table_from_records(records: list[dict], max_rows: int | None = None) -> str:
    if not records:
        return "<p>No records.</p>"
    df = pd.DataFrame(records)
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_html(index=False, escape=True, border=0)



def _decimal_year_from_utc_timestamp(ts: pd.Timestamp) -> float:
    """
    Convert UTC timestamp to decimal year.
    """
    year_start = pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")
    next_year_start = pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")
    return ts.year + (ts - year_start).total_seconds() / (next_year_start - year_start).total_seconds()


def _cors_ppp_reference_epoch_note(cors: CorsSolution) -> str:
    """
    Build CORS PPP reference epoch and reference-frame note from the source GINAN/PEA daily table.
    """
    if cors.daily_table is None or "time_mean_all_epochs_utc" not in cors.daily_table.columns:
        return (
            "<h3>CORS PPP reference epoch and terrestrial frame</h3>"
            "<p>The CORS PPP reference epoch could not be computed because "
            "<code>time_mean_all_epochs_utc</code> was not available in the parsed source report.</p>"
            "<p><strong>Reference frame:</strong> IGS operational frame aligned to ITRF2020. "
            "The exact frame label was not explicitly parsed from the source GINAN/PEA report.</p>"
        )

    times = pd.to_datetime(
        cors.daily_table["time_mean_all_epochs_utc"],
        utc=True,
        errors="coerce",
    ).dropna()

    if times.empty:
        return (
            "<h3>CORS PPP reference epoch and terrestrial frame</h3>"
            "<p>The CORS PPP reference epoch could not be computed because no valid "
            "<code>time_mean_all_epochs_utc</code> values were found.</p>"
            "<p><strong>Reference frame:</strong> IGS operational frame aligned to ITRF2020. "
            "The exact frame label was not explicitly parsed from the source GINAN/PEA report.</p>"
        )

    mean_time = times.mean()
    decimal_year = _decimal_year_from_utc_timestamp(mean_time)

    summary_html = _html_table_from_records([{
        "n_times": int(len(times)),
        "mean_time_utc": mean_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ").rstrip("0").replace(".Z", "Z"),
        "decimal_year": f"{decimal_year:.10f}",
        "decimal_year_YY": f"{decimal_year:.2f}",
    }])

    frame_note = (
        "<p><strong>Reference frame:</strong> IGS operational frame aligned to ITRF2020.</p>"
        "<p>The CORS PPP reference solution is assigned to the decimal epoch computed above, "
        "based on the mean of all <code>time_mean_all_epochs_utc</code> values from the "
        "CORS daily/per-file PPP solutions. The terrestrial reference frame is inherited "
        "from the GNSS products and models used in the source GINAN/PEA PPP processing, "
        "primarily the precise orbits/clocks, SINEX/coordinate products and antenna model. "
        "It is not independently estimated by this RTKLIB baseline report.</p>"
        "<p>For traceability, the relevant IGS operational-frame sequence is: "
        "IGS20/igs20.atx, closely related to ITRF2020, from GPS week 2238; "
        "IGb20, based on ITRF2020-u2023, from GPS week 2352; and "
        "IGc20, based on ITRF2020-u2024, from GPS week 2401. "
        "For the present 2026.22 CORS epoch, the expected operational-frame context is "
        "therefore IGc20 / ITRF2020-u2024, unless the source products/configuration indicate otherwise.</p>"
    )

    return (
        "<h3>CORS PPP reference epoch and terrestrial frame</h3>"
        + summary_html
        + frame_note
    )


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")



def _lonlat_to_web_mercator(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """
    Convert WGS84 lon/lat degrees to Web Mercator EPSG:3857 metres.
    Used only for satellite basemap plotting.
    """
    r = 6378137.0
    lon_rad = math.radians(float(lon_deg))
    lat_rad = math.radians(float(lat_deg))

    # Avoid Web Mercator singularities.
    max_lat = 85.05112878
    lat_deg_clamped = max(min(float(lat_deg), max_lat), -max_lat)
    lat_rad = math.radians(lat_deg_clamped)

    x = r * lon_rad
    y = r * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))
    return x, y


def _set_metric_extent(ax, xs: list[float], ys: list[float], margin_m: float = 120.0) -> None:
    """
    Set map extent in EPSG:3857 metres with minimum margin.
    """
    if not xs or not ys:
        return

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = xmax - xmin
    dy = ymax - ymin

    margin = max(margin_m, 0.08 * max(dx, dy, 1.0))

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)


def _add_satellite_basemap(ax) -> str | None:
    """
    Add satellite basemap using contextily/Esri WorldImagery.

    Returns:
        None if successful, otherwise a warning string.
    """
    try:
        import contextily as ctx
    except Exception as exc:
        return f"Satellite basemap was not added because contextily is unavailable: {exc}"

    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            attribution_size=6,
            reset_extent=True,
        )
        return None
    except Exception as exc:
        return f"Satellite basemap was not added: {exc}"


def _static_solution_map(cors: CorsSolution, solutions: list[BaselineSolution]) -> str:
    """
    Static baseline map.

    Output:
    - satellite image basemap,
    - CORS/base station as blue inverted triangle,
    - solved rover point(s) as blue upright triangles,
    - baseline line(s) from CORS to solved rover point(s).

    No error ellipses and no height accuracy bar are plotted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cors_lat_rad, cors_lon_rad, _ = ecef_to_geodetic(cors.X_m, cors.Y_m, cors.Z_m)
    cors_lat_deg = math.degrees(cors_lat_rad)
    cors_lon_deg = math.degrees(cors_lon_rad)

    cors_x, cors_y = _lonlat_to_web_mercator(cors_lon_deg, cors_lat_deg)

    plot_items = []
    xs = [cors_x]
    ys = [cors_y]

    for sol in solutions:
        if sol.lon_deg is None or sol.lat_deg is None:
            continue

        rover_x, rover_y = _lonlat_to_web_mercator(sol.lon_deg, sol.lat_deg)

        plot_items.append({
            "label": sol.benchmark_id,
            "x": rover_x,
            "y": rover_y,
        })

        xs.append(rover_x)
        ys.append(rover_y)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    _set_metric_extent(ax, xs, ys, margin_m=150.0)

    basemap_warning = _add_satellite_basemap(ax)

    # Baseline lines.
    for item in plot_items:
        ax.plot(
            [cors_x, item["x"]],
            [cors_y, item["y"]],
            color="blue",
            linewidth=1.2,
            zorder=4,
        )

    # CORS/base station.
    cors_label = cors.station_id or "CORS"
    ax.scatter(
        [cors_x],
        [cors_y],
        marker="v",
        color="blue",
        s=90,
        label=cors_label,
        zorder=5,
    )
    ax.text(cors_x, cors_y, " " + cors_label, fontsize=8, color="blue", zorder=6)

    # Solved rover points.
    for item in plot_items:
        ax.scatter(
            [item["x"]],
            [item["y"]],
            marker="^",
            color="blue",
            s=80,
            zorder=5,
        )
        ax.text(item["x"], item["y"], " " + str(item["label"]), fontsize=8, color="blue", zorder=6)

    ax.set_title("Static baseline solution map")
    ax.set_xlabel("Web Mercator Easting, EPSG:3857 (m)")
    ax.set_ylabel("Web Mercator Northing, EPSG:3857 (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    if basemap_warning:
        ax.text(
            0.01,
            0.01,
            basemap_warning,
            transform=ax.transAxes,
            fontsize=7,
            color="red",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=10,
        )

    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return f'<img class="plot" src="data:image/png;base64,{encoded}" />'


def _dynamic_trajectory_map(cors: CorsSolution, parsed_pos_items: list) -> str:
    """
    Dynamic trajectory map.

    Output:
    - satellite image basemap,
    - CORS/base station as blue inverted triangle,
    - solved rover trajectory as red dots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cors_lat_rad, cors_lon_rad, _ = ecef_to_geodetic(cors.X_m, cors.Y_m, cors.Z_m)
    cors_lat_deg = math.degrees(cors_lat_rad)
    cors_lon_deg = math.degrees(cors_lon_rad)

    cors_x, cors_y = _lonlat_to_web_mercator(cors_lon_deg, cors_lat_deg)

    traj_x = []
    traj_y = []

    for parsed in parsed_pos_items:
        df = parsed.dataframe
        if df.empty:
            continue

        for _, row in df.iterrows():
            lat_rad, lon_rad, _ = ecef_to_geodetic(float(row["X"]), float(row["Y"]), float(row["Z"]))
            lon_deg = math.degrees(lon_rad)
            lat_deg = math.degrees(lat_rad)
            x, y = _lonlat_to_web_mercator(lon_deg, lat_deg)
            traj_x.append(x)
            traj_y.append(y)

    xs = [cors_x] + traj_x
    ys = [cors_y] + traj_y

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    _set_metric_extent(ax, xs, ys, margin_m=150.0)

    basemap_warning = _add_satellite_basemap(ax)

    if traj_x and traj_y:
        ax.scatter(
            traj_x,
            traj_y,
            marker=".",
            color="red",
            s=6,
            label="Solved rover trajectory",
            zorder=4,
        )

    cors_label = cors.station_id or "CORS"
    ax.scatter(
        [cors_x],
        [cors_y],
        marker="v",
        color="blue",
        s=90,
        label=cors_label,
        zorder=5,
    )
    ax.text(cors_x, cors_y, " " + cors_label, fontsize=8, color="blue", zorder=6)

    ax.set_title("Dynamic rover trajectory map")
    ax.set_xlabel("Web Mercator Easting, EPSG:3857 (m)")
    ax.set_ylabel("Web Mercator Northing, EPSG:3857 (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    if basemap_warning:
        ax.text(
            0.01,
            0.01,
            basemap_warning,
            transform=ax.transAxes,
            fontsize=7,
            color="red",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=10,
        )

    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return f'<img class="plot" src="data:image/png;base64,{encoded}" />'



def _fixed_segment_diagnostic_records(parsed_pos_items: list | None, solutions: list[BaselineSolution]) -> list[dict]:
    """
    Build per-run diagnostics for continuous Q=1 fixed POS segments.

    run_label is inferred from ParsedPos.path.stem, because RTKLIB output POS files
    are written as <run_label>.pos.
    """
    if not parsed_pos_items:
        return []

    solution_by_run = {s.run_label: s for s in solutions}
    records = []

    for parsed in parsed_pos_items:
        run_label = Path(parsed.path).stem
        solution = solution_by_run.get(run_label)
        solution_h = getattr(solution, "h_m", None) if solution is not None else None

        df = parsed.dataframe.copy()
        if df.empty or "Q" not in df.columns or "time" not in df.columns:
            continue

        fixed = df[df["Q"] == 1].copy()
        if fixed.empty:
            records.append({
                "run_label": run_label,
                "segment_id": None,
                "segment_start": None,
                "segment_end": None,
                "duration_min": None,
                "n_epochs": 0,
                "mean_h_m": None,
                "std_h_m": None,
                "delta_h_from_solution_mean_m": None,
                "note": "NO_Q1_FIXED_EPOCHS",
            })
            continue

        fixed["time"] = pd.to_datetime(fixed["time"])
        fixed = fixed.sort_values("time").reset_index(drop=True)

        time_diffs = fixed["time"].diff().dt.total_seconds().dropna()
        time_diffs = time_diffs[time_diffs > 0]
        median_dt_sec = float(time_diffs.median()) if not time_diffs.empty else None
        gap_threshold_sec = max(2.5 * median_dt_sec, 2.0) if median_dt_sec else 2.0

        gaps = fixed["time"].diff().dt.total_seconds().fillna(0.0)
        segment_start_idx = 0
        segments = []

        for i in range(1, len(fixed)):
            if float(gaps.iloc[i]) > gap_threshold_sec:
                segments.append((segment_start_idx, i - 1))
                segment_start_idx = i
        segments.append((segment_start_idx, len(fixed) - 1))

        for seg_id, (a, b) in enumerate(segments, start=1):
            seg = fixed.iloc[a:b+1].copy()
            start_time = seg["time"].iloc[0]
            end_time = seg["time"].iloc[-1]

            duration_sec = float((end_time - start_time).total_seconds())
            if median_dt_sec is not None and len(seg) > 1:
                duration_sec += median_dt_sec

            h_values = []
            for _, row in seg.iterrows():
                try:
                    _, _, h = ecef_to_geodetic(float(row["X"]), float(row["Y"]), float(row["Z"]))
                    h_values.append(h)
                except Exception:
                    pass

            h_series = pd.Series(h_values, dtype="float64")
            mean_h = float(h_series.mean()) if not h_series.empty else None
            std_h = float(h_series.std(ddof=1)) if len(h_series) >= 2 else None
            delta_h = (mean_h - solution_h) if (mean_h is not None and solution_h is not None) else None

            records.append({
                "run_label": run_label,
                "segment_id": seg_id,
                "segment_start": start_time,
                "segment_end": end_time,
                "duration_min": duration_sec / 60.0,
                "n_epochs": int(len(seg)),
                "mean_h_m": mean_h,
                "std_h_m": std_h,
                "delta_h_from_solution_mean_m": delta_h,
                "note": "",
            })

    return records



def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _rss2(a, b):
    a = _safe_float(a)
    b = _safe_float(b)

    if a is None and b is None:
        return None
    if a is None:
        a = 0.0
    if b is None:
        b = 0.0

    return math.sqrt(a*a + b*b)


def _cors_enu_sigmas_from_xyz(cors: CorsSolution) -> tuple[float | None, float | None, float | None]:
    """
    Approximate CORS E/N/U standard deviations from diagonal XYZ standard deviations.

    Assumption:
    - CORS XYZ covariance terms are unavailable;
    - std_X_m, std_Y_m, std_Z_m are treated as uncorrelated.
    """
    sx = _safe_float(cors.std_X_m)
    sy = _safe_float(cors.std_Y_m)
    sz = _safe_float(cors.std_Z_m)

    if sx is None and sy is None and sz is None:
        return None, None, None

    sx = 0.0 if sx is None else sx
    sy = 0.0 if sy is None else sy
    sz = 0.0 if sz is None else sz

    lat_rad, lon_rad, _ = ecef_to_geodetic(cors.X_m, cors.Y_m, cors.Z_m)
    lat = float(lat_rad)
    lon = float(lon_rad)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # ENU unit vectors in ECEF.
    e = (-sin_lon, cos_lon, 0.0)
    n = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    u = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)

    def proj_sigma(v):
        return math.sqrt(
            (v[0] * sx) ** 2 +
            (v[1] * sy) ** 2 +
            (v[2] * sz) ** 2
        )

    return proj_sigma(e), proj_sigma(n), proj_sigma(u)


def _find_pair_for_solution(solution: BaselineSolution, pairs: list[BaselinePair]) -> BaselinePair | None:
    for pair in pairs:
        if pair.run_label == solution.run_label:
            return pair
    return None


def _enu_offset_to_ecef(e_m: float, n_m: float, u_m: float, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """
    Convert a local ENU offset vector to ECEF dX,dY,dZ.

    RINEX ANTENNA: DELTA H/E/N convention:
    - H = Up offset from marker/BM to antenna
    - E = East offset from marker/BM to antenna
    - N = North offset from marker/BM to antenna

    Therefore:
        antenna_XYZ = BM_XYZ + ENU_to_ECEF(E, N, H)
        BM_XYZ      = antenna_XYZ - ENU_to_ECEF(E, N, H)
    """
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    dx = -sin_lon * e_m - sin_lat * cos_lon * n_m + cos_lat * cos_lon * u_m
    dy =  cos_lon * e_m - sin_lat * sin_lon * n_m + cos_lat * sin_lon * u_m
    dz =  cos_lat * n_m + sin_lat * u_m

    return dx, dy, dz



def _final_gnssbm_solution_records(
    cors: CorsSolution,
    pairs: list[BaselinePair],
    solutions: list[BaselineSolution],
) -> list[dict]:
    """
    Build final GNSSBM coordinate solution records.

    RTKLIB solution strategy remains antenna-to-antenna.

    This report-level transformation applies rover RINEX ANTENNA: DELTA H/E/N
    to convert the solved antenna coordinate to the benchmark/marker coordinate:

        BM_XYZ = antenna_XYZ - ENU_to_ECEF(E, N, H)

    Current uncertainty propagation:
    - RTKLIB per-run scatter + CORS PPP repeatability by RSS;
    - antenna-height/eccentricity measurement uncertainty is not yet included.
    """
    cors_std_e, cors_std_n, cors_std_u = _cors_enu_sigmas_from_xyz(cors)

    records = []

    for s in solutions:
        pair = _find_pair_for_solution(s, pairs)

        antenna_delta_h_m = None
        antenna_delta_e_m = None
        antenna_delta_n_m = None
        antenna_offset_source = "MISSING"
        rover_file = s.rover_file

        if pair is not None:
            rover_file = pair.rover.path
            antenna_delta_h_m = _safe_float(pair.rover.antenna_delta_h_m)
            antenna_delta_e_m = _safe_float(pair.rover.antenna_delta_e_m)
            antenna_delta_n_m = _safe_float(pair.rover.antenna_delta_n_m)

            if antenna_delta_h_m is not None:
                antenna_offset_source = "RINEX_ANTENNA_DELTA_H/E/N"

        antenna_X = _safe_float(s.X_m)
        antenna_Y = _safe_float(s.Y_m)
        antenna_Z = _safe_float(s.Z_m)
        antenna_lat = _safe_float(s.lat_deg)
        antenna_lon = _safe_float(s.lon_deg)

        bm_X = antenna_X
        bm_Y = antenna_Y
        bm_Z = antenna_Z
        bm_lat_deg = _safe_float(s.lat_deg)
        bm_lon_deg = _safe_float(s.lon_deg)
        bm_h_m = _safe_float(s.h_m)

        if (
            antenna_X is not None and
            antenna_Y is not None and
            antenna_Z is not None and
            antenna_lat is not None and
            antenna_lon is not None and
            antenna_delta_h_m is not None
        ):
            e_m = 0.0 if antenna_delta_e_m is None else antenna_delta_e_m
            n_m = 0.0 if antenna_delta_n_m is None else antenna_delta_n_m
            u_m = antenna_delta_h_m

            dX, dY, dZ = _enu_offset_to_ecef(
                e_m=e_m,
                n_m=n_m,
                u_m=u_m,
                lat_deg=antenna_lat,
                lon_deg=antenna_lon,
            )

            bm_X = antenna_X - dX
            bm_Y = antenna_Y - dY
            bm_Z = antenna_Z - dZ

            bm_lat_rad, bm_lon_rad, bm_h_m = ecef_to_geodetic(bm_X, bm_Y, bm_Z)
            bm_lat_deg = math.degrees(bm_lat_rad)
            bm_lon_deg = math.degrees(bm_lon_rad)

        records.append({
            "run_label": s.run_label,
            "benchmark_id": s.benchmark_id,
            "rover_file": rover_file,

            "antenna_delta_h_m": _fmt(antenna_delta_h_m),
            "antenna_delta_e_m": _fmt(antenna_delta_e_m),
            "antenna_delta_n_m": _fmt(antenna_delta_n_m),
            "antenna_offset_source": antenna_offset_source,

            "X_m": _fmt(bm_X),
            "Y_m": _fmt(bm_Y),
            "Z_m": _fmt(bm_Z),
            "lon_deg": _fmt(bm_lon_deg, 10),
            "lat_deg": _fmt(bm_lat_deg, 10),
            "h_m": _fmt(bm_h_m, 4),

            "std_X_m": _fmt(_rss2(s.std_X_m, cors.std_X_m)),
            "std_Y_m": _fmt(_rss2(s.std_Y_m, cors.std_Y_m)),
            "std_Z_m": _fmt(_rss2(s.std_Z_m, cors.std_Z_m)),
            "std_lon_m": _fmt(_rss2(s.std_lon_m, cors_std_e)),
            "std_lat_m": _fmt(_rss2(s.std_lat_m, cors_std_n)),
            "std_h_m": _fmt(_rss2(s.std_h_m, cors_std_u)),

            "rtklib_X_antenna_m": _fmt(s.X_m),
            "rtklib_Y_antenna_m": _fmt(s.Y_m),
            "rtklib_Z_antenna_m": _fmt(s.Z_m),
            "rtklib_lon_antenna_deg": _fmt(s.lon_deg, 10),
            "rtklib_lat_antenna_deg": _fmt(s.lat_deg, 10),
            "rtklib_h_antenna_m": _fmt(s.h_m, 4),

            "rtklib_std_h_m": _fmt(s.std_h_m),
            "cors_std_up_m": _fmt(cors_std_u),
            "uncertainty_note": "RSS of RTKLIB scatter and CORS PPP repeatability; antenna H/E/N measurement uncertainty not yet included.",
            "qc_flags": "; ".join(s.qc_flags),
        })

    return records



def build_report(
    inputs: UserInputs,
    cors: CorsSolution,
    rover_inventory: list[RinexObsFile],
    base_inventory: list[RinexObsFile],
    pairs: list[BaselinePair],
    products: list[ResolvedProducts],
    run_results: list[RunResult],
    solutions: list[BaselineSolution],
    parsed_pos_items: list | None = None,
) -> Path:
    output_path = Path(inputs.output_root) / inputs.report_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; line-height: 1.42; color: #222; }
    h1 { font-size: 24px; border-bottom: 2px solid #444; padding-bottom: 8px; }
    h2 { font-size: 19px; margin-top: 28px; border-bottom: 1px solid #aaa; padding-bottom: 4px; }
    h3 { font-size: 16px; margin-top: 18px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; font-size: 13px; }
    th, td { border: 1px solid #ccc; padding: 5px 7px; vertical-align: top; }
    th { background: #f0f0f0; text-align: left; }
    code { background: #f4f4f4; padding: 1px 4px; }
    pre { background: #f6f6f6; padding: 10px; overflow-x: auto; }
    img.plot { max-width: 100%; border: 1px solid #ccc; margin: 6px 0 18px 0; }
    .small { font-size: 12px; color: #555; }
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>RTKLIB Baseline Orchestrator - Baseline Solution Report</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>RTKLIB Baseline Orchestrator - Baseline Solution Report</h1>",
    ]

    html_parts.append("<h2>1. Report header</h2>")
    html_parts.append(_html_table_from_records([{
        "Project name": inputs.project_name,
        "Report path": output_path,
        "RTK_process path": inputs.output_root,
        "Software": "RTKLIB Baseline Orchestrator",
    }]))

    html_parts.append("<h2>2. Processing strategy</h2>")
    html_parts.append("<pre>CORS GINAN PPP report\n→ rover/base RINEX discovery\n→ RINEX header time-span extraction\n→ rover–base overlap matching\n→ product/model resolution/download\n→ RTKLIB rnx2rtkp processing\n→ POS parsing\n→ fixed/float QC\n→ GNSSBM coordinate solution\n→ HTML report</pre>")

    html_parts.append("<h2>3. User inputs and paths</h2>")
    html_parts.append(_html_table_from_records([{
        "CORS report": inputs.cors_solution_report_path,
        "Rover RINEX root": inputs.rover_rinex_root,
        "Base RINEX root": inputs.base_rinex_root,
        "Products root": inputs.products_root,
        "Output root": inputs.output_root,
        "rnx2rtkp": inputs.rnx2rtkp_path,
        "Product family": f"{inputs.product_provider}/{inputs.product_series}/{inputs.product_project}",
        "Product mode": inputs.product_mode,
        "Processing mode": inputs.processing_mode,
        "Minimum overlap minutes": inputs.minimum_overlap_minutes,
        "Matching strategy": inputs.matching_strategy,
        "Final window minutes": inputs.final_window_minutes,
    }]))

    html_parts.append("<h2>4. CORS PPP reference solution from GINAN report</h2>")
    html_parts.append(_html_table_from_records([{
        "station_id": cors.station_id,
        "X_m": _fmt(cors.X_m),
        "Y_m": _fmt(cors.Y_m),
        "Z_m": _fmt(cors.Z_m),
        "std_X_m": _fmt(cors.std_X_m),
        "std_Y_m": _fmt(cors.std_Y_m),
        "std_Z_m": _fmt(cors.std_Z_m),
        "n_solutions": cors.n_solutions,
        "source_report_path": cors.source_report_path,
    }]))

    html_parts.append("<h2>5. CORS daily/per-file PPP solution summary</h2>")
    if cors.daily_table is not None:
        html_parts.append(cors.daily_table.to_html(index=False, escape=True, border=0))
    else:
        html_parts.append("<p>No CORS daily/per-file solution table was found.</p>")

    html_parts.append(_cors_ppp_reference_epoch_note(cors))

    html_parts.append("<h2>6. RINEX inventory</h2>")
    html_parts.append("<h3>Rover RINEX inventory</h3>")
    html_parts.append(_html_table_from_records([{
        "file": x.path,
        "marker_name": x.marker_name,
        "rinex_version": x.rinex_version,
        "first_obs": x.first_obs,
        "last_obs": x.last_obs,
        "duration_minutes": _fmt(_rinex_duration_minutes(x), 4),
        "interval_sec": x.interval_sec,
        "receiver": x.receiver,
        "antenna": x.antenna,
    } for x in rover_inventory]))
    html_parts.append("<h3>CORS/base RINEX inventory</h3>")
    html_parts.append(_html_table_from_records([{
        "file": x.path,
        "marker_name": x.marker_name,
        "rinex_version": x.rinex_version,
        "first_obs": x.first_obs,
        "last_obs": x.last_obs,
        "duration_minutes": _fmt(_rinex_duration_minutes(x), 4),
        "interval_sec": x.interval_sec,
        "receiver": x.receiver,
        "antenna": x.antenna,
    } for x in base_inventory]))

    html_parts.append("<h2>7. Rover–base overlap matching</h2>")
    html_parts.append(_html_table_from_records([{
        "run_label": p.run_label,
        "rover_file": p.rover.path,
        "base_file": p.base.path,
        "overlap_start": p.overlap_start,
        "overlap_end": p.overlap_end,
        "overlap_minutes": round(p.overlap_minutes, 4),
        "matching_status": p.matching_status,
    } for p in pairs]))

    html_parts.append("<h2>8. Product and model resolution</h2>")
    html_parts.append(_html_table_from_records([{
        "run_label": p.run_label,
        "nav_files": "; ".join(map(str, p.nav_files)),
        "sp3_files": "; ".join(map(str, p.sp3_files)),
        "clk_files": "; ".join(map(str, p.clk_files)),
        "ionex_files": "; ".join(map(str, p.ionex_files)),
        "antex_file": p.antex_file,
        "blq_file": p.blq_file,
        "bia_files": "; ".join(map(str, p.bia_files)),
        "missing_files": "; ".join(p.missing_files),
        "product_status": p.product_status,
    } for p in products]))

    html_parts.append("<h2>9. RTKLIB processing options</h2>")
    html_parts.append(_html_table_from_records([{
        "processing_mode": inputs.processing_mode,
        "frequency_mode": inputs.frequency_mode,
        "solution_type": inputs.solution_type,
        "elevation_mask_deg": inputs.elevation_mask_deg,
        "ambiguity_mode": inputs.ambiguity_mode,
        "ambiguity_threshold": inputs.ambiguity_threshold,
        "nav_systems": ",".join(inputs.nav_systems),
        "output_coordinate_format": inputs.output_coordinate_format,
        "trace_level": inputs.trace_level,
    }]))

    html_parts.append("<h2>10. Execution summary</h2>")
    html_parts.append(_html_table_from_records([{
        "run_label": r.run_label,
        "status": r.status,
        "exit_code": r.exit_code,
        "processing_duration_sec": r.processing_duration_sec,
        "output_pos_path": r.output_pos_path,
        "stdout_path": r.stdout_path,
        "stderr_path": r.stderr_path,
        "warnings": "; ".join(r.warnings),
        "errors": "; ".join(r.errors),
    } for r in run_results]))

    html_parts.append("<h2>11. Per-run RTKLIB POS/QC summary</h2>")
    html_parts.append("<p>Detailed POS/QC summary is stored in run QC files and final solution tables.</p>")

    html_parts.append("<h2>12. Per-run baseline solution</h2>")
    html_parts.append(_html_table_from_records([{
        "run_label": s.run_label,
        "benchmark_id": s.benchmark_id,
        "solution_method": s.solution_method,
        "final_window_minutes": s.final_window_minutes,
        "X_m": _fmt(s.X_m),
        "Y_m": _fmt(s.Y_m),
        "Z_m": _fmt(s.Z_m),
        "lon_deg": _fmt(s.lon_deg, 10),
        "lat_deg": _fmt(s.lat_deg, 10),
        "h_m": _fmt(s.h_m, 4),
        "std_X_m": _fmt(s.std_X_m),
        "std_Y_m": _fmt(s.std_Y_m),
        "std_Z_m": _fmt(s.std_Z_m),
        "std_lon_m": _fmt(s.std_lon_m),
        "std_lat_m": _fmt(s.std_lat_m),
        "std_h_m": _fmt(s.std_h_m),
        "baseline_E_m": _fmt(s.baseline_E_m),
        "baseline_N_m": _fmt(s.baseline_N_m),
        "baseline_U_m": _fmt(s.baseline_U_m),
        "baseline_length_m": _fmt(s.baseline_length_m),
        "q1_fixed_percent": _fmt(s.q1_fixed_percent, 2),
        "n_fixed_epochs_used": s.n_fixed_epochs_used,
        "fixed_time_start": s.fixed_time_start,
        "fixed_time_end": s.fixed_time_end,
        "fixed_total_duration_min": _fmt(s.fixed_total_duration_min, 3),
        "longest_fixed_segment_start": s.longest_fixed_segment_start,
        "longest_fixed_segment_end": s.longest_fixed_segment_end,
        "longest_fixed_segment_duration_min": _fmt(s.longest_fixed_segment_duration_min, 3),
        "longest_fixed_segment_epochs": s.longest_fixed_segment_epochs,
        "ratio_mean": _fmt(s.ratio_mean, 2),
        "qc_flags": "; ".join(s.qc_flags),
    } for s in solutions]))

    html_parts.append("<h2>13. Q=1 fixed segment diagnostics</h2>")
    segment_records = _fixed_segment_diagnostic_records(parsed_pos_items or [], solutions)
    html_parts.append(_html_table_from_records([{
        "run_label": r["run_label"],
        "segment_id": r["segment_id"],
        "segment_start": r["segment_start"],
        "segment_end": r["segment_end"],
        "duration_min": _fmt(r["duration_min"], 3),
        "n_epochs": r["n_epochs"],
        "mean_h_m": _fmt(r["mean_h_m"], 4),
        "std_h_m": _fmt(r["std_h_m"], 4),
        "delta_h_from_solution_mean_m": _fmt(r["delta_h_from_solution_mean_m"], 4),
        "note": r["note"],
    } for r in segment_records]))

    html_parts.append("<h2>14. Final GNSSBM coordinate solutions</h2>")
    html_parts.append("<p>Final per-run GNSSBM coordinate solutions. The RTKLIB solution remains antenna-to-antenna. At report level, rover RINEX ANTENNA: DELTA H/E/N is applied as a full local ENU correction to convert antenna XYZ/lon/lat/h to BM XYZ/lon/lat/h. Coordinate standard deviations include RTKLIB per-run scatter and CORS PPP repeatability by RSS propagation. Antenna H/E/N measurement uncertainty is not yet included.</p>")
    html_parts.append(_html_table_from_records(_final_gnssbm_solution_records(
        cors=cors,
        pairs=pairs,
        solutions=solutions,
    )))

    html_parts.append("<h2>15. Warnings and rejected runs</h2>")
    warning_records = []
    for r in run_results:
        for w in r.warnings:
            warning_records.append({"run_label": r.run_label, "severity": "WARNING", "message": w})
        for e in r.errors:
            warning_records.append({"run_label": r.run_label, "severity": "ERROR", "message": e})
    for s in solutions:
        for flag in s.qc_flags:
            warning_records.append({"run_label": s.run_label, "severity": "QC", "message": flag})
    html_parts.append(_html_table_from_records(warning_records))

    html_parts.append("<h2>16. Plots</h2>")
    if inputs.generate_plots:
        if inputs.processing_mode == "static":
            html_parts.append("<h3>Static solution map</h3>")
            html_parts.append(_static_solution_map(cors, solutions))
        elif inputs.processing_mode == "dynamic":
            html_parts.append("<h3>Dynamic trajectory map</h3>")
            html_parts.append(_dynamic_trajectory_map(cors, parsed_pos_items or []))
    else:
        html_parts.append("<p>Plot generation disabled.</p>")

    html_parts.append("<h2>17. Reproducibility appendix</h2>")
    html_parts.append(_html_table_from_records([{
        "run_label": r.run_label,
        "command_path": r.command_path,
        "output_pos_path": r.output_pos_path,
        "stdout_path": r.stdout_path,
        "stderr_path": r.stderr_path,
    } for r in run_results]))

    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path
