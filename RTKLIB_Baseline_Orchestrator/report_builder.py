
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
        "duration_minutes": x.duration_minutes,
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
        "duration_minutes": x.duration_minutes,
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
        "ratio_mean": _fmt(s.ratio_mean, 2),
        "qc_flags": "; ".join(s.qc_flags),
    } for s in solutions]))

    html_parts.append("<h2>13. Final GNSSBM coordinate solutions</h2>")
    html_parts.append("<p>v0.1 reports per-run GNSSBM coordinate solutions. Benchmark-level weighted combination is a later extension.</p>")

    html_parts.append("<h2>14. Warnings and rejected runs</h2>")
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

    html_parts.append("<h2>15. Plots</h2>")
    if inputs.generate_plots:
        if inputs.processing_mode == "static":
            html_parts.append("<h3>Static solution map</h3>")
            html_parts.append(_static_solution_map(cors, solutions))
        elif inputs.processing_mode == "dynamic":
            html_parts.append("<h3>Dynamic trajectory map</h3>")
            html_parts.append(_dynamic_trajectory_map(cors, parsed_pos_items or []))
    else:
        html_parts.append("<p>Plot generation disabled.</p>")

    html_parts.append("<h2>16. Reproducibility appendix</h2>")
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
