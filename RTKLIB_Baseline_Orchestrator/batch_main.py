from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import paths_config
from models import UserInputs
from system_check import run_system_check
from cors_report_parser import parse_cors_report
from rinex_header import discover_rinex_obs
from overlap_matcher import match_rover_base_overlaps
from products_resolver import resolve_products_for_pair
from rtklib_conf_builder import build_run_config
from rnx2rtkp_runner import run_rnx2rtkp
from pos_parser import parse_pos
from baseline_qc import compute_baseline_solution, build_pos_qc_table
from report_builder import build_report


def _bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _duration_minutes(item: Any) -> float | None:
    first_obs = getattr(item, "first_obs", None)
    last_obs = getattr(item, "last_obs", None)

    if first_obs is None or last_obs is None:
        return None

    return (last_obs - first_obs).total_seconds() / 60.0


def parse_args():
    p = argparse.ArgumentParser(description="RTKLIB Baseline Orchestrator")

    p.add_argument("--project-name", default="")
    p.add_argument("--cors-report", required=True)
    p.add_argument("--rover-root", required=True)
    p.add_argument("--base-root", required=True)
    p.add_argument("--products-root", required=True)
    p.add_argument("--output-root", default="")
    p.add_argument("--rnx2rtkp", default=paths_config.get_default_rnx2rtkp_path())

    p.add_argument("--provider", default=paths_config.DEFAULTS["product_provider"], choices=paths_config.PRODUCT_PROVIDERS)
    p.add_argument("--series", default=paths_config.DEFAULTS["product_series"], choices=paths_config.PRODUCT_SERIES)
    p.add_argument("--project", default=paths_config.DEFAULTS["product_project"], choices=paths_config.PRODUCT_PROJECTS)
    p.add_argument("--product-mode", default=paths_config.DEFAULTS["product_mode"], choices=paths_config.PRODUCT_MODES)

    p.add_argument("--download-missing", default="y")
    p.add_argument("--downloader-script", default=paths_config.get_default_downloader_script_path())
    p.add_argument("--downloader-python", default=paths_config.get_default_downloader_python_path())
    p.add_argument("--use-ionex", default="n")
    p.add_argument("--use-antex", default="y")
    p.add_argument("--use-blq", default="n")
    p.add_argument("--use-bia-osb", default="n")

    p.add_argument("--processing-mode", default="static", choices=paths_config.PROCESSING_MODES)
    p.add_argument("--min-overlap-min", type=float, default=paths_config.DEFAULTS["minimum_overlap_minutes"])
    p.add_argument("--matching-strategy", default="best_overlap_per_rover", choices=["best_overlap_per_rover", "all_valid_overlaps"])
    p.add_argument("--overwrite", default="n")

    p.add_argument("--frequencies", default=paths_config.DEFAULTS["frequency_mode"], choices=paths_config.FREQUENCY_MODES)
    p.add_argument("--el-mask", type=float, default=paths_config.DEFAULTS["elevation_mask_deg"])
    p.add_argument("--solution-type", default=paths_config.DEFAULTS["solution_type"], choices=paths_config.SOLUTION_TYPES)
    p.add_argument("--ar-mode", default=paths_config.DEFAULTS["ambiguity_mode"], choices=paths_config.AR_MODES)
    p.add_argument("--ar-threshold", type=float, default=paths_config.DEFAULTS["ambiguity_threshold"])
    p.add_argument("--nav-systems", default="G,E,C")
    p.add_argument("--output-format", default=paths_config.DEFAULTS["output_coordinate_format"], choices=paths_config.OUTPUT_FORMATS)

    p.add_argument("--final-window-min", type=float, default=paths_config.DEFAULTS["final_window_minutes"])
    p.add_argument("--q1-only-final", default="y")
    p.add_argument("--min-fixed-percent", type=float, default=paths_config.DEFAULTS["min_fixed_percent"])
    p.add_argument("--min-ratio", type=float, default=paths_config.DEFAULTS["min_ratio_for_fixed"])
    p.add_argument("--generate-plots", default="y")

    p.add_argument("--execution-mode", default=paths_config.DEFAULTS["execution_mode"], choices=paths_config.EXECUTION_MODES)
    p.add_argument("--generate-report", default="y")
    p.add_argument("--report-filename", default=paths_config.REPORT_FILENAME)
    p.add_argument("--trace-level", type=int, default=0)

    return p.parse_args()


def build_inputs(args) -> UserInputs:
    rover_root = Path(args.rover_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else paths_config.output_root_from_rover_root(rover_root)

    return UserInputs(
        project_name=args.project_name,
        cors_solution_report_path=Path(args.cors_report).expanduser().resolve(),
        rover_rinex_root=rover_root,
        base_rinex_root=Path(args.base_root).expanduser().resolve(),
        products_root=Path(args.products_root).expanduser().resolve(),
        output_root=output_root,
        rnx2rtkp_path=Path(args.rnx2rtkp).expanduser().resolve(),
        product_provider=args.provider,
        product_series=args.series,
        product_project=args.project,
        product_mode=args.product_mode,
        download_missing_products=_bool_text(args.download_missing),
        downloader_script_path=Path(args.downloader_script).expanduser() if args.downloader_script else None,
        downloader_python_path=Path(args.downloader_python).expanduser() if args.downloader_python else None,
        use_ionex=_bool_text(args.use_ionex),
        use_antex=_bool_text(args.use_antex),
        use_blq=_bool_text(args.use_blq),
        use_bia_osb=_bool_text(args.use_bia_osb),
        processing_mode=args.processing_mode,
        minimum_overlap_minutes=args.min_overlap_min,
        matching_strategy=args.matching_strategy,
        overwrite_existing_outputs=_bool_text(args.overwrite),
        frequency_mode=args.frequencies,
        elevation_mask_deg=args.el_mask,
        solution_type=args.solution_type,
        ambiguity_mode=args.ar_mode,
        ambiguity_threshold=args.ar_threshold,
        nav_systems=[s.strip() for s in args.nav_systems.split(",") if s.strip()],
        output_coordinate_format=args.output_format,
        final_window_minutes=args.final_window_min,
        q_fixed_only_for_final=_bool_text(args.q1_only_final),
        min_fixed_percent=args.min_fixed_percent,
        min_ratio_for_fixed=args.min_ratio,
        generate_plots=_bool_text(args.generate_plots),
        execution_mode=args.execution_mode,
        generate_report=_bool_text(args.generate_report),
        report_filename=args.report_filename,
        trace_level=args.trace_level,
    )


def write_inventory_csv(path: Path, items) -> None:
    records = []

    for x in items:
        records.append({
            "path": str(getattr(x, "path", "")),
            "filename": getattr(x, "filename", ""),
            "marker_name": getattr(x, "marker_name", ""),
            "rinex_version": getattr(x, "rinex_version", ""),
            "first_obs": getattr(x, "first_obs", None),
            "last_obs": getattr(x, "last_obs", None),
            "duration_minutes": _duration_minutes(x),
            "interval_sec": getattr(x, "interval_sec", None),
            "receiver": getattr(x, "receiver", ""),
            "antenna": getattr(x, "antenna", ""),
            "antenna_delta_h_m": getattr(x, "antenna_delta_h_m", None),
            "antenna_delta_e_m": getattr(x, "antenna_delta_e_m", None),
            "antenna_delta_n_m": getattr(x, "antenna_delta_n_m", None),
            "warnings": "; ".join(getattr(x, "warnings", []) or []),
        })

    pd.DataFrame(records).to_csv(path, index=False)


def run_batch(inputs: UserInputs, print_fn: Callable[[str], None] = print) -> dict[str, Any]:
    if inputs.output_root is None:
        inputs.output_root = paths_config.output_root_from_rover_root(Path(inputs.rover_rinex_root))

    inputs.output_root = Path(inputs.output_root)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    (inputs.output_root / "runs").mkdir(exist_ok=True)
    (inputs.output_root / "logs").mkdir(exist_ok=True)

    print_fn("===== RTKLIB Baseline Orchestrator =====")
    print_fn(f"Output root: {inputs.output_root}")

    check = run_system_check(inputs)

    for warning in check["warnings"]:
        if str(warning).strip():
            print_fn(f"WARNING: {warning}")

    if not check["ok"]:
        for error in check["errors"]:
            if str(error).strip():
                print_fn(f"ERROR: {error}")
        return {
            "ok": False,
            "stage": "system_check",
            "check": check,
            "errors": check["errors"],
            "warnings": check["warnings"],
        }

    print_fn("Parsing CORS GINAN report...")
    cors = parse_cors_report(inputs.cors_solution_report_path)
    print_fn(f"CORS station: {cors.station_id} X={cors.X_m:.4f} Y={cors.Y_m:.4f} Z={cors.Z_m:.4f}")

    print_fn("Discovering rover RINEX files...")
    rover_inventory = discover_rinex_obs(inputs.rover_rinex_root)
    print_fn(f"Rover files: {len(rover_inventory)}")

    print_fn("Discovering CORS/base RINEX files...")
    base_inventory = discover_rinex_obs(inputs.base_rinex_root)
    print_fn(f"Base files: {len(base_inventory)}")

    write_inventory_csv(inputs.output_root / "rover_inventory.csv", rover_inventory)
    write_inventory_csv(inputs.output_root / "base_inventory.csv", base_inventory)

    pairs = match_rover_base_overlaps(
        rover_files=rover_inventory,
        base_files=base_inventory,
        minimum_overlap_minutes=inputs.minimum_overlap_minutes,
        matching_strategy=inputs.matching_strategy,
    )

    print_fn(f"Accepted baseline pairs: {len(pairs)}")

    pd.DataFrame([{
        "run_label": p.run_label,
        "rover_file": str(p.rover.path),
        "base_file": str(p.base.path),
        "overlap_start": p.overlap_start,
        "overlap_end": p.overlap_end,
        "overlap_minutes": p.overlap_minutes,
    } for p in pairs]).to_csv(inputs.output_root / "overlap_matches.csv", index=False)

    all_products = []
    run_results = []
    solutions = []
    parsed_pos_items = []

    for pair in pairs:
        print_fn(f"\n===== RUN {pair.run_label} =====")

        products = resolve_products_for_pair(pair=pair, inputs=inputs)
        all_products.append(products)
        print_fn(f"Product status: {products.product_status}")

        if products.product_status == "MISSING_REQUIRED":
            print_fn(f"ERROR: required products missing: {products.missing_files}")
            continue

        run_config = build_run_config(
            inputs=inputs,
            cors=cors,
            pair=pair,
            products=products,
        )

        result = run_rnx2rtkp(
            run_config=run_config,
            inputs=inputs,
        )

        run_results.append(result)
        print_fn(f"Run status: {result.status} exit: {result.exit_code}")

        if not result.output_pos_path or not result.output_pos_path.exists():
            continue

        if result.output_pos_path.stat().st_size <= 0:
            continue

        parsed = parse_pos(result.output_pos_path)
        parsed_pos_items.append(parsed)

        qc = build_pos_qc_table(parsed)
        pd.DataFrame([qc]).to_csv(run_config.run_dir / f"{pair.run_label}.qc.csv", index=False)

        if len(parsed.dataframe) == 0:
            print_fn("WARNING: POS file has no solution rows.")
            continue

        solution = compute_baseline_solution(
            parsed=parsed,
            pair=pair,
            cors=cors,
            inputs=inputs,
        )

        solutions.append(solution)
        print_fn(f"Q1 fixed percent: {solution.q1_fixed_percent}")

    pd.DataFrame([{
        "run_label": p.run_label,
        "nav_files": "; ".join(map(str, p.nav_files)),
        "sp3_files": "; ".join(map(str, p.sp3_files)),
        "clk_files": "; ".join(map(str, p.clk_files)),
        "ionex_files": "; ".join(map(str, p.ionex_files)),
        "missing_files": "; ".join(p.missing_files),
        "product_status": p.product_status,
    } for p in all_products]).to_csv(inputs.output_root / "products_inventory.csv", index=False)

    pd.DataFrame([{
        "run_label": r.run_label,
        "status": r.status,
        "exit_code": r.exit_code,
        "output_pos_path": str(r.output_pos_path) if r.output_pos_path else "",
        "stdout_path": str(r.stdout_path) if r.stdout_path else "",
        "stderr_path": str(r.stderr_path) if r.stderr_path else "",
        "warnings": "; ".join(r.warnings),
        "errors": "; ".join(r.errors),
    } for r in run_results]).to_csv(inputs.output_root / "run_index.csv", index=False)

    pd.DataFrame([{
        "run_label": s.run_label,
        "benchmark_id": s.benchmark_id,
        "X_m": s.X_m,
        "Y_m": s.Y_m,
        "Z_m": s.Z_m,
        "lon_deg": s.lon_deg,
        "lat_deg": s.lat_deg,
        "h_m": s.h_m,
        "std_X_m": s.std_X_m,
        "std_Y_m": s.std_Y_m,
        "std_Z_m": s.std_Z_m,
        "std_lon_m": s.std_lon_m,
        "std_lat_m": s.std_lat_m,
        "std_h_m": s.std_h_m,
        "baseline_E_m": s.baseline_E_m,
        "baseline_N_m": s.baseline_N_m,
        "baseline_U_m": s.baseline_U_m,
        "baseline_length_m": s.baseline_length_m,
        "q1_fixed_percent": s.q1_fixed_percent,
        "n_fixed_epochs_used": s.n_fixed_epochs_used,
        "fixed_time_start": s.fixed_time_start,
        "fixed_time_end": s.fixed_time_end,
        "fixed_total_duration_min": s.fixed_total_duration_min,
        "longest_fixed_segment_start": s.longest_fixed_segment_start,
        "longest_fixed_segment_end": s.longest_fixed_segment_end,
        "longest_fixed_segment_duration_min": s.longest_fixed_segment_duration_min,
        "longest_fixed_segment_epochs": s.longest_fixed_segment_epochs,
        "ratio_mean": s.ratio_mean,
        "qc_flags": "; ".join(s.qc_flags),
    } for s in solutions]).to_csv(inputs.output_root / "final_baseline_solutions.csv", index=False)

    report_path = None
    if inputs.generate_report:
        report_path = build_report(
            inputs=inputs,
            cors=cors,
            rover_inventory=rover_inventory,
            base_inventory=base_inventory,
            pairs=pairs,
            products=all_products,
            run_results=run_results,
            solutions=solutions,
            parsed_pos_items=parsed_pos_items,
        )
        print_fn(f"Report written: {report_path}")

    return {
        "ok": True,
        "check": check,
        "cors": cors,
        "rover_inventory": rover_inventory,
        "base_inventory": base_inventory,
        "pairs": pairs,
        "products": all_products,
        "run_results": run_results,
        "solutions": solutions,
        "parsed_pos_items": parsed_pos_items,
        "report_path": report_path,
    }


def main() -> int:
    args = parse_args()
    inputs = build_inputs(args)
    result = run_batch(inputs)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
