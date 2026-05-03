from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import paths_config
import system_check
import rinex_header
import resample_rinex
import products_download
import yaml_builder
import pea_runner
import results_check
import position_timeseries
import timeseries_report


VALID_REPORT_PLOT_COLUMNS = {"X", "Y", "Z", "lon", "lat", "h"}


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _prompt_str(prompt_text: str, default: str | None = None, allow_empty: bool = False) -> str:
    if default is None:
        raw = _safe_input(f"{prompt_text}: ").strip()
    else:
        raw = _safe_input(f"{prompt_text} [{default}]: ").strip()

    if raw == "":
        if default is not None:
            return default
        if allow_empty:
            return ""
        raise ValueError(f"Input required: {prompt_text}")

    return raw


def _prompt_int(prompt_text: str, default: int | None = None, allow_empty: bool = False) -> int | None:
    if default is None:
        raw = _safe_input(f"{prompt_text}: ").strip()
    else:
        raw = _safe_input(f"{prompt_text} [{default}]: ").strip()

    if raw == "":
        if default is not None:
            return default
        if allow_empty:
            return None
        raise ValueError(f"Input required: {prompt_text}")

    return int(raw)


def _prompt_bool(prompt_text: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    raw = _safe_input(f"{prompt_text} [y/n, default={default_str}]: ").strip().lower()

    if raw == "":
        return default

    if raw in {"y", "yes", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False

    raise ValueError(f"Invalid boolean input for: {prompt_text}")


def _parse_report_plot_columns(raw: str) -> list[str]:
    default = ["X", "Y", "Z", "h"]

    if raw is None:
        return default

    raw = str(raw).strip()

    if raw == "":
        return default

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
    }

    all_columns = ["X", "Y", "Z", "lon", "lat", "h", "E", "N", "U"]

    parts = [p.strip() for p in raw.split(",") if p.strip()]

    if not parts:
        return default

    if any(p.lower() == "all" for p in parts):
        return all_columns

    selected = []
    for part in parts:
        key = part.lower()

        if key not in aliases:
            valid = "X,Y,Z,lon,lat,h,E,N,U,all"
            raise ValueError(f"Invalid report plot column '{part}'. Valid values: {valid}")

        canonical = aliases[key]

        if canonical not in selected:
            selected.append(canonical)

    return selected
def process_single_dataset(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    try:
        header_info = rinex_header.parse_rinex_header(ctx["raw"]["raw_rinex_file"])
        ctx = rinex_header.enrich_dataset_with_header(ctx, header_info)

        covered_dates, day_codes = rinex_header.derive_covered_dates(
            ctx["header"]["time_first_obs"],
            ctx["header"]["time_last_obs"],
        )
        ctx["time_window"]["start_epoch"] = ctx["header"]["time_first_obs"]
        ctx["time_window"]["end_epoch"] = ctx["header"]["time_last_obs"]
        ctx["time_window"]["covered_dates"] = covered_dates
        ctx["time_window"]["day_codes"] = day_codes

        ctx = resample_rinex.plan_resampling(config, ctx)
        ctx = resample_rinex.build_resampled_paths(config, ctx)

        resample_result = resample_rinex.run_resampling(config, ctx)
        ctx = resample_result["dataset_context"]
        if not resample_result["ok"]:
            raise RuntimeError(resample_result["message"])

        ctx = products_download.ensure_products_dir(config, ctx)

        download_plan = products_download.build_download_plan(config, ctx)
        download_results = products_download.run_download_plan(download_plan)

        failed_downloads = [r for r in download_results if not r["ok"]]
        if failed_downloads:
            first = failed_downloads[0]
            raise RuntimeError(
                f"Download step failed ({first['kind']} / {first['date']}) with return code {first['returncode']}"
            )

        ctx["execution"]["downloads_completed"] = True

        ctx = products_download.resolve_product_files(config, ctx)
        product_validation = products_download.validate_product_files(ctx)
        if not product_validation["ok"]:
            raise FileNotFoundError(
                "Missing product files: " + " | ".join(product_validation["missing"])
            )

        ctx = yaml_builder.build_output_paths(config, ctx)
        yaml_text = yaml_builder.render_yaml_from_template(config, ctx)
        write_result = yaml_builder.write_yaml_file(yaml_text, ctx["outputs"]["yaml_path"])
        if not write_result["ok"]:
            raise RuntimeError(write_result["message"])

        ctx["execution"]["yaml_written"] = True

        if config["user_inputs"]["execution_mode"] == "build_only":
            ctx["execution"]["status"] = "BUILT"
            ctx["execution"]["message"] = "YAML built successfully. PEA execution skipped."
            return ctx

        run_result = pea_runner.run_pea(config, ctx)
        ctx = run_result["dataset_context"]
        if not run_result["ok"]:
            raise RuntimeError(f"PEA failed with exit code {run_result['exit_code']}")

        early_result = pea_runner.detect_early_stop(ctx["outputs"]["stdout_path"])
        if config["validation"]["fail_on_epoch1_stop"] and early_result["early_stop"]:
            raise RuntimeError("Early stop detected at epoch #1")

        validation_result = results_check.validate_run_outputs(config, ctx)
        ctx = results_check.final_dataset_status(ctx, validation_result)

        if not validation_result["ok"]:
            raise RuntimeError(ctx["execution"]["message"])

        return ctx

    except Exception as e:
        ctx["execution"]["status"] = "FAILED"
        ctx["execution"]["message"] = str(e)
        ctx["execution"]["validation_passed"] = False
        return ctx


def process_batch(config: dict) -> list[dict]:
    raw_root = config["user_inputs"]["raw_root"]
    limit_datasets = config["user_inputs"]["limit_datasets"]

    dataset_contexts = rinex_header.discover_raw_datasets(raw_root, limit=limit_datasets)

    results = []
    for i, ctx in enumerate(dataset_contexts, start=1):
        dataset_name = ctx["identity"]["dataset_name"]
        print(f"[{i}/{len(dataset_contexts)}] Processing dataset: {dataset_name}")
        result_ctx = process_single_dataset(config, ctx)
        print(f"    status  : {result_ctx['execution']['status']}")
        print(f"    message : {result_ctx['execution']['message']}")
        results.append(result_ctx)

    return results


def print_final_summary(results: list[dict], config: dict) -> None:
    total = len(results)
    success = sum(1 for r in results if r["execution"]["status"] == "SUCCESS")
    built = sum(1 for r in results if r["execution"]["status"] == "BUILT")
    failed = total - success - built

    print("\nBatch summary")
    print("-------------")
    print(f"Execution mode : {config['user_inputs']['execution_mode']}")
    print(f"Total datasets : {total}")
    print(f"Success        : {success}")
    print(f"Built only     : {built}")
    print(f"Failed         : {failed}")

    print("\nPer-dataset status")
    print("------------------")
    for r in results:
        print(f"{r['identity']['dataset_name']}: {r['execution']['status']} -> {r['execution']['message']}")


def generate_timeseries_and_report(results: list[dict], config: dict) -> dict:
    requested = config["user_inputs"].get("generate_timeseries_report", True)

    if not requested:
        print("\nTimeseries/report generation skipped by user.")
        return {
            "ok": True,
            "skipped": True,
            "reason": "user_selected_no",
        }

    if config["user_inputs"]["execution_mode"] != "run":
        print("\nTimeseries/report generation skipped: execution mode is not 'run'.")
        return {
            "ok": True,
            "skipped": True,
            "reason": "execution_mode_not_run",
        }

    if not results:
        print("\nTimeseries/report generation skipped: no datasets were processed.")
        return {
            "ok": False,
            "skipped": True,
            "reason": "no_results",
        }

    successful = [
        r for r in results
        if r.get("execution", {}).get("status") == "SUCCESS"
    ]

    non_successful = [
        r for r in results
        if r.get("execution", {}).get("status") != "SUCCESS"
    ]

    if not successful:
        print("\nTimeseries/report generation skipped: no successful datasets are available.")
        return {
            "ok": False,
            "skipped": True,
            "reason": "no_successful_datasets",
            "n_non_successful": len(non_successful),
        }

    def _failed_dataset_record(r: dict) -> dict:
        identity = r.get("identity", {})
        execution = r.get("execution", {})
        outputs = r.get("outputs", {})
        time_window = r.get("time_window", {})
        raw = r.get("raw", {})

        return {
            "dataset_name": identity.get("dataset_name", ""),
            "status": execution.get("status", ""),
            "message": execution.get("message", ""),
            "run_dir": outputs.get("run_dir", ""),
            "yaml_path": outputs.get("yaml_path", ""),
            "start_epoch": time_window.get("start_epoch", ""),
            "end_epoch": time_window.get("end_epoch", ""),
            "raw_rinex_file": raw.get("raw_rinex_file", ""),
        }

    failed_datasets = [_failed_dataset_record(r) for r in non_successful]

    run_dirs = []
    skipped_success = []

    for r in successful:
        dataset_name = r.get("identity", {}).get("dataset_name", "")
        run_dir = r.get("outputs", {}).get("run_dir", "")

        if not run_dir:
            skipped_success.append((dataset_name, "missing run_dir"))
            continue

        run_dirs.append(Path(run_dir))

    if not run_dirs:
        print("\nTimeseries/report generation skipped: no valid successful run directories are available.")
        return {
            "ok": False,
            "skipped": True,
            "reason": "no_valid_successful_run_dirs",
            "n_successful": len(successful),
            "n_non_successful": len(non_successful),
        }

    ginan_process_dir = Path(successful[0]["outputs"]["ginan_process_dir"])
    timeseries_path = ginan_process_dir / "timeseries.out"

    print("\nGenerating position timeseries")
    print("------------------------------")
    print(f"Successful datasets used : {len(run_dirs)}")
    print(f"Non-successful datasets  : {len(non_successful)}")
    print(f"Output file              : {timeseries_path}")

    if skipped_success:
        print()
        print("Successful datasets skipped from timeseries because of missing output metadata:")
        for dataset_name, reason in skipped_success:
            print(f"  {dataset_name}: {reason}")

    ts_result = position_timeseries.build_timeseries_out(
        run_dirs=run_dirs,
        output_path=timeseries_path,
        convergence_config=None,
        overwrite=True,
    )

    print(f"timeseries.out  : {ts_result['output_path']}")
    print(f"Rows            : {ts_result['n_rows']}")
    print(f"ENU reference   : {ts_result['series_enu_reference_run_label']}")

    print("\nGenerating timeseries report")
    print("----------------------------")

    report_plot_columns = config["user_inputs"].get("report_plot_columns", ["X", "Y", "Z", "h"])
    print(f"Report plots    : {', '.join(report_plot_columns)}")

    report_result = timeseries_report.build_timeseries_report(
        timeseries_path=timeseries_path,
        report_path=ginan_process_dir / "timeseries.report",
        plot_columns=report_plot_columns,
        failed_datasets=failed_datasets,
    )

    print(f"timeseries.report : {report_result['report_path']}")
    print(f"Report format      : {report_result.get('report_format', '')}")
    print(f"Report lines       : {report_result['n_report_lines']}")

    if non_successful:
        print()
        print("Timeseries/report generation completed with warnings: not all datasets completed successfully.")
        print("Non-successful datasets included in report metadata:")
        for item in failed_datasets:
            print(f"  {item['dataset_name']}: {item['status']} -> {item['message']}")
    else:
        print()
        print("Timeseries/report generation completed successfully.")

    return {
        "ok": True,
        "skipped": False,
        "warnings": bool(non_successful),
        "n_successful_used": len(run_dirs),
        "n_non_successful": len(non_successful),
        "failed_datasets": failed_datasets,
        "timeseries_result": ts_result,
        "report_result": report_result,
    }
def main() -> None:
    config = collect_user_inputs()

    preflight = system_check.run_preflight_checks(config)
    print("\nPreflight check")
    print("---------------")
    for item in preflight["checks"]:
        status = "OK" if item["ok"] else "FAIL"
        print(f"[{status}] {item['check_type']}: {item['name']} -> {item['message']}")

    if not preflight["ok"]:
        print("\nAborting: preflight checks failed.")
        return

    results = process_batch(config)
    print_final_summary(results, config)

    generate_timeseries_and_report(results, config)


# === PATCH: GUI/CLI path prompts START ===
# User prompt order:
# 1. RAW RINEX root directory
# 2. Static Ginan products directory
# 3. YAML template path
# 4. Remaining processing parameters

def collect_user_inputs() -> dict:
    base_config = paths_config.get_default_config()
    system_cfg = base_config.get("system", {})
    user_cfg = base_config.get("user_inputs", {})

    raw_root = _prompt_str(
        "Enter RAW root path containing the daily dataset folders",
        default=user_cfg.get("raw_root") if user_cfg.get("raw_root") else None,
    )

    print()
    print("Static Ginan auxiliary products directory.")
    print("This is usually located at:")
    print("  ~/ginan-gui-linux-x64/_internal/scripts/GinanUI/app/resources/inputData/products")
    print("or, in this system, possibly at:")
    print("  ~/opt/ginan/ginan-gui-linux-x64/_internal/scripts/GinanUI/app/resources/inputData/products")

    default_static_root = str(
        system_cfg.get("static_products_root", "")
        or system_cfg.get("static_products_root_hint", "")
    ).strip()

    static_products_root = _prompt_str(
        "Enter static Ginan products directory",
        default=default_static_root if default_static_root else None,
        allow_empty=False,
    )

    default_template_yaml = str(system_cfg.get("template_yaml_path", "")).strip()

    template_yaml_path = _prompt_str(
        "Enter YAML template path",
        default=default_template_yaml if default_template_yaml else None,
        allow_empty=False,
    )

    requested_sample_rate_sec = _prompt_int(
        "Enter requested sample rate in seconds",
        default=user_cfg.get("requested_sample_rate_sec"),
    )

    provider = _prompt_str(
        "Enter PPP provider (e.g. COD, GRG, GFZ, WUM, EMR, IGS)",
        default=user_cfg.get("provider", "COD"),
    ).upper()

    series = _prompt_str(
        "Enter PPP series (FIN or RAP)",
        default=user_cfg.get("series", "FIN"),
    ).upper()

    project = _prompt_str(
        "Enter PPP project (MGX or OPS)",
        default=user_cfg.get("project", "MGX"),
    ).upper()

    execution_mode = _prompt_str(
        "Enter execution mode (run or build_only)",
        default=user_cfg.get("execution_mode", "run"),
    ).lower()

    overwrite = _prompt_bool(
        "Overwrite existing resampled files / outputs when possible",
        default=bool(user_cfg.get("overwrite", False)),
    )

    limit_raw = _safe_input("Limit number of datasets to process [empty/all = all]: ").strip()
    if limit_raw == "" or limit_raw.lower() == "all":
        limit_datasets = None
    else:
        limit_datasets = int(limit_raw)

    generate_timeseries_report = _prompt_bool(
        "Generate position timeseries and report after successful run",
        default=True,
    )

    report_plot_columns_raw = _prompt_str(
        "Report plot columns, comma-separated; valid: X,Y,Z,lon,lat,h,E,N,U,all",
        default="X,Y,Z,h",
    )
    report_plot_columns = _parse_report_plot_columns(report_plot_columns_raw)

    updates = {
        "system": {
            "static_products_root": static_products_root,
            "template_yaml_path": template_yaml_path,
        },
        "user_inputs": {
            "raw_root": raw_root,
            "requested_sample_rate_sec": requested_sample_rate_sec,
            "provider": provider,
            "series": series,
            "project": project,
            "execution_mode": execution_mode,
            "overwrite": overwrite,
            "limit_datasets": limit_datasets,
            "generate_timeseries_report": generate_timeseries_report,
            "report_plot_columns": report_plot_columns,
        },
    }

    return paths_config.merge_user_inputs(base_config, updates)
# === PATCH: GUI/CLI path prompts END ===

if __name__ == "__main__":
    main()
