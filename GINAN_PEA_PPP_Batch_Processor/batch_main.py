from __future__ import annotations

from copy import deepcopy
import json
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


VALID_REPORT_PLOT_COLUMNS = {"X", "Y", "Z", "lon", "lat", "h", "E", "N", "U"}


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

def write_run_manifest(dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    manifest_path = ctx.get("outputs", {}).get("manifest_path")
    if not manifest_path:
        return {
            "ok": False,
            "message": "manifest_path is missing in dataset_context",
            "manifest_path": "",
        }

    p = Path(manifest_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    ctx.setdefault("execution", {})
    ctx["execution"]["manifest_written"] = True
    ctx["execution"]["manifest_path"] = str(p)

    with p.open("w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False, sort_keys=True)

    return {
        "ok": True,
        "message": "Run manifest written successfully.",
        "manifest_path": str(p),
    }


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

        ctx = products_download.resolve_product_files(config, ctx)
        product_validation = products_download.validate_product_files(ctx)

        if failed_downloads or not product_validation["ok"]:
            ctx = products_download.ensure_fallback_products_available(config, ctx)
            ctx = products_download.resolve_product_files(config, ctx)
            product_validation = products_download.validate_product_files(ctx)

        if failed_downloads and not product_validation["ok"]:
            first = failed_downloads[0]
            raise RuntimeError(
                f"Download step failed ({first['kind']} / {first['date']}) with return code {first['returncode']}"
            )

        if not product_validation["ok"]:
            raise FileNotFoundError(
                "Missing product files: " + " | ".join(product_validation["missing"])
            )

        ctx["execution"]["downloads_completed"] = len(failed_downloads) == 0

        if failed_downloads:
            ctx["execution"]["download_warnings"] = [
                f"{r['kind']} / {r['date']} returned code {r['returncode']}"
                for r in failed_downloads
            ]

        if ctx.get("products", {}).get("fallback_product_report"):
            ctx["execution"]["fallback_products_used"] = True
            ctx["execution"]["fallback_product_report"] = ctx["products"]["fallback_product_report"]

        ctx = yaml_builder.build_output_paths(config, ctx)
        yaml_text = yaml_builder.render_yaml_from_template(config, ctx)
        write_result = yaml_builder.write_yaml_file(yaml_text, ctx["outputs"]["yaml_path"])
        if not write_result["ok"]:
            raise RuntimeError(write_result["message"])

        ctx["execution"]["yaml_written"] = True

        if config["user_inputs"]["execution_mode"] == "build_only":
            ctx["execution"]["status"] = "BUILT"
            ctx["execution"]["message"] = "YAML built successfully. PEA execution skipped."
            manifest_result = write_run_manifest(ctx)
            ctx["execution"]["manifest_written"] = manifest_result["ok"]
            ctx["execution"]["manifest_path"] = manifest_result["manifest_path"]
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

        manifest_result = write_run_manifest(ctx)
        ctx["execution"]["manifest_written"] = manifest_result["ok"]
        ctx["execution"]["manifest_path"] = manifest_result["manifest_path"]

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
    report_analysis_config = config["user_inputs"].get("report_analysis_config", {})
    print(f"Report plots    : {', '.join(report_plot_columns)}")
    print(f"Report analysis config supplied: {bool(report_analysis_config)}")

    report_result = timeseries_report.build_timeseries_report(
        timeseries_path=timeseries_path,
        report_path=ginan_process_dir / "timeseries.report",
        plot_columns=report_plot_columns,
        failed_datasets=failed_datasets,
        report_analysis_config=config["user_inputs"].get("report_analysis_config", {}),
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

def generate_report_only_from_existing_outputs(config: dict) -> dict:
    raw_root = Path(config["user_inputs"]["raw_root"]).expanduser().resolve()
    ginan_process_dir = raw_root / "GINAN_process"

    print("\nReport-only / post-processing-only mode")
    print("---------------------------------------")
    print(f"RAW root          : {raw_root}")
    print(f"GINAN_process dir : {ginan_process_dir}")

    if not ginan_process_dir.exists() or not ginan_process_dir.is_dir():
        raise NotADirectoryError(f"GINAN_process directory does not exist: {ginan_process_dir}")

    candidate_dirs = sorted(
        p for p in ginan_process_dir.iterdir()
        if p.is_dir()
    )

    run_dirs = []
    skipped = []

    for run_dir in candidate_dirs:
        try:
            position_timeseries.find_pos_files(run_dir, use_smoothed_pos=True)
            run_dirs.append(run_dir)
            continue
        except Exception as exc_smoothed:
            try:
                position_timeseries.find_pos_files(run_dir, use_smoothed_pos=False)
                run_dirs.append(run_dir)
                continue
            except Exception as exc_raw:
                skipped.append((run_dir.name, str(exc_raw) or str(exc_smoothed)))

    if not run_dirs:
        raise RuntimeError(f"No existing PEA/POS run directories found under: {ginan_process_dir}")

    timeseries_path = ginan_process_dir / "timeseries.out"
    report_path = ginan_process_dir / "timeseries.report.html"

    print()
    print("Generating position timeseries from existing run outputs")
    print("--------------------------------------------------------")
    print(f"Detected run directories : {len(candidate_dirs)}")
    print(f"Usable run directories   : {len(run_dirs)}")
    print(f"Skipped directories      : {len(skipped)}")
    print(f"Output timeseries        : {timeseries_path}")

    if skipped:
        print()
        print("Directories skipped because no usable POS files were found:")
        for name, reason in skipped[:30]:
            print(f"  {name}: {reason}")
        if len(skipped) > 30:
            print(f"  ... {len(skipped) - 30} more skipped directories")

    ts_result = position_timeseries.build_timeseries_out(
        run_dirs=run_dirs,
        output_path=timeseries_path,
        convergence_config=None,
        overwrite=True,
    )

    print(f"timeseries.out : {ts_result['output_path']}")
    print(f"Rows           : {ts_result['n_rows']}")
    print(f"ENU reference  : {ts_result['series_enu_reference_run_label']}")

    print()
    print("Generating timeseries report")
    print("----------------------------")

    report_plot_columns = config["user_inputs"].get("report_plot_columns", ["X", "Y", "Z", "h"])
    report_analysis_config = config["user_inputs"].get("report_analysis_config", {})
    print(f"Report plots   : {', '.join(report_plot_columns)}")
    print(f"Report analysis config supplied: {bool(report_analysis_config)}")
    print(f"Output report  : {report_path}")

    report_result = timeseries_report.build_timeseries_report(
        timeseries_path=timeseries_path,
        report_path=report_path,
        plot_columns=report_plot_columns,
        failed_datasets=[],
        report_analysis_config=config["user_inputs"].get("report_analysis_config", {}),
    )

    print(f"timeseries.report.html : {report_result['report_path']}")
    print(f"Report format          : {report_result.get('report_format', '')}")
    print(f"Report lines           : {report_result['n_report_lines']}")
    print(f"Shift clusters         : {report_result.get('n_shift_clusters', '')}")
    print(f"Meta clusters          : {report_result.get('n_meta_clusters', '')}")
    print(f"Velocity change clusters: {report_result.get('n_velocity_change_clusters', '')}")
    print(f"Transient windows: {report_result.get('n_transient_windows', '')}")
    print(f"Component-wise transient model fits: {report_result.get('n_transient_model_fits', '')}")
    print(f"Joint horizontal transient model fits: {report_result.get('n_joint_horizontal_transient_model_fits', '')}")
    print(f"Shift-related report-grade velocity changes: {report_result.get('n_shift_related_report_grade_velocity_changes', '')}")

    print()
    print("Report-only generation completed successfully.")

    return {
        "ok": True,
        "mode": "report_only",
        "n_detected_run_dirs": len(candidate_dirs),
        "n_used_run_dirs": len(run_dirs),
        "n_skipped_run_dirs": len(skipped),
        "timeseries_result": ts_result,
        "report_result": report_result,
    }



def generate_report_from_existing_timeseries(config: dict) -> dict:
    raw_root = Path(config["user_inputs"]["raw_root"]).expanduser().resolve()
    ginan_process_dir = raw_root / "GINAN_process"
    timeseries_path = ginan_process_dir / "timeseries.out"
    report_path = ginan_process_dir / "timeseries.report.html"

    print("\nReport-from-timeseries mode")
    print("---------------------------")
    print(f"RAW root          : {raw_root}")
    print(f"GINAN_process dir : {ginan_process_dir}")
    print(f"Input timeseries  : {timeseries_path}")
    print(f"Output report     : {report_path}")

    if not ginan_process_dir.exists() or not ginan_process_dir.is_dir():
        raise NotADirectoryError(f"GINAN_process directory does not exist: {ginan_process_dir}")

    if not timeseries_path.exists() or not timeseries_path.is_file():
        raise FileNotFoundError(f"timeseries.out does not exist: {timeseries_path}")

    print()
    print("Generating timeseries report from existing timeseries.out")
    print("---------------------------------------------------------")

    report_plot_columns = config["user_inputs"].get("report_plot_columns", ["X", "Y", "Z", "h"])
    report_analysis_config = config["user_inputs"].get("report_analysis_config", {})
    print(f"Report plots   : {', '.join(report_plot_columns)}")
    print(f"Report analysis config supplied: {bool(report_analysis_config)}")

    report_result = timeseries_report.build_timeseries_report(
        timeseries_path=timeseries_path,
        report_path=report_path,
        plot_columns=report_plot_columns,
        failed_datasets=[],
        report_analysis_config=config["user_inputs"].get("report_analysis_config", {}),
    )

    print(f"timeseries.report.html : {report_result['report_path']}")
    print(f"Report format          : {report_result.get('report_format', '')}")
    print(f"Report lines           : {report_result['n_report_lines']}")
    print(f"Shift clusters         : {report_result.get('n_shift_clusters', '')}")
    print(f"Meta clusters          : {report_result.get('n_meta_clusters', '')}")
    print(f"Velocity change clusters: {report_result.get('n_velocity_change_clusters', '')}")
    print(f"Transient windows: {report_result.get('n_transient_windows', '')}")
    print(f"Component-wise transient model fits: {report_result.get('n_transient_model_fits', '')}")
    print(f"Joint horizontal transient model fits: {report_result.get('n_joint_horizontal_transient_model_fits', '')}")
    print(f"Shift-related report-grade velocity changes: {report_result.get('n_shift_related_report_grade_velocity_changes', '')}")

    print()
    print("Report-from-timeseries generation completed successfully.")

    return {
        "ok": True,
        "mode": "report_from_timeseries",
        "timeseries_path": str(timeseries_path),
        "report_result": report_result,
    }


def main() -> None:
    config = collect_user_inputs()

    if config["user_inputs"].get("execution_mode") == "report_from_timeseries":
        generate_report_from_existing_timeseries(config)
        return

    if config["user_inputs"].get("execution_mode") == "report_only":
        generate_report_only_from_existing_outputs(config)
        return

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


def _prompt_report_analysis_config_json() -> dict:
    prompt = "Report analysis config JSON [empty = defaults]: "

    try:
        raw = input(prompt).strip()
    except EOFError:
        return {}

    if raw == "":
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid report analysis config JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Report analysis config JSON must decode to an object/dict.")

    return parsed

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
    print("  /home/ioannis/opt/ginan/ginan-gui-linux-x64/_internal/scripts/GinanUI/app/resources/inputData/products")

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
        "Enter execution mode (run, build_only, report_only, or report_from_timeseries)",
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

    report_analysis_config = _prompt_report_analysis_config_json()

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
            "report_analysis_config": report_analysis_config,
        },
    }

    return paths_config.merge_user_inputs(base_config, updates)
# === PATCH: GUI/CLI path prompts END ===

if __name__ == "__main__":
    main()
