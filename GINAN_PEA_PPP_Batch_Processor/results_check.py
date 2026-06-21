from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re


def collect_run_outputs(run_dir: str) -> dict:
    run_path = Path(run_dir).expanduser().resolve()

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    if not run_path.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_path}")

    all_files = sorted([p for p in run_path.iterdir() if p.is_file()])

    pos_files = sorted([p for p in all_files if p.name.endswith(".POS") and not p.name.endswith("_smoothed.POS")])
    smoothed_pos_files = sorted([p for p in all_files if p.name.endswith("_smoothed.POS")])
    trace_files = sorted([p for p in all_files if p.name.endswith(".TRACE")])
    gpx_files = sorted([p for p in all_files if p.name.endswith(".GPX")])
    stdout_files = sorted([p for p in all_files if p.name.startswith("stdout_") and p.name.endswith(".txt")])

    return {
        "run_dir": str(run_path),
        "all_files": [str(p) for p in all_files],
        "pos_files": [str(p) for p in pos_files],
        "smoothed_pos_files": [str(p) for p in smoothed_pos_files],
        "trace_files": [str(p) for p in trace_files],
        "gpx_files": [str(p) for p in gpx_files],
        "stdout_files": [str(p) for p in stdout_files],
        "counts": {
            "all_files": len(all_files),
            "pos_files": len(pos_files),
            "smoothed_pos_files": len(smoothed_pos_files),
            "trace_files": len(trace_files),
            "gpx_files": len(gpx_files),
            "stdout_files": len(stdout_files),
        },
    }


def summarize_pos_coverage(pos_file: str) -> dict:
    pos_path = Path(pos_file).expanduser().resolve()

    if not pos_path.exists():
        raise FileNotFoundError(f"POS file does not exist: {pos_path}")
    if not pos_path.is_file():
        raise FileNotFoundError(f"POS path is not a file: {pos_path}")

    epoch_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\b")

    count = 0
    first_epoch = None
    last_epoch = None

    with pos_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = epoch_pattern.match(line)
            if not m:
                continue

            epoch = m.group(1)
            if first_epoch is None:
                first_epoch = epoch
            last_epoch = epoch
            count += 1

    return {
        "pos_file": str(pos_path),
        "epochs": count,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "ok": count > 0,
    }


def validate_run_outputs(config: dict, dataset_context: dict) -> dict:
    run_dir = dataset_context["outputs"]["run_dir"]
    outputs = collect_run_outputs(run_dir)

    require_pos = config["validation"]["require_pos"]
    require_smoothed_pos = config["validation"]["require_smoothed_pos"]
    require_trace = config["validation"]["require_trace"]

    problems = []

    if require_pos and outputs["counts"]["pos_files"] == 0:
        problems.append("No .POS file found")
    if require_smoothed_pos and outputs["counts"]["smoothed_pos_files"] == 0:
        problems.append("No _smoothed.POS file found")
    if require_trace and outputs["counts"]["trace_files"] == 0:
        problems.append("No .TRACE file found")

    pos_summary = None
    smoothed_pos_summary = None

    if outputs["counts"]["pos_files"] > 0:
        pos_summary = summarize_pos_coverage(outputs["pos_files"][0])
        if not pos_summary["ok"]:
            problems.append("Primary .POS file contains no solution epochs")

    if outputs["counts"]["smoothed_pos_files"] > 0:
        smoothed_pos_summary = summarize_pos_coverage(outputs["smoothed_pos_files"][0])
        if not smoothed_pos_summary["ok"]:
            problems.append("Primary _smoothed.POS file contains no solution epochs")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "outputs": outputs,
        "pos_summary": pos_summary,
        "smoothed_pos_summary": smoothed_pos_summary,
    }


def final_dataset_status(dataset_context: dict, validation_result: dict) -> dict:
    ctx = deepcopy(dataset_context)

    ctx["execution"]["validation_passed"] = validation_result["ok"]
    ctx["execution"]["run_output_validation"] = validation_result

    if validation_result["ok"]:
        ctx["execution"]["status"] = "SUCCESS"
        ctx["execution"]["message"] = "Run outputs validated successfully."
    else:
        ctx["execution"]["status"] = "FAILED_VALIDATION"
        ctx["execution"]["message"] = " ; ".join(validation_result["problems"])

    return ctx
