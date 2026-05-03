from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import subprocess


def ensure_products_dir(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    raw_dataset_dir = Path(ctx["raw"]["raw_dataset_dir"])
    raw_root = Path(ctx["raw"].get("raw_root") or raw_dataset_dir.parent).expanduser().resolve()
    igs_precise_dir = raw_root / config["processing"]["igs_precise_folder_name"]
    igs_precise_dir.mkdir(parents=True, exist_ok=True)

    ctx["products"]["igs_precise_dir"] = str(igs_precise_dir)
    return ctx


def _day_start_end(date_str: str) -> tuple[str, str]:
    return f"{date_str}_00:00:00", f"{date_str}_23:59:59"


def build_download_plan(config: dict, dataset_context: dict) -> list[dict]:
    covered_dates = dataset_context["time_window"]["covered_dates"]
    igs_precise_dir = dataset_context["products"]["igs_precise_dir"]

    if not covered_dates:
        raise ValueError("covered_dates is missing in dataset_context")
    if not igs_precise_dir:
        raise ValueError("igs_precise_dir is missing in dataset_context")

    python_path = config["system"]["downloader_python_path"]
    auto_download_script = config["system"]["auto_download_script"]

    provider = config["user_inputs"]["provider"].upper()
    project = config["user_inputs"]["project"].upper()
    series = config["user_inputs"]["series"].upper()
    dont_replace = config["download_policy"]["dont_replace"]

    plan = []

    for date_str in covered_dates:
        start_dt, end_dt = _day_start_end(date_str)

        cmd_snx_nav_erp = [
            python_path,
            auto_download_script,
            "--target-dir", igs_precise_dir,
            "--start-datetime", start_dt,
            "--end-datetime", end_dt,
            "--snx", "--nav", "--erp",
        ]
        if dont_replace:
            cmd_snx_nav_erp.append("--dont-replace")

        plan.append({
            "kind": "snx_nav_erp",
            "date": date_str,
            "command": cmd_snx_nav_erp,
        })

        cmd_sp3_clk = [
            python_path,
            auto_download_script,
            "--target-dir", igs_precise_dir,
            "--start-datetime", start_dt,
            "--end-datetime", end_dt,
            "--analysis-center", provider,
            "--project-type", project,
            "--solution-type", series,
            "--sp3", "--clk",
        ]
        if dont_replace:
            cmd_sp3_clk.append("--dont-replace")

        plan.append({
            "kind": "sp3_clk",
            "date": date_str,
            "command": cmd_sp3_clk,
        })

        cmd_bia = [
            python_path,
            auto_download_script,
            "--target-dir", igs_precise_dir,
            "--start-datetime", start_dt,
            "--end-datetime", end_dt,
            "--analysis-center", provider,
            "--bia-ac", provider,
            "--solution-type", series,
            "--project-type", project,
            "--bia",
        ]
        if dont_replace:
            cmd_bia.append("--dont-replace")

        plan.append({
            "kind": "bia",
            "date": date_str,
            "command": cmd_bia,
        })

    return plan


def run_download_plan(download_plan: list[dict]) -> list[dict]:
    results = []

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    for item in download_plan:
        proc = subprocess.run(
            item["command"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        results.append({
            "kind": item["kind"],
            "date": item["date"],
            "command": item["command"],
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })

    return results


def resolve_product_files(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    igs_precise_dir = Path(ctx["products"]["igs_precise_dir"])
    day_codes = ctx["time_window"]["day_codes"]

    provider = config["user_inputs"]["provider"].upper()
    project = config["user_inputs"]["project"].upper()
    series = config["user_inputs"]["series"].upper()

    if not day_codes:
        raise ValueError("day_codes is missing in dataset_context")

    snx_files = []
    nav_files = []
    erp_files = []
    clk_files = []
    bsx_files = []
    sp3_files = []

    for day_code in day_codes:
        stamp = f"{day_code}0000"

        snx_files.append(str(igs_precise_dir / f"IGS0OPSSNX_{stamp}_01D_01D_CRD.SNX"))
        nav_files.append(str(igs_precise_dir / f"BRDC00IGS_R_{stamp}_01D_MN.rnx"))
        sp3_files.append(str(igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_05M_ORB.SP3"))
        clk_files.append(str(igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_30S_CLK.CLK"))
        bsx_files.append(str(igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_01D_OSB.BIA"))

    local_erp = igs_precise_dir / "finals.data.iau2000.txt"
    if local_erp.exists():
        erp_files.append(str(local_erp))

    ctx["products"]["snx_files"] = snx_files
    ctx["products"]["nav_files"] = nav_files
    ctx["products"]["erp_files"] = erp_files
    ctx["products"]["clk_files"] = clk_files
    ctx["products"]["bsx_files"] = bsx_files
    ctx["products"]["sp3_files"] = sp3_files

    return ctx


def validate_product_files(dataset_context: dict) -> dict:
    required_keys = ["snx_files", "nav_files", "clk_files", "bsx_files", "sp3_files"]

    missing = []
    counts = {}

    for key in required_keys:
        files = dataset_context["products"].get(key, [])
        counts[key] = len(files)

        if not files:
            missing.append(f"{key}: <empty list>")
            continue

        for f in files:
            if not Path(f).exists():
                missing.append(f)

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "counts": counts,
    }


# === PATCH: copy static products START ===
# Copy static Ginan auxiliary products from user-defined static_products_root
# into the per-project RAW_ROOT/IGS_PRECISE workspace.

import shutil as _shutil_static_products

REQUIRED_STATIC_PRODUCTS = [
    "igs_satellite_metadata.snx",
    "igs20.atx",
    "tables/sat_yaw_bias_rate.snx",
    "tables/qzss_yaw_modes.snx",
    "tables/bds_yaw_modes.snx",
    "tables/OLOAD_GO.BLQ",
    "tables/ALOAD_GO.BLQ",
    "tables/opoleloadcoefcmcor.txt",
    "tables/igrf14coeffs.txt",
    "tables/DE436.1950.2050",
    "tables/gpt_25.grd",
]

_products_download_original_ensure_products_dir_static_copy = ensure_products_dir

def copy_static_products_to_workspace(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    static_root_raw = str(config.get("system", {}).get("static_products_root", "")).strip()
    if not static_root_raw:
        raise ValueError("static_products_root is missing in config")

    static_root = Path(static_root_raw).expanduser().resolve()
    if not static_root.is_dir():
        raise NotADirectoryError(f"Static products directory does not exist: {static_root}")

    igs_precise_dir = Path(ctx["products"]["igs_precise_dir"]).expanduser().resolve()
    igs_precise_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    existing = []

    for rel_path in REQUIRED_STATIC_PRODUCTS:
        src = static_root / rel_path
        dst = igs_precise_dir / rel_path

        if not src.is_file():
            raise FileNotFoundError(f"Required static Ginan product is missing: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            existing.append(str(dst))
            continue

        _shutil_static_products.copy2(src, dst)
        copied.append(str(dst))

    ctx["products"]["static_products_root"] = str(static_root)
    ctx["products"]["static_products_copied"] = copied
    ctx["products"]["static_products_existing"] = existing
    ctx["products"]["static_products_required"] = [
        str(igs_precise_dir / rel_path) for rel_path in REQUIRED_STATIC_PRODUCTS
    ]

    return ctx


def ensure_products_dir(config: dict, dataset_context: dict) -> dict:
    ctx = _products_download_original_ensure_products_dir_static_copy(config, dataset_context)
    ctx = copy_static_products_to_workspace(config, ctx)
    return ctx
# === PATCH: copy static products END ===

