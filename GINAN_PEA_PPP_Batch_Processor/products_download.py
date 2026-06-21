from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import subprocess
import requests
import shutil


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
            "--data-source", "cddis",
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


def _gps_week_and_dow_from_day_code(day_code: str) -> tuple[int, int]:
    from datetime import datetime

    dt = datetime.strptime(day_code, "%Y%j")
    gps_epoch = datetime(1980, 1, 6)
    delta_days = (dt - gps_epoch).days

    if delta_days < 0:
        raise ValueError(f"Date is before GPS epoch: {day_code}")

    gps_week = delta_days // 7
    gps_dow = delta_days % 7

    return gps_week, gps_dow


def _first_existing_or_first(candidates: list[Path]) -> Path:
    if not candidates:
        raise ValueError("No product candidates provided")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]




def _cddis_product_base_url(gps_week: int) -> str:
    return f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}"


def _remote_product_exists(url: str, timeout_sec: int = 30) -> bool:
    try:
        response = requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout_sec,
            headers={"Range": "bytes=0-0"},
        )
        ok = response.status_code in (200, 206)
        response.close()
        return ok
    except Exception:
        return False


def _download_binary_file(url: str, output_path: Path, timeout_sec: int = 120) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "ok": True,
            "status": "exists",
            "url": url,
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    response = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=timeout_sec,
    )

    if response.status_code not in (200, 206):
        text = response.text[:500] if hasattr(response, "text") else ""
        response.close()
        return {
            "ok": False,
            "status": f"http_{response.status_code}",
            "url": url,
            "path": str(output_path),
            "message": text,
        }

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    response.close()

    return {
        "ok": output_path.exists() and output_path.stat().st_size > 0,
        "status": "downloaded",
        "url": url,
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def _decompress_with_gzip_cli(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "ok": True,
            "status": "exists",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    with output_path.open("wb") as fout:
        proc = subprocess.run(
            ["gzip", "-dc", str(input_path)],
            stdout=fout,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )

    if proc.returncode != 0:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "ok": False,
            "status": "decompress_failed",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "returncode": proc.returncode,
            "stderr": proc.stderr.decode("utf-8", errors="ignore") if proc.stderr else "",
        }

    return {
        "ok": output_path.exists() and output_path.stat().st_size > 0,
        "status": "decompressed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def _fallback_product_specs_for_day(
    day_code: str,
    provider: str,
    project: str,
    series: str,
) -> dict[str, list[dict]]:
    gps_week, gps_dow = _gps_week_and_dow_from_day_code(day_code)

    stamp = f"{day_code}0000"
    year = day_code[:4]
    yy = year[2:4]
    gps_week_day = f"{gps_week}{gps_dow}"

    provider = provider.upper()
    project = project.upper()
    series = series.upper()
    provider_lower = provider.lower()

    base_url = _cddis_product_base_url(gps_week)

    def spec(remote_name: str, local_name: str) -> dict:
        return {
            "url": f"{base_url}/{remote_name}",
            "compressed_name": remote_name,
            "local_name": local_name,
        }

    return {
        "snx": [
            spec(
                f"IGS0OPSSNX_{stamp}_01D_01D_CRD.SNX.gz",
                f"IGS0OPSSNX_{stamp}_01D_01D_CRD.SNX",
            ),
            spec(
                f"igs{yy}P{gps_week}.snx.Z",
                f"igs{yy}P{gps_week}.snx",
            ),
        ],
        "sp3": [
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_05M_ORB.SP3.gz",
                f"{provider}0{project}{series}_{stamp}_01D_05M_ORB.SP3",
            ),
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_15M_ORB.SP3.gz",
                f"{provider}0{project}{series}_{stamp}_01D_15M_ORB.SP3",
            ),
            spec(
                f"{provider_lower}{gps_week_day}.sp3.Z",
                f"{provider_lower}{gps_week_day}.sp3",
            ),
            spec(
                f"igs{gps_week_day}.sp3.Z",
                f"igs{gps_week_day}.sp3",
            ),
        ],
        "clk": [
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_30S_CLK.CLK.gz",
                f"{provider}0{project}{series}_{stamp}_01D_30S_CLK.CLK",
            ),
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_05M_CLK.CLK.gz",
                f"{provider}0{project}{series}_{stamp}_01D_05M_CLK.CLK",
            ),
            spec(
                f"{provider_lower}{gps_week_day}.clk.Z",
                f"{provider_lower}{gps_week_day}.clk",
            ),
            spec(
                f"igs{gps_week_day}.clk.Z",
                f"igs{gps_week_day}.clk",
            ),
        ],
        "bia": [
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_01D_OSB.BIA.gz",
                f"{provider}0{project}{series}_{stamp}_01D_01D_OSB.BIA",
            ),
            spec(
                f"{provider}0{project}{series}_{stamp}_01D_01D_BIA.BIA.gz",
                f"{provider}0{project}{series}_{stamp}_01D_01D_BIA.BIA",
            ),
            spec(
                f"{provider_lower}{gps_week_day}.bia.Z",
                f"{provider_lower}{gps_week_day}.bia",
            ),
            spec(
                f"igs{gps_week_day}.bia.Z",
                f"igs{gps_week_day}.bia",
            ),
        ],
    }


def _any_existing_product_for_specs(igs_precise_dir: Path, specs: list[dict]) -> Path | None:
    for item in specs:
        candidate = igs_precise_dir / item["local_name"]
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def ensure_fallback_products_available(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    igs_precise_dir = Path(ctx["products"]["igs_precise_dir"]).expanduser().resolve()
    igs_precise_dir.mkdir(parents=True, exist_ok=True)

    day_codes = ctx["time_window"]["day_codes"]

    provider = config["user_inputs"]["provider"].upper()
    project = config["user_inputs"]["project"].upper()
    series = config["user_inputs"]["series"].upper()

    report = []

    for day_code in day_codes:
        specs_by_kind = _fallback_product_specs_for_day(
            day_code=day_code,
            provider=provider,
            project=project,
            series=series,
        )

        for kind, specs in specs_by_kind.items():
            existing = _any_existing_product_for_specs(igs_precise_dir, specs)

            if existing is not None:
                report.append({
                    "day_code": day_code,
                    "kind": kind,
                    "status": "exists",
                    "path": str(existing),
                })
                continue

            selected = None
            for spec in specs:
                if _remote_product_exists(spec["url"]):
                    selected = spec
                    break

            if selected is None:
                report.append({
                    "day_code": day_code,
                    "kind": kind,
                    "status": "not_found_remote",
                    "candidates": [item["url"] for item in specs],
                })
                continue

            compressed_path = igs_precise_dir / selected["compressed_name"]
            local_path = igs_precise_dir / selected["local_name"]

            download_result = _download_binary_file(selected["url"], compressed_path)
            if not download_result["ok"]:
                report.append({
                    "day_code": day_code,
                    "kind": kind,
                    "status": "download_failed",
                    "url": selected["url"],
                    "result": download_result,
                })
                continue

            decompress_result = _decompress_with_gzip_cli(compressed_path, local_path)

            report.append({
                "day_code": day_code,
                "kind": kind,
                "status": "ready" if decompress_result["ok"] else "decompress_failed",
                "url": selected["url"],
                "compressed_path": str(compressed_path),
                "local_path": str(local_path),
                "download_result": download_result,
                "decompress_result": decompress_result,
            })

    ctx.setdefault("products", {})
    ctx["products"]["fallback_product_report"] = report

    return ctx




def _compute_product_staging_run_label(dataset_context: dict) -> str:
    dataset_name = dataset_context["identity"]["dataset_name"]
    eff = dataset_context["resampling"]["effective_interval_sec"]

    if eff is None:
        raise ValueError("effective_interval_sec is missing in dataset_context")

    return f"{dataset_name}_{int(eff)}s"


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy_product(src: Path, dst: Path) -> dict:
    src = src.expanduser().resolve()
    dst = dst.expanduser()

    if not src.exists():
        raise FileNotFoundError(f"Cannot stage missing product: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_path(dst)

    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return {
            "mode": "symlink",
            "source": str(src),
            "target": str(dst),
        }
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
            mode = "copytree"
        else:
            shutil.copy2(src, dst)
            mode = "copy2"

        return {
            "mode": mode,
            "source": str(src),
            "target": str(dst),
        }


def _stage_products_for_dataset(config: dict, dataset_context: dict) -> dict:
    """
    Build an isolated per-dataset product workspace.

    PEA/Ginan is kept in its GUI-style wildcard mode for satellite products
    (*.SP3, *.CLK, *.BIA, BRDC*) but inputs_root points to this isolated
    staging folder, not to the shared IGS_PRECISE directory. This prevents
    loading every product in a large shared product folder while preserving
    the stable wildcard loading behavior.
    """
    ctx = deepcopy(dataset_context)

    products = ctx.setdefault("products", {})
    igs_precise_dir = Path(products["igs_precise_dir"]).expanduser().resolve()

    raw_dataset_dir = Path(ctx["raw"]["raw_dataset_dir"])
    raw_root = Path(ctx["raw"].get("raw_root") or raw_dataset_dir.parent).expanduser().resolve()

    staging_folder_name = config.get("processing", {}).get(
        "product_staging_folder_name",
        "PRODUCT_STAGING",
    )

    run_label = _compute_product_staging_run_label(ctx)
    staging_dir = raw_root / staging_folder_name / run_label

    _remove_existing_path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged = []

    # The template references many static products under tables/.
    tables_src = igs_precise_dir / "tables"
    staged.append(_link_or_copy_product(tables_src, staging_dir / "tables"))

    # Top-level static files referenced by the template.
    for rel_path in [
        "igs20.atx",
        "finals.data.iau2000.txt",
        "igs_satellite_metadata.snx",
    ]:
        staged.append(_link_or_copy_product(igs_precise_dir / rel_path, staging_dir / rel_path))

    # Explicitly resolved dynamic products selected for this dataset.
    dynamic_keys = ["snx_files", "nav_files", "clk_files", "bsx_files", "sp3_files"]
    seen_targets = set()

    for key in dynamic_keys:
        for item in products.get(key, []):
            src = Path(item).expanduser().resolve()
            dst = staging_dir / src.name

            if dst.name in seen_targets:
                continue

            staged.append(_link_or_copy_product(src, dst))
            seen_targets.add(dst.name)

    products["product_staging_dir"] = str(staging_dir)
    products["product_staging_source_dir"] = str(igs_precise_dir)
    products["product_staging_files"] = staged
    products["product_staging_mode"] = "isolated_wildcard_inputs_root"

    return ctx


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
        year = day_code[:4]
        yy = year[2:4]
        gps_week, gps_dow = _gps_week_and_dow_from_day_code(day_code)
        gps_week_day = f"{gps_week}{gps_dow}"

        provider_lower = provider.lower()

        snx_candidates = [
            igs_precise_dir / f"IGS0OPSSNX_{stamp}_01D_01D_CRD.SNX",
            igs_precise_dir / f"IGS0OPSSNX_{stamp}_01D_01D_CRD.snx",
            igs_precise_dir / f"igs{yy}P{gps_week}.snx",
            igs_precise_dir / f"IGS{yy}P{gps_week}.SNX",
        ]

        nav_candidates = [
            igs_precise_dir / f"BRDC00IGS_R_{stamp}_01D_MN.rnx",
            igs_precise_dir / f"BRDC00IGS_R_{stamp}_01D_MN.RNX",
        ]

        sp3_candidates = [
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_05M_ORB.SP3",
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_15M_ORB.SP3",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.sp3",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.SP3",
        ]

        clk_candidates = [
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_30S_CLK.CLK",
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_05M_CLK.CLK",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.clk",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.CLK",
        ]

        bia_candidates = [
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_01D_OSB.BIA",
            igs_precise_dir / f"{provider}0{project}{series}_{stamp}_01D_01D_BIA.BIA",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.bia",
            igs_precise_dir / f"{provider_lower}{gps_week_day}.BIA",
        ]

        snx_files.append(str(_first_existing_or_first(snx_candidates)))
        nav_files.append(str(_first_existing_or_first(nav_candidates)))
        sp3_files.append(str(_first_existing_or_first(sp3_candidates)))
        clk_files.append(str(_first_existing_or_first(clk_candidates)))
        bsx_files.append(str(_first_existing_or_first(bia_candidates)))

    local_erp = igs_precise_dir / "finals.data.iau2000.txt"
    if local_erp.exists():
        erp_files.append(str(local_erp))

    ctx["products"]["snx_files"] = snx_files
    ctx["products"]["nav_files"] = nav_files
    ctx["products"]["erp_files"] = erp_files
    ctx["products"]["clk_files"] = clk_files
    ctx["products"]["bsx_files"] = bsx_files
    ctx["products"]["sp3_files"] = sp3_files

    ctx = _stage_products_for_dataset(config, ctx)

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
    "tables/fes2014b_Cnm-Snm.dat",
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

