from __future__ import annotations

from pathlib import Path
import os
import shutil


def check_executable(path_or_cmd: str) -> dict:
    p = Path(path_or_cmd).expanduser()

    if p.is_absolute() or "/" in path_or_cmd:
        ok = p.exists() and p.is_file() and os.access(p, os.X_OK)
        return {
            "name": str(path_or_cmd),
            "ok": ok,
            "message": "Executable found" if ok else f"Executable not found or not executable: {p}",
        }

    resolved = shutil.which(path_or_cmd)
    ok = resolved is not None
    return {
        "name": str(path_or_cmd),
        "ok": ok,
        "message": f"Executable found at {resolved}" if ok else f"Executable not found in PATH: {path_or_cmd}",
    }


def check_file_exists(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    ok = p.exists() and p.is_file()
    return {
        "name": str(p),
        "ok": ok,
        "message": "File exists" if ok else f"File does not exist: {p}",
    }


def check_directory_exists(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    ok = p.exists() and p.is_dir()
    return {
        "name": str(p),
        "ok": ok,
        "message": "Directory exists" if ok else f"Directory does not exist: {p}",
    }


def ensure_directory_exists(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    try:
        existed = p.exists()
        p.mkdir(parents=True, exist_ok=True)
        ok = p.exists() and p.is_dir()
        message = "Directory exists" if existed else "Directory created"
        return {
            "name": str(p),
            "ok": ok,
            "message": message if ok else f"Could not create directory: {p}",
        }
    except Exception as exc:
        return {
            "name": str(p),
            "ok": False,
            "message": f"Could not create directory: {p} -> {exc}",
        }


def check_directory_writable(path: str) -> dict:
    p = Path(path).expanduser().resolve()

    if not p.exists() or not p.is_dir():
        return {
            "name": str(p),
            "ok": False,
            "message": f"Directory does not exist: {p}",
        }

    ok = os.access(p, os.W_OK)
    return {
        "name": str(p),
        "ok": ok,
        "message": "Directory is writable" if ok else f"Directory is not writable: {p}",
    }


def _derive_station_paths(raw_root: str, config: dict) -> dict:
    raw_root_path = Path(raw_root).expanduser().resolve()

    if not raw_root_path.exists() or not raw_root_path.is_dir():
        raise FileNotFoundError(f"RAW root does not exist or is not a directory: {raw_root_path}")

    return {
        "raw_root": str(raw_root_path),
        "resampled_dir": str(raw_root_path / config["processing"]["resampled_folder_name"]),
        "igs_precise_dir": str(raw_root_path / config["processing"]["igs_precise_folder_name"]),
        "ginan_process_dir": str(raw_root_path / config["processing"]["ginan_process_folder_name"]),
        "yaml_dir": str(raw_root_path / config["processing"]["yaml_subfolder_name"]),
    }


def run_preflight_checks(config: dict) -> dict:
    checks = []

    # executables / scripts / static files
    checks.append({"check_type": "executable", **check_executable(config["system"]["pea_path"])})
    checks.append({"check_type": "executable", **check_executable(config["system"]["gfzrnx_path"])})
    checks.append({"check_type": "executable", **check_executable(config["system"]["python_path"])})
    checks.append({"check_type": "executable", **check_executable(config["system"]["downloader_python_path"])})
    checks.append({"check_type": "file", **check_file_exists(config["system"]["auto_download_script"])})
    checks.append({"check_type": "file", **check_file_exists(config["system"]["template_yaml_path"])})
    checks.append({"check_type": "directory", **check_directory_exists(config["system"]["static_products_root"])})

    raw_root = config["user_inputs"]["raw_root"]
    if not raw_root:
        checks.append({
            "check_type": "input",
            "name": "raw_root",
            "ok": False,
            "message": "raw_root is empty in config['user_inputs']",
        })
    else:
        checks.append({"check_type": "directory", **check_directory_exists(raw_root)})

        try:
            station_paths = _derive_station_paths(raw_root, config)

            for key in ["igs_precise_dir", "resampled_dir", "yaml_dir", "ginan_process_dir"]:
                checks.append({"check_type": "directory", **ensure_directory_exists(station_paths[key])})

            for key in ["raw_root", "igs_precise_dir", "resampled_dir", "yaml_dir", "ginan_process_dir"]:
                checks.append({"check_type": "writable", **check_directory_writable(station_paths[key])})

        except Exception as e:
            checks.append({
                "check_type": "derived_paths",
                "name": "station_paths",
                "ok": False,
                "message": str(e),
            })

    ok = all(item["ok"] for item in checks)

    return {
        "ok": ok,
        "checks": checks,
    }

# === PATCH: static_products_root preflight START ===
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

_system_check_original_run_preflight_checks_static_products = run_preflight_checks

def run_preflight_checks(config: dict) -> dict:
    result = _system_check_original_run_preflight_checks_static_products(config)
    checks = result.get("checks", [])

    static_root = str(config.get("system", {}).get("static_products_root", "")).strip()

    if not static_root:
        checks.append({
            "check_type": "static_products",
            "ok": False,
            "name": "static_products_root",
            "message": "Static Ginan products directory is missing in config.",
        })
    else:
        static_root_path = Path(static_root).expanduser().resolve()

        checks.append({
            "check_type": "static_products",
            **check_directory_exists(str(static_root_path)),
        })

        for rel_path in REQUIRED_STATIC_PRODUCTS:
            checks.append({
                "check_type": "static_product_file",
                **check_file_exists(str(static_root_path / rel_path)),
            })

    result["checks"] = checks
    result["ok"] = all(item["ok"] for item in checks)
    return result
# === PATCH: static_products_root preflight END ===
