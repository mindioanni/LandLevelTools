
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from models import UserInputs


def _exists_file(path: Path | None) -> bool:
    return path is not None and Path(path).expanduser().exists() and Path(path).expanduser().is_file()


def _exists_dir(path: Path | None) -> bool:
    return path is not None and Path(path).expanduser().exists() and Path(path).expanduser().is_dir()


def check_rnx2rtkp(path: Path) -> tuple[bool, str]:
    path = Path(path).expanduser()
    if not path.exists() or not path.is_file():
        return False, f"rnx2rtkp executable not found: {path}"

    try:
        proc = subprocess.run(
            [str(path), "-?"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return False, f"rnx2rtkp help check failed: {exc}"

    text = proc.stdout or ""
    if "usage: rnx2rtkp" not in text:
        return False, "rnx2rtkp help output did not contain expected usage line."

    return True, text.splitlines()[0] if text.splitlines() else "rnx2rtkp help OK"


def run_system_check(inputs: UserInputs) -> dict:
    errors = []
    warnings = []

    for label, path in [
        ("CORS GINAN report HTML", inputs.cors_solution_report_path),
        ("rnx2rtkp executable", inputs.rnx2rtkp_path),
    ]:
        if not _exists_file(path):
            errors.append(f"{label} does not exist or is not a file: {path}")

    for label, path in [
        ("Rover GNSSBM RINEX folder", inputs.rover_rinex_root),
        ("CORS/base RINEX folder", inputs.base_rinex_root),
    ]:
        if not _exists_dir(path):
            errors.append(f"{label} does not exist or is not a directory: {path}")

    products_root = Path(inputs.products_root).expanduser()
    if not products_root.exists():
        try:
            products_root.mkdir(parents=True, exist_ok=True)
            warnings.append(f"Products root did not exist and was created: {products_root}")
        except Exception as exc:
            errors.append(f"Products root could not be created: {products_root} ({exc})")
    elif not products_root.is_dir():
        errors.append(f"Products root is not a directory: {products_root}")

    try:
        Path(inputs.output_root).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Output root could not be created: {inputs.output_root} ({exc})")

    if inputs.download_missing_products:
        if not _exists_file(inputs.downloader_script_path):
            warnings.append(f"Downloader script not found; missing products will not be downloaded: {inputs.downloader_script_path}")
        if not _exists_file(inputs.downloader_python_path):
            warnings.append(f"Downloader Python not found; missing products will not be downloaded: {inputs.downloader_python_path}")

    if _exists_file(inputs.rnx2rtkp_path):
        ok, message = check_rnx2rtkp(inputs.rnx2rtkp_path)
        if not ok:
            errors.append(message)
        else:
            if message:
                warnings.append(message)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
