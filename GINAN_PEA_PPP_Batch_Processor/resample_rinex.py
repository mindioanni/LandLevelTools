from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess


def _format_datetime(dt: datetime) -> str:
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def _parse_rinex_time_fields(path: str) -> dict:
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"RINEX file does not exist: {p}")

    interval_sec = None
    first_obs = None
    last_obs = None

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            label = line[60:].strip() if len(line) >= 60 else ""

            if label == "INTERVAL":
                try:
                    interval_sec = int(round(float(line[:60].split()[0])))
                except Exception:
                    interval_sec = None

            elif label == "TIME OF FIRST OBS":
                parts = line[:60].split()
                if len(parts) >= 6:
                    year, month, day, hour, minute = map(int, parts[:5])
                    sec = float(parts[5])
                    first_obs = datetime(year, month, day, hour, minute) + timedelta(seconds=sec)

            elif label == "TIME OF LAST OBS":
                parts = line[:60].split()
                if len(parts) >= 6:
                    year, month, day, hour, minute = map(int, parts[:5])
                    sec = float(parts[5])
                    last_obs = datetime(year, month, day, hour, minute) + timedelta(seconds=sec)

            elif "END OF HEADER" in line:
                break

    if first_obs is None:
        raise ValueError(f"Could not read TIME OF FIRST OBS from: {p}")

    if last_obs is None:
        raise ValueError(f"Could not read TIME OF LAST OBS from: {p}")

    return {
        "interval_sec": interval_sec,
        "time_first_obs": _format_datetime(first_obs),
        "time_last_obs": _format_datetime(last_obs),
    }


def _apply_effective_observation_span(ctx: dict, rinex_file: str) -> dict:
    header = _parse_rinex_time_fields(rinex_file)

    ctx["resampling"]["effective_interval_sec"] = header["interval_sec"]
    ctx["resampling"]["effective_first_obs"] = header["time_first_obs"]
    ctx["resampling"]["effective_last_obs"] = header["time_last_obs"]

    if "time_window" in ctx:
        ctx["time_window"]["start_epoch"] = header["time_first_obs"]
        ctx["time_window"]["end_epoch"] = header["time_last_obs"]

    return ctx


def plan_resampling(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    requested = config["user_inputs"]["requested_sample_rate_sec"]
    raw_interval = ctx["raw"]["raw_interval_sec"]

    if requested is None:
        raise ValueError("requested_sample_rate_sec is missing in config")

    if raw_interval is None:
        raise ValueError("raw_interval_sec is missing in dataset_context")

    if requested <= 0:
        raise ValueError("requested_sample_rate_sec must be positive")

    ctx["resampling"]["requested_interval_sec"] = requested

    if requested < raw_interval:
        raise ValueError(
            f"Requested sample rate ({requested}s) is smaller than raw interval ({raw_interval}s)"
        )

    if requested == raw_interval:
        ctx["resampling"]["resample_needed"] = False
        ctx["resampling"]["effective_interval_sec"] = raw_interval
    else:
        if requested % raw_interval != 0:
            raise ValueError(
                f"Requested sample rate ({requested}s) is not an integer multiple of raw interval ({raw_interval}s)"
            )
        ctx["resampling"]["resample_needed"] = True
        ctx["resampling"]["effective_interval_sec"] = requested

    return ctx


def _derive_resampled_filename(raw_filename: str, requested_interval_sec: int) -> str:
    p = Path(raw_filename)
    return f"{p.stem}_{requested_interval_sec}s{p.suffix}"


def _derive_temp_long_rinex_filename(
    marker_name: str,
    time_first_obs: str,
    requested_interval_sec: int,
) -> str:
    dt = datetime.strptime(time_first_obs, "%Y-%m-%d %H:%M:%S")

    marker4 = (marker_name.strip().upper()[:4] or "SITE").ljust(4, "X")
    doy = dt.strftime("%j")
    year = dt.strftime("%Y")
    hour = dt.strftime("%H")
    minute = dt.strftime("%M")

    return f"{marker4}00XXX_R_{year}{doy}{hour}{minute}_24H_{int(requested_interval_sec):02d}S_MO.rnx"


def build_resampled_paths(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    raw_dataset_dir = Path(ctx["raw"]["raw_dataset_dir"])
    raw_filename = ctx["raw"]["raw_rinex_filename"]
    requested = ctx["resampling"]["requested_interval_sec"]
    resample_needed = ctx["resampling"]["resample_needed"]

    if not raw_dataset_dir.name.lower().endswith(".rnx"):
        raise ValueError(f"Unexpected raw dataset directory name: {raw_dataset_dir.name}")

    if not resample_needed:
        ctx["resampling"]["resampled_dataset_dir"] = str(raw_dataset_dir)
        ctx["resampling"]["resampled_rinex_file"] = ctx["raw"]["raw_rinex_file"]
        ctx["resampling"]["resampled_rinex_filename"] = raw_filename
        ctx["resampling"]["temp_long_rinex_file"] = ""
        return ctx

    raw_root = Path(ctx["raw"].get("raw_root") or raw_dataset_dir.parent).expanduser().resolve()
    resampled_root = raw_root / config["processing"]["resampled_folder_name"]
    resampled_dataset_dir = resampled_root / raw_dataset_dir.name

    final_filename = _derive_resampled_filename(raw_filename, requested)
    temp_long_filename = _derive_temp_long_rinex_filename(
        marker_name=ctx["header"]["marker_name"],
        time_first_obs=ctx["header"]["time_first_obs"],
        requested_interval_sec=requested,
    )

    ctx["resampling"]["resampled_dataset_dir"] = str(resampled_dataset_dir)
    ctx["resampling"]["resampled_rinex_filename"] = final_filename
    ctx["resampling"]["resampled_rinex_file"] = str(resampled_dataset_dir / final_filename)
    ctx["resampling"]["temp_long_rinex_file"] = str(resampled_dataset_dir / temp_long_filename)

    return ctx


def run_resampling(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    gfzrnx_path = config["system"]["gfzrnx_path"]
    overwrite = config["user_inputs"]["overwrite"]

    resample_needed = ctx["resampling"]["resample_needed"]
    final_file = Path(ctx["resampling"]["resampled_rinex_file"])
    final_dir = Path(ctx["resampling"]["resampled_dataset_dir"])

    if not resample_needed:
        ctx["execution"]["resample_performed"] = False
        ctx = _apply_effective_observation_span(ctx, str(final_file))
        return {
            "ok": True,
            "skipped": True,
            "message": "Resampling not needed; raw file will be used directly.",
            "command": [],
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    temp_file = Path(ctx["resampling"]["temp_long_rinex_file"])
    raw_file = Path(ctx["raw"]["raw_rinex_file"])
    requested = ctx["resampling"]["requested_interval_sec"]

    final_dir.mkdir(parents=True, exist_ok=True)

    if final_file.exists() and not overwrite:
        ctx["execution"]["resample_performed"] = False
        ctx = _apply_effective_observation_span(ctx, str(final_file))
        return {
            "ok": True,
            "skipped": True,
            "message": f"Resampled file already exists and overwrite=False: {final_file}",
            "command": [],
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    if final_file.exists() and overwrite:
        final_file.unlink()

    if temp_file.exists():
        temp_file.unlink()

    command = [
        gfzrnx_path,
        "-finp", str(raw_file),
        "-fout", str(temp_file),
        "-smp", str(requested),
        "-sei", "out",
    ]

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return {
            "ok": False,
            "skipped": False,
            "message": "GFZRNX returned a non-zero exit code.",
            "command": command,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    if not temp_file.exists():
        return {
            "ok": False,
            "skipped": False,
            "message": f"GFZRNX completed but temporary output file was not created: {temp_file}",
            "command": command,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    shutil.move(str(temp_file), str(final_file))
    ctx["execution"]["resample_performed"] = True
    ctx = _apply_effective_observation_span(ctx, str(final_file))

    return {
        "ok": True,
        "skipped": False,
        "message": f"Resampling completed: {final_file}",
        "command": command,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output_file": str(final_file),
        "dataset_context": ctx,
    }

# === PATCH: compact RINEX resampling support START ===
def _is_compact_rinex_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return (
        name.endswith(".crx")
        or bool(__import__("re").search(r"\.\d{2}d$", name))
        or name.endswith(".d")
    )


def _standard_rinex_name_from_raw(raw_filename: str, requested_interval_sec: int) -> str:
    p = Path(raw_filename)
    name = p.name

    if name.lower().endswith(".gz"):
        name = name[:-3]

    lower = name.lower()

    if lower.endswith(".crx"):
        base = name[:-4]
        return f"{base}_{int(requested_interval_sec)}s.rnx"

    if __import__("re").search(r"\.\d{2}d$", lower):
        base = name[:-4]
        return f"{base}_{int(requested_interval_sec)}s.rnx"

    if lower.endswith(".d"):
        base = name[:-2]
        return f"{base}_{int(requested_interval_sec)}s.rnx"

    p2 = Path(name)
    return f"{p2.stem}_{int(requested_interval_sec)}s{p2.suffix}"


_resample_original_plan_resampling_compact = plan_resampling

def plan_resampling(config: dict, dataset_context: dict) -> dict:
    ctx = _resample_original_plan_resampling_compact(config, dataset_context)

    raw_file = ctx["raw"]["raw_rinex_file"]
    if _is_compact_rinex_file(raw_file):
        ctx["resampling"]["resample_needed"] = True
        ctx["resampling"]["compact_rinex_input"] = True

    return ctx


def build_resampled_paths(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    raw_dataset_dir = Path(ctx["raw"]["raw_dataset_dir"]).expanduser().resolve()
    raw_filename = ctx["raw"]["raw_rinex_filename"]
    requested = ctx["resampling"]["requested_interval_sec"]
    resample_needed = ctx["resampling"]["resample_needed"]

    raw_root = Path(ctx["raw"].get("raw_root") or raw_dataset_dir.parent).expanduser().resolve()
    resampled_root = raw_root / config["processing"]["resampled_folder_name"]
    resampled_dataset_dir = resampled_root / raw_dataset_dir.name

    if not resample_needed:
        ctx["resampling"]["resampled_dataset_dir"] = str(raw_dataset_dir)
        ctx["resampling"]["resampled_rinex_file"] = ctx["raw"]["raw_rinex_file"]
        ctx["resampling"]["resampled_rinex_filename"] = raw_filename
        ctx["resampling"]["temp_long_rinex_file"] = ""
        ctx["resampling"]["normalized_rinex_file"] = ""
        return ctx

    final_filename = _standard_rinex_name_from_raw(raw_filename, requested)
    temp_long_filename = _derive_temp_long_rinex_filename(
        marker_name=ctx["header"]["marker_name"],
        time_first_obs=ctx["header"]["time_first_obs"],
        requested_interval_sec=requested,
    )

    normalized_filename = Path(final_filename).with_suffix(".normalized.rnx").name

    ctx["resampling"]["resampled_dataset_dir"] = str(resampled_dataset_dir)
    ctx["resampling"]["resampled_rinex_filename"] = final_filename
    ctx["resampling"]["resampled_rinex_file"] = str(resampled_dataset_dir / final_filename)
    ctx["resampling"]["temp_long_rinex_file"] = str(resampled_dataset_dir / temp_long_filename)
    ctx["resampling"]["normalized_rinex_file"] = str(resampled_dataset_dir / normalized_filename)

    return ctx


def _decompress_compact_rinex_to_file(input_file: Path, output_file: Path) -> None:
    import hatanaka

    data = input_file.read_bytes()
    decompressed = hatanaka.decompress(data)

    if isinstance(decompressed, str):
        decompressed = decompressed.encode("utf-8")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(decompressed)


def run_resampling(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    gfzrnx_path = config["system"]["gfzrnx_path"]
    overwrite = config["user_inputs"]["overwrite"]

    resample_needed = ctx["resampling"]["resample_needed"]
    final_file = Path(ctx["resampling"]["resampled_rinex_file"]).expanduser().resolve()
    final_dir = Path(ctx["resampling"]["resampled_dataset_dir"]).expanduser().resolve()

    raw_file = Path(ctx["raw"]["raw_rinex_file"]).expanduser().resolve()
    requested = ctx["resampling"]["requested_interval_sec"]
    raw_interval = ctx["raw"]["raw_interval_sec"]

    compact_input = _is_compact_rinex_file(raw_file)

    if not resample_needed:
        ctx["execution"]["resample_performed"] = False
        ctx = _apply_effective_observation_span(ctx, str(final_file))
        return {
            "ok": True,
            "skipped": True,
            "message": "Resampling not needed; raw file will be used directly.",
            "command": [],
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    final_dir.mkdir(parents=True, exist_ok=True)

    if final_file.exists() and not overwrite:
        ctx["execution"]["resample_performed"] = False
        ctx = _apply_effective_observation_span(ctx, str(final_file))
        return {
            "ok": True,
            "skipped": True,
            "message": f"Resampled/normalized file already exists and overwrite=False: {final_file}",
            "command": [],
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    if final_file.exists() and overwrite:
        final_file.unlink()

    normalized_file = Path(ctx["resampling"].get("normalized_rinex_file") or final_dir / "normalized.rnx").expanduser().resolve()
    temp_file = Path(ctx["resampling"]["temp_long_rinex_file"]).expanduser().resolve()

    for p in [normalized_file, temp_file]:
        if p.exists():
            p.unlink()

    if compact_input:
        _decompress_compact_rinex_to_file(raw_file, normalized_file)
        gfzrnx_input = normalized_file
    else:
        gfzrnx_input = raw_file

    if compact_input and raw_interval == requested:
        shutil.copy2(normalized_file, final_file)
        ctx["execution"]["resample_performed"] = True
        ctx = _apply_effective_observation_span(ctx, str(final_file))
        return {
            "ok": True,
            "skipped": False,
            "message": "Compact RINEX decompressed to standard RINEX; GFZRNX resampling not needed.",
            "command": ["hatanaka.decompress", str(raw_file), str(final_file)],
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    command = [
        gfzrnx_path,
        "-finp", str(gfzrnx_input),
        "-fout", str(temp_file),
        "-smp", str(requested),
        "-sei", "out",
    ]

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return {
            "ok": False,
            "skipped": False,
            "message": "GFZRNX resampling failed",
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    if not temp_file.exists():
        return {
            "ok": False,
            "skipped": False,
            "message": f"GFZRNX finished but temp output file was not created: {temp_file}",
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "output_file": str(final_file),
            "dataset_context": ctx,
        }

    shutil.move(str(temp_file), str(final_file))

    ctx["execution"]["resample_performed"] = True
    ctx = _apply_effective_observation_span(ctx, str(final_file))

    return {
        "ok": True,
        "skipped": False,
        "message": "Resampling completed.",
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "output_file": str(final_file),
        "dataset_context": ctx,
    }
# === PATCH: compact RINEX resampling support END ===
