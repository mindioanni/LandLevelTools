from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import re


def _empty_dataset_context() -> dict:
    return {
        "identity": {
            "dataset_name": "",
            "dataset_folder_name": "",
            "station_code": "",
        },
        "raw": {
            "raw_root": "",
            "raw_dataset_dir": "",
            "raw_rinex_file": "",
            "raw_rinex_filename": "",
            "raw_interval_sec": None,
        },
        "header": {
            "marker_name": "",
            "marker_number": "",
            "receiver_serial": "",
            "receiver_type": "",
            "receiver_version": "",
            "antenna_serial": "",
            "antenna_type": "",
            "approx_position_xyz": [None, None, None],
            "antenna_delta_hen": [None, None, None],
            "time_first_obs": "",
            "time_last_obs": "",
            "interval_sec": None,
            "obs_types": {},
        },
        "resampling": {
            "requested_interval_sec": None,
            "resample_needed": False,
            "resampled_dataset_dir": "",
            "resampled_rinex_file": "",
            "resampled_rinex_filename": "",
            "effective_interval_sec": None,
        },
        "time_window": {
            "start_epoch": "",
            "end_epoch": "",
            "covered_dates": [],
            "day_codes": [],
        },
        "products": {
            "igs_precise_dir": "",
            "snx_files": [],
            "nav_files": [],
            "erp_files": [],
            "clk_files": [],
            "bsx_files": [],
            "sp3_files": [],
        },
        "outputs": {
            "ginan_process_dir": "",
            "yaml_dir": "",
            "run_label": "",
            "run_dir": "",
            "yaml_path": "",
            "stdout_path": "",
            "manifest_path": "",
            "commands_path": "",
        },
        "execution": {
            "status": "PENDING",
            "message": "",
            "pea_exit_code": None,
            "resample_performed": False,
            "downloads_completed": False,
            "yaml_written": False,
            "pea_completed": False,
            "validation_passed": False,
        },
    }


def discover_raw_datasets(raw_root: str, limit: int | None = None) -> list[dict]:
    raw_root_path = Path(raw_root).expanduser().resolve()

    if not raw_root_path.exists():
        raise FileNotFoundError(f"RAW root does not exist: {raw_root_path}")
    if not raw_root_path.is_dir():
        raise NotADirectoryError(f"RAW root is not a directory: {raw_root_path}")

    candidate_items = []

    # Pass 1: classic one-level dataset directories.
    # Example: RAW_ROOT/2025_141/file.rnx
    for d in sorted(raw_root_path.iterdir()):
        if not d.is_dir():
            continue

        if d.name.lower() in _WORKSPACE_DIR_NAMES:
            continue

        try:
            raw_file = Path(find_observation_file(str(d))).expanduser().resolve()
        except FileNotFoundError:
            continue
        except RuntimeError:
            # If a directory contains many RINEX files, do not treat the directory
            # itself as one dataset. Let the file-based pass handle each file.
            continue

        dataset_name = _make_dataset_name(raw_root_path, d, raw_file)
        candidate_items.append((d, raw_file, dataset_name, d.name))

    # Pass 2: file-based discovery.
    # Handles:
    # - root-level daily files: RAW_ROOT/*.crx.gz
    # - nested files: RAW_ROOT/YYYY/ddd/*.??d.Z
    if not candidate_items:
        candidate_files = []

        for pattern in _rinex_candidate_patterns():
            candidate_files.extend(raw_root_path.rglob(pattern))

        candidate_files = sorted(
            p for p in candidate_files
            if p.is_file()
            and not _is_inside_workspace_dir(p, raw_root_path)
            and _is_observation_rinex_file(p)
        )

        seen_logical = set()

        for raw_file in candidate_files:
            raw_file = raw_file.expanduser().resolve()
            dataset_dir = raw_file.parent
            dataset_name = _make_dataset_name(raw_root_path, dataset_dir, raw_file)

            logical_key = (dataset_name, _strip_compression_suffix(raw_file).lower())
            if logical_key in seen_logical:
                continue

            candidate_items.append((dataset_dir, raw_file, dataset_name, raw_file.name))
            seen_logical.add(logical_key)

    candidate_items = sorted(candidate_items, key=lambda item: item[2])

    if limit is not None:
        candidate_items = candidate_items[:limit]

    contexts = []

    for dataset_dir, raw_file, dataset_name, dataset_folder_name in candidate_items:
        ctx = _empty_dataset_context()

        ctx["identity"]["dataset_name"] = dataset_name
        ctx["identity"]["dataset_folder_name"] = dataset_folder_name
        ctx["raw"]["raw_root"] = str(raw_root_path)
        ctx["raw"]["raw_dataset_dir"] = str(dataset_dir)
        ctx["raw"]["raw_rinex_file"] = str(raw_file)
        ctx["raw"]["raw_rinex_filename"] = raw_file.name
        ctx["raw"]["raw_is_compact_rinex"] = _is_compact_rinex_file(raw_file)
        ctx["raw"]["raw_is_gzip"] = raw_file.name.lower().endswith(".gz")
        ctx["raw"]["raw_is_unix_compressed"] = raw_file.name.lower().endswith(".z")

        contexts.append(ctx)

    return contexts
def find_observation_file(raw_dataset_dir: str) -> str:
    dataset_dir = Path(raw_dataset_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    candidate_patterns = [
        "*.??o", "*.??O", "*.rnx", "*.RNX", "*.crx", "*.CRX",
        "*.d", "*.D",
    ]

    matches = []
    for pattern in candidate_patterns:
        matches.extend(dataset_dir.glob(pattern))

    matches = sorted(p for p in matches if p.is_file())

    if not matches:
        raise FileNotFoundError(f"No observation file found in dataset directory: {dataset_dir}")

    if len(matches) > 1:
        preferred = [
            p for p in matches
            if re.search(r"\.(\d{2}[oO]|rnx|RNX|crx|CRX|d|D)$", p.name)
        ]
        if len(preferred) == 1:
            return str(preferred[0])
        if len(preferred) > 1:
            raise RuntimeError(
                f"Multiple observation files found in dataset directory: {dataset_dir} -> {[p.name for p in preferred]}"
            )

    return str(matches[0])


def _label_of(line: str) -> str:
    if len(line) >= 61:
        return line[60:].strip()
    return ""


def _parse_time_line(line: str) -> str:
    body = line[:60]
    parts = body.split()

    if len(parts) < 6:
        raise ValueError(f"Cannot parse RINEX time line: {line.rstrip()}")

    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    hour = int(parts[3])
    minute = int(parts[4])
    second = float(parts[5])

    sec_int = int(second)
    micro = int(round((second - sec_int) * 1_000_000))

    if micro == 1_000_000:
        sec_int += 1
        micro = 0

    dt = datetime(year, month, day, hour, minute, sec_int, micro)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_obs_types(header_lines: list[str]) -> dict:
    obs_types = {}
    i = 0

    while i < len(header_lines):
        line = header_lines[i]
        if _label_of(line) != "SYS / # / OBS TYPES":
            i += 1
            continue

        system = line[0].strip()
        body = line[:60]
        tokens = body.split()

        if len(tokens) < 2:
            i += 1
            continue

        total_count = int(tokens[1])
        current = tokens[2:]

        i += 1
        while len(current) < total_count and i < len(header_lines):
            next_line = header_lines[i]
            if _label_of(next_line) != "SYS / # / OBS TYPES":
                break
            current.extend(next_line[:60].split())
            i += 1

        obs_types[system] = current[:total_count]

    return obs_types


def parse_rinex_header(rinex_path: str) -> dict:
    rinex_file = Path(rinex_path).expanduser().resolve()

    if not rinex_file.exists():
        raise FileNotFoundError(f"RINEX file does not exist: {rinex_file}")
    if not rinex_file.is_file():
        raise FileNotFoundError(f"RINEX path is not a file: {rinex_file}")

    header_lines = []

    with rinex_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            header_lines.append(line.rstrip("\n"))
            if _label_of(line) == "END OF HEADER":
                break

    if not header_lines or _label_of(header_lines[-1]) != "END OF HEADER":
        raise ValueError(f"END OF HEADER not found in file: {rinex_file}")

    info = {
        "marker_name": "",
        "marker_number": "",
        "receiver_serial": "",
        "receiver_type": "",
        "receiver_version": "",
        "antenna_serial": "",
        "antenna_type": "",
        "approx_position_xyz": [None, None, None],
        "antenna_delta_hen": [None, None, None],
        "time_first_obs": "",
        "time_last_obs": "",
        "interval_sec": None,
        "obs_types": {},
    }

    for line in header_lines:
        label = _label_of(line)
        body = line[:60]

        if label == "MARKER NAME":
            info["marker_name"] = body.strip()

        elif label == "MARKER NUMBER":
            info["marker_number"] = body.strip()

        elif label == "REC # / TYPE / VERS":
            info["receiver_serial"] = body[:20].strip()
            info["receiver_type"] = body[20:40].strip()
            info["receiver_version"] = body[40:60].strip()

        elif label == "ANT # / TYPE":
            info["antenna_serial"] = body[:20].strip()
            info["antenna_type"] = body[20:40].strip()

        elif label == "APPROX POSITION XYZ":
            parts = body.split()
            if len(parts) >= 3:
                info["approx_position_xyz"] = [float(parts[0]), float(parts[1]), float(parts[2])]

        elif label == "ANTENNA: DELTA H/E/N":
            parts = body.split()
            if len(parts) >= 3:
                info["antenna_delta_hen"] = [float(parts[0]), float(parts[1]), float(parts[2])]

        elif label == "TIME OF FIRST OBS":
            info["time_first_obs"] = _parse_time_line(line)

        elif label == "TIME OF LAST OBS":
            info["time_last_obs"] = _parse_time_line(line)

        elif label == "INTERVAL":
            text = body.strip()
            if text:
                val = float(text)
                info["interval_sec"] = int(val) if val.is_integer() else val

    info["obs_types"] = _parse_obs_types(header_lines)

    return info


def enrich_dataset_with_header(dataset_context: dict, header_info: dict) -> dict:
    ctx = deepcopy(dataset_context)

    ctx["header"]["marker_name"] = header_info.get("marker_name", "")
    ctx["header"]["marker_number"] = header_info.get("marker_number", "")
    ctx["header"]["receiver_serial"] = header_info.get("receiver_serial", "")
    ctx["header"]["receiver_type"] = header_info.get("receiver_type", "")
    ctx["header"]["receiver_version"] = header_info.get("receiver_version", "")
    ctx["header"]["antenna_serial"] = header_info.get("antenna_serial", "")
    ctx["header"]["antenna_type"] = header_info.get("antenna_type", "")
    ctx["header"]["approx_position_xyz"] = header_info.get("approx_position_xyz", [None, None, None])
    ctx["header"]["antenna_delta_hen"] = header_info.get("antenna_delta_hen", [None, None, None])
    ctx["header"]["time_first_obs"] = header_info.get("time_first_obs", "")
    ctx["header"]["time_last_obs"] = header_info.get("time_last_obs", "")
    ctx["header"]["interval_sec"] = header_info.get("interval_sec", None)
    ctx["header"]["obs_types"] = header_info.get("obs_types", {})

    ctx["raw"]["raw_interval_sec"] = header_info.get("interval_sec", None)

    return ctx


def derive_covered_dates(time_first_obs: str, time_last_obs: str) -> tuple[list[str], list[str]]:
    """
    Derive covered calendar dates and YYYYDOY product day codes from RINEX
    first/last observation epochs.

    Some Compact RINEX / Hatanaka daily files, including HEPOS .YYd files,
    may contain TIME OF FIRST OBS but omit TIME OF LAST OBS. In that case,
    the file is treated as covering the calendar day of TIME OF FIRST OBS.
    This fallback is used only for date/product coverage, not as a claim
    that the last observation epoch is known from the header.
    """
    if not str(time_first_obs).strip():
        raise ValueError("time_first_obs is missing or empty")

    dt_first = datetime.strptime(str(time_first_obs).strip(), "%Y-%m-%d %H:%M:%S")

    if str(time_last_obs).strip():
        dt_last = datetime.strptime(str(time_last_obs).strip(), "%Y-%m-%d %H:%M:%S")
    else:
        dt_last = dt_first

    if dt_last < dt_first:
        raise ValueError("time_last_obs is earlier than time_first_obs")

    covered_dates = []
    day_codes = []

    current_date = dt_first.date()
    last_date = dt_last.date()

    while current_date <= last_date:
        covered_dates.append(current_date.strftime("%Y-%m-%d"))
        day_codes.append(current_date.strftime("%Y%j"))
        current_date += timedelta(days=1)

    return covered_dates, day_codes

# === PATCH: compact RINEX / YYYY_DDD discovery support START ===
import gzip as _gzip_compact_rinex


_WORKSPACE_DIR_NAMES = {
    "igs_precise",
    "resampled",
    "yaml",
    "ginan_process",
    "__pycache__",
}


def _strip_compression_suffix(path: Path) -> str:
    name = Path(path).name
    lower = name.lower()

    if lower.endswith(".gz"):
        return name[:-3]

    if lower.endswith(".z"):
        return name[:-2]

    return name


def _is_compact_rinex_file(path: str | Path) -> bool:
    name = _strip_compression_suffix(Path(path)).lower()

    return (
        name.endswith(".crx")
        or bool(re.search(r"\.\d{2}d$", name))
        or name.endswith(".d")
    )


def _is_observation_rinex_file(path: str | Path) -> bool:
    name = _strip_compression_suffix(Path(path)).lower()

    return (
        name.endswith(".rnx")
        or name.endswith(".crx")
        or bool(re.search(r"\.\d{2}[od]$", name))
        or name.endswith(".o")
        or name.endswith(".d")
    )


def _is_inside_workspace_dir(path: Path, raw_root: Path) -> bool:
    try:
        rel_parts = path.relative_to(raw_root).parts
    except ValueError:
        rel_parts = path.parts

    return any(part.lower() in _WORKSPACE_DIR_NAMES for part in rel_parts)


def _read_rinex_text_for_header(rinex_path: str) -> str:
    p = Path(rinex_path).expanduser().resolve()

    if _is_compact_rinex_file(p):
        import hatanaka

        data = p.read_bytes()
        decompressed = hatanaka.decompress(data)

        if isinstance(decompressed, bytes):
            return decompressed.decode("utf-8", errors="ignore")

        return str(decompressed)

    if p.name.lower().endswith(".gz"):
        with _gzip_compact_rinex.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return p.read_text(encoding="utf-8", errors="ignore")


def _rinex_candidate_patterns() -> list[str]:
    return [
        "*.??o", "*.??O", "*.rnx", "*.RNX",
        "*.??o.gz", "*.??O.gz", "*.rnx.gz", "*.RNX.gz",
        "*.??o.Z", "*.??O.Z", "*.rnx.Z", "*.RNX.Z",

        "*.crx", "*.CRX", "*.??d", "*.??D", "*.d", "*.D",
        "*.crx.gz", "*.CRX.gz", "*.??d.gz", "*.??D.gz", "*.d.gz", "*.D.gz",
        "*.crx.Z", "*.CRX.Z", "*.??d.Z", "*.??D.Z", "*.d.Z", "*.D.Z",
    ]


def _dataset_name_from_file(raw_file: Path) -> str:
    name = raw_file.name

    # RINEX 3 long filename, e.g.
    # DELO00GRC_R_20251410000_01D_30S_MO.crx.gz -> 2025_141
    m = re.search(r"(19\d{2}|20\d{2})(\d{3})\d{4}", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    stripped = _strip_compression_suffix(raw_file)

    # RINEX 2 short filename, e.g.
    # sant0010.20d -> 2020_001
    m = re.search(r"^[a-zA-Z0-9]{4}(\d{3})\d\.(\d{2})[odOD]$", stripped)
    if m:
        yy = int(m.group(2))
        yyyy = 2000 + yy if yy < 80 else 1900 + yy
        return f"{yyyy}_{m.group(1)}"

    stem = stripped
    for suffix in [".crx", ".CRX", ".rnx", ".RNX"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def _make_dataset_name(raw_root_path: Path, dataset_dir: Path, raw_file: Path | None = None) -> str:
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    try:
        rel_dir = dataset_dir.relative_to(raw_root_path)
        dir_parts = rel_dir.parts
    except ValueError:
        dir_parts = dataset_dir.parts

    # Nested YYYY/ddd layout.
    if len(dir_parts) >= 2 and re.fullmatch(r"\d{4}", dir_parts[0]) and re.fullmatch(r"\d{3}", dir_parts[1]):
        return f"{dir_parts[0]}_{dir_parts[1]}"

    if raw_file is not None:
        return _dataset_name_from_file(Path(raw_file))

    name = dataset_dir.name
    if name.lower().endswith(".rnx"):
        name = name[:-4]

    return name


def discover_raw_datasets(raw_root: str, limit: int | None = None) -> list[dict]:
    raw_root_path = Path(raw_root).expanduser().resolve()

    if not raw_root_path.exists():
        raise FileNotFoundError(f"RAW root does not exist: {raw_root_path}")
    if not raw_root_path.is_dir():
        raise NotADirectoryError(f"RAW root is not a directory: {raw_root_path}")

    candidate_items = []

    # Pass 1: classic one-level dataset directories.
    # Example: RAW_ROOT/2025_141/file.rnx
    for d in sorted(raw_root_path.iterdir()):
        if not d.is_dir():
            continue

        if d.name.lower() in _WORKSPACE_DIR_NAMES:
            continue

        try:
            raw_file = Path(find_observation_file(str(d))).expanduser().resolve()
        except FileNotFoundError:
            continue
        except RuntimeError:
            # Directory contains multiple RINEX files; handle them as separate
            # datasets in the file-based pass below.
            continue

        dataset_name = _make_dataset_name(raw_root_path, d, raw_file)
        candidate_items.append((d, raw_file, dataset_name, d.name))

    # Pass 2: file-based discovery.
    # Handles:
    # - root-level daily files: RAW_ROOT/*.crx.gz
    # - nested files: RAW_ROOT/YYYY/ddd/*.??d.Z
    if not candidate_items:
        candidate_files = []

        for pattern in _rinex_candidate_patterns():
            candidate_files.extend(raw_root_path.rglob(pattern))

        candidate_files = sorted(
            f.expanduser().resolve()
            for f in candidate_files
            if f.is_file()
            and not _is_inside_workspace_dir(f, raw_root_path)
            and _is_observation_rinex_file(f)
        )

        seen = set()

        for raw_file in candidate_files:
            dataset_dir = raw_file.parent
            dataset_name = _make_dataset_name(raw_root_path, dataset_dir, raw_file)

            if dataset_name in seen:
                continue

            candidate_items.append((dataset_dir, raw_file, dataset_name, raw_file.name))
            seen.add(dataset_name)

    candidate_items = sorted(candidate_items, key=lambda item: item[2])

    if limit is not None:
        candidate_items = candidate_items[:limit]

    contexts = []

    for dataset_dir, raw_file, dataset_name, dataset_folder_name in candidate_items:
        ctx = _empty_dataset_context()

        ctx["identity"]["dataset_name"] = dataset_name
        ctx["identity"]["dataset_folder_name"] = dataset_folder_name
        ctx["raw"]["raw_root"] = str(raw_root_path)
        ctx["raw"]["raw_dataset_dir"] = str(dataset_dir)
        ctx["raw"]["raw_rinex_file"] = str(raw_file)
        ctx["raw"]["raw_rinex_filename"] = raw_file.name
        ctx["raw"]["raw_is_compact_rinex"] = _is_compact_rinex_file(raw_file)
        ctx["raw"]["raw_is_gzip"] = raw_file.name.lower().endswith(".gz")
        ctx["raw"]["raw_is_unix_compressed"] = raw_file.name.lower().endswith(".z")

        contexts.append(ctx)

    return contexts


def find_observation_file(raw_dataset_dir: str) -> str:
    dataset_dir = Path(raw_dataset_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    matches = []
    for pattern in _rinex_candidate_patterns():
        matches.extend(dataset_dir.glob(pattern))

    matches = sorted(p for p in matches if p.is_file() and _is_observation_rinex_file(p))

    if not matches:
        raise FileNotFoundError(f"No observation file found in dataset directory: {dataset_dir}")

    def logical_name(p: Path) -> str:
        return _strip_compression_suffix(p).lower()

    logical_groups = {}
    for p in matches:
        logical_groups.setdefault(logical_name(p), []).append(p)

    if len(logical_groups) > 1:
        raise RuntimeError(
            f"Multiple observation files found in dataset directory: {dataset_dir} -> {[p.name for p in matches]}"
        )

    group = list(logical_groups.values())[0]

    uncompressed = [
        p for p in group
        if not p.name.lower().endswith(".gz") and not p.name.lower().endswith(".z")
    ]
    if uncompressed:
        return str(sorted(uncompressed)[0])

    gzip_files = [p for p in group if p.name.lower().endswith(".gz")]
    if gzip_files:
        return str(sorted(gzip_files)[0])

    return str(sorted(group)[0])


def parse_rinex_header(rinex_path: str) -> dict:
    rinex_file = Path(rinex_path).expanduser().resolve()

    if not rinex_file.exists():
        raise FileNotFoundError(f"RINEX file does not exist: {rinex_file}")

    text = _read_rinex_text_for_header(str(rinex_file))

    header_lines = []
    for line in text.splitlines():
        if len(line) < 80:
            line = line.rstrip("\n").ljust(80)
        header_lines.append(line)
        if "END OF HEADER" in line:
            break

    if not any("END OF HEADER" in line for line in header_lines):
        raise ValueError(f"END OF HEADER not found in RINEX file: {rinex_file}")

    info = {
        "marker_name": "",
        "marker_number": "",
        "receiver_serial": "",
        "receiver_type": "",
        "receiver_version": "",
        "antenna_serial": "",
        "antenna_type": "",
        "approx_position_xyz": [None, None, None],
        "antenna_delta_hen": [None, None, None],
        "time_first_obs": "",
        "time_last_obs": "",
        "interval_sec": None,
        "obs_types": _parse_obs_types(header_lines),
    }

    for line in header_lines:
        label = _label_of(line)
        body = line[:60]

        if label == "MARKER NAME":
            info["marker_name"] = body.strip()

        elif label == "MARKER NUMBER":
            info["marker_number"] = body.strip()

        elif label == "REC # / TYPE / VERS":
            info["receiver_serial"] = body[0:20].strip()
            info["receiver_type"] = body[20:40].strip()
            info["receiver_version"] = body[40:60].strip()

        elif label == "ANT # / TYPE":
            info["antenna_serial"] = body[0:20].strip()
            info["antenna_type"] = body[20:40].strip()

        elif label == "APPROX POSITION XYZ":
            parts = body.split()
            if len(parts) >= 3:
                info["approx_position_xyz"] = [float(parts[0]), float(parts[1]), float(parts[2])]

        elif label == "ANTENNA: DELTA H/E/N":
            parts = body.split()
            if len(parts) >= 3:
                info["antenna_delta_hen"] = [float(parts[0]), float(parts[1]), float(parts[2])]

        elif label == "TIME OF FIRST OBS":
            info["time_first_obs"] = _parse_time_line(line)

        elif label == "TIME OF LAST OBS":
            info["time_last_obs"] = _parse_time_line(line)

        elif label == "INTERVAL":
            parts = body.split()
            if parts:
                try:
                    value = float(parts[0])
                    info["interval_sec"] = int(round(value))
                except Exception:
                    info["interval_sec"] = None

    return info
# === PATCH: compact RINEX / YYYY_DDD discovery support END ===
