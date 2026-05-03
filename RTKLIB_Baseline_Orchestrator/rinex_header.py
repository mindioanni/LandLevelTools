from __future__ import annotations

from pathlib import Path
from datetime import datetime
import gzip
import re

from models import RinexObsFile


OBS_EXTENSIONS = {
    ".obs", ".rnx", ".o",
}


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def _parse_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def _parse_time_of_obs(line: str) -> datetime | None:
    parts = line[:43].split()
    if len(parts) < 6:
        return None

    try:
        year = int(float(parts[0]))
        month = int(float(parts[1]))
        day = int(float(parts[2]))
        hour = int(float(parts[3]))
        minute = int(float(parts[4]))
        sec_float = float(parts[5])

        sec = int(sec_float)
        micro = int(round((sec_float - sec) * 1_000_000))

        if micro >= 1_000_000:
            sec += 1
            micro -= 1_000_000

        return datetime(year, month, day, hour, minute, sec, micro)
    except Exception:
        return None


def _parse_rinex_header(path: Path) -> RinexObsFile | None:
    info = RinexObsFile(path=path, filename=path.name)

    file_type = ""

    try:
        with _open_text(path) as f:
            for line in f:
                label = line[60:].strip() if len(line) >= 60 else ""

                if "RINEX VERSION / TYPE" in line:
                    info.rinex_version = line[:20].strip()
                    file_type = line[20:21].strip().upper()

                elif "MARKER NAME" in line:
                    info.marker_name = line[:60].strip()

                elif "REC # / TYPE / VERS" in line:
                    info.receiver = line[:60].strip()

                elif "ANT # / TYPE" in line:
                    info.antenna = line[:60].strip()

                elif "APPROX POSITION XYZ" in line:
                    pass

                elif "ANTENNA: DELTA H/E/N" in line:
                    parts = line[:60].split()
                    if len(parts) >= 3:
                        info.antenna_delta_h_m = _parse_float(parts[0])
                        info.antenna_delta_e_m = _parse_float(parts[1])
                        info.antenna_delta_n_m = _parse_float(parts[2])

                elif "TIME OF FIRST OBS" in line:
                    info.first_obs = _parse_time_of_obs(line)

                elif "TIME OF LAST OBS" in line:
                    info.last_obs = _parse_time_of_obs(line)

                elif "INTERVAL" in line:
                    parts = line[:60].split()
                    if parts:
                        info.interval_sec = _parse_float(parts[0])

                elif "END OF HEADER" in line:
                    break

    except Exception:
        return None

    if file_type != "O":
        return None

    return info


def discover_rinex_obs(root: Path | str) -> list[RinexObsFile]:
    root = Path(root).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    candidates = []

    for path in root.rglob("*"):
        if not path.is_file:
            continue

        name = path.name.lower()

        # Common RINEX OBS names:
        # RINEX 2: *.YYo, *.YYO, optionally .gz
        # RINEX 3: *.rnx / *.obs, but must be confirmed from header as type O
        is_candidate = False

        if re.search(r"\.\d{2}o(\.gz)?$", name):
            is_candidate = True
        elif name.endswith((".obs", ".obs.gz", ".rnx", ".rnx.gz")):
            is_candidate = True
        elif name.endswith((".crx", ".crx.gz", ".d", ".d.gz")):
            # Compact/Hatanaka RINEX placeholder.
            # v0.1 does not decode CRX directly here.
            is_candidate = True

        if not is_candidate:
            continue

        parsed = _parse_rinex_header(path)
        if parsed is not None:
            candidates.append(parsed)

    return sorted(candidates, key=lambda x: str(x.path))


def write_inventory_csv(items: list[RinexObsFile], output_path: Path | str) -> Path:
    import pandas as pd

    output_path = Path(output_path)

    rows = []
    for item in items:
        rows.append({
            "path": str(item.path),
            "filename": item.filename,
            "marker_name": item.marker_name,
            "rinex_version": item.rinex_version,
            "first_obs": item.first_obs.isoformat() if item.first_obs else "",
            "last_obs": item.last_obs.isoformat() if item.last_obs else "",
            "interval_sec": item.interval_sec,
            "receiver": item.receiver,
            "antenna": item.antenna,
            "antenna_delta_h_m": item.antenna_delta_h_m,
            "antenna_delta_e_m": item.antenna_delta_e_m,
            "antenna_delta_n_m": item.antenna_delta_n_m,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
