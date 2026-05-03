from __future__ import annotations

from datetime import datetime
from pathlib import Path

from models import RinexObsFile, BaselinePair


def _file_stem(path: Path) -> str:
    name = path.name

    for suffix in [".gz", ".Z"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return Path(name).stem


def _make_run_label(rover: RinexObsFile, base: RinexObsFile) -> str:
    rover_stem = _file_stem(rover.path)
    base_stem = _file_stem(base.path)
    return f"{rover_stem}__{base_stem}"


def _compute_overlap(
    rover: RinexObsFile,
    base: RinexObsFile,
) -> tuple[datetime, datetime, float] | None:
    if rover.first_obs is None or rover.last_obs is None:
        return None

    if base.first_obs is None or base.last_obs is None:
        return None

    overlap_start = max(rover.first_obs, base.first_obs)
    overlap_end = min(rover.last_obs, base.last_obs)

    if overlap_end <= overlap_start:
        return None

    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60.0

    return overlap_start, overlap_end, overlap_minutes


def match_rover_base_overlaps(
    rover_files: list[RinexObsFile],
    base_files: list[RinexObsFile],
    minimum_overlap_minutes: float = 45.0,
    matching_strategy: str = "best_overlap_per_rover",
) -> list[BaselinePair]:
    """
    Match rover and base/CORS RINEX OBS files by TIME OF FIRST OBS / TIME OF LAST OBS.

    matching_strategy:
        - best_overlap_per_rover
        - all_valid_overlaps
    """
    valid_pairs: list[BaselinePair] = []

    for rover in rover_files:
        for base in base_files:
            overlap = _compute_overlap(rover, base)

            if overlap is None:
                continue

            overlap_start, overlap_end, overlap_minutes = overlap

            if overlap_minutes < minimum_overlap_minutes:
                continue

            valid_pairs.append(
                BaselinePair(
                    run_label=_make_run_label(rover, base),
                    rover=rover,
                    base=base,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    overlap_minutes=overlap_minutes,
                    matching_status="accepted",
                )
            )

    matching_strategy = matching_strategy.strip().lower()

    if matching_strategy == "all_valid_overlaps":
        return sorted(
            valid_pairs,
            key=lambda p: (str(p.rover.path), str(p.base.path)),
        )

    if matching_strategy == "best_overlap_per_rover":
        best_by_rover: dict[str, BaselinePair] = {}

        for pair in valid_pairs:
            rover_key = str(pair.rover.path)

            current = best_by_rover.get(rover_key)

            if current is None or pair.overlap_minutes > current.overlap_minutes:
                best_by_rover[rover_key] = pair

        return sorted(
            best_by_rover.values(),
            key=lambda p: str(p.rover.path),
        )

    raise ValueError(
        "Unknown matching_strategy. "
        "Allowed values: 'best_overlap_per_rover', 'all_valid_overlaps'."
    )


def matches_to_rows(pairs: list[BaselinePair]) -> list[dict]:
    rows = []

    for pair in pairs:
        rows.append(
            {
                "run_label": pair.run_label,
                "rover_file": str(pair.rover.path),
                "base_file": str(pair.base.path),
                "rover_first_obs": pair.rover.first_obs,
                "rover_last_obs": pair.rover.last_obs,
                "base_first_obs": pair.base.first_obs,
                "base_last_obs": pair.base.last_obs,
                "overlap_start": pair.overlap_start,
                "overlap_end": pair.overlap_end,
                "overlap_minutes": pair.overlap_minutes,
                "matching_status": pair.matching_status,
            }
        )

    return rows
