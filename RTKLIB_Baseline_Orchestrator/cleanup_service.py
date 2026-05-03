
from __future__ import annotations

from pathlib import Path
import shutil


PROTECTED_NAMES = {
    "bases.solution.report.html",
}


def list_generated_files(
    rtk_process_root: str | Path,
    remove_runs: bool = True,
    remove_logs: bool = True,
    remove_tables: bool = True,
    remove_report: bool = False,
) -> list[Path]:
    root = Path(rtk_process_root).expanduser().resolve()
    if not root.exists():
        return []

    targets = []

    if remove_runs:
        runs = root / "runs"
        if runs.exists():
            targets.append(runs)

    if remove_logs:
        logs = root / "logs"
        if logs.exists():
            targets.append(logs)

    if remove_tables:
        for pattern in ["*.csv", "*.json"]:
            targets.extend(root.glob(pattern))

    if remove_report:
        report = root / "bases.solution.report.html"
        if report.exists():
            targets.append(report)

    return sorted(set(targets))


def clean_generated_files(
    rtk_process_root: str | Path,
    execute: bool,
    remove_runs: bool = True,
    remove_logs: bool = True,
    remove_tables: bool = True,
    remove_report: bool = False,
) -> dict:
    targets = list_generated_files(
        rtk_process_root=rtk_process_root,
        remove_runs=remove_runs,
        remove_logs=remove_logs,
        remove_tables=remove_tables,
        remove_report=remove_report,
    )

    deleted = []
    errors = []

    if execute:
        for p in targets:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                deleted.append(p)
            except Exception as exc:
                errors.append(f"{p}: {exc}")

    return {
        "execute": execute,
        "targets": [str(p) for p in targets],
        "deleted": [str(p) for p in deleted],
        "errors": errors,
        "message": f"{len(deleted)} deleted; {len(errors)} errors" if execute else f"{len(targets)} targets found",
    }
