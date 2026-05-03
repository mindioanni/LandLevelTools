
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from models import RunConfig, RunResult, UserInputs


def run_rnx2rtkp(run_config: RunConfig, inputs: UserInputs) -> RunResult:
    stdout_path = run_config.run_dir / "stdout.txt"
    stderr_path = run_config.run_dir / "stderr.txt"

    if run_config.output_pos_path.exists() and not inputs.overwrite_existing_outputs:
        return RunResult(
            run_label=run_config.run_label,
            status="SKIPPED_EXISTING",
            exit_code=None,
            output_pos_path=run_config.output_pos_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command_path=run_config.command_path,
            warnings=["Output exists and overwrite_existing_outputs=False."],
        )

    if inputs.execution_mode == "build_only":
        return RunResult(
            run_label=run_config.run_label,
            status="BUILT_ONLY",
            exit_code=None,
            output_pos_path=run_config.output_pos_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command_path=run_config.command_path,
        )

    t0 = time.time()
    proc = subprocess.run(
        run_config.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.time() - t0

    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")

    errors = []
    warnings = []

    if proc.returncode != 0:
        errors.append(f"rnx2rtkp exited with code {proc.returncode}")

    if not run_config.output_pos_path.exists() or run_config.output_pos_path.stat().st_size == 0:
        errors.append("Output POS file was not created or is empty.")

    status = "SUCCESS" if not errors else "FAILED"

    return RunResult(
        run_label=run_config.run_label,
        status=status,
        exit_code=proc.returncode,
        output_pos_path=run_config.output_pos_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        command_path=run_config.command_path,
        processing_duration_sec=duration,
        warnings=warnings,
        errors=errors,
    )
