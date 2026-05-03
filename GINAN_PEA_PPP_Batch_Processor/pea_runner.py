from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess


def build_pea_command(config: dict, dataset_context: dict) -> list[str]:
    pea_path = config["system"]["pea_path"]
    yaml_path = dataset_context["outputs"]["yaml_path"]

    if not pea_path:
        raise ValueError("pea_path is missing in config")
    if not yaml_path:
        raise ValueError("yaml_path is missing in dataset_context")

    return [pea_path, "--config", yaml_path]


def run_pea(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    command = build_pea_command(config, ctx)

    run_dir = Path(ctx["outputs"]["run_dir"]).expanduser().resolve()
    stdout_path = Path(ctx["outputs"]["stdout_path"]).expanduser().resolve()
    yaml_path = Path(ctx["outputs"]["yaml_path"]).expanduser().resolve()

    if not yaml_path.exists():
        return {
            "ok": False,
            "exit_code": None,
            "command": command,
            "stdout_path": str(stdout_path),
            "message": f"YAML file does not exist: {yaml_path}",
            "dataset_context": ctx,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8", errors="ignore") as f:
        proc = subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    ctx["execution"]["pea_completed"] = True
    ctx["execution"]["pea_exit_code"] = proc.returncode

    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "command": command,
        "stdout_path": str(stdout_path),
        "message": "PEA execution completed" if proc.returncode == 0 else "PEA execution failed",
        "dataset_context": ctx,
    }


def detect_early_stop(stdout_path: str) -> dict:
    p = Path(stdout_path).expanduser().resolve()

    if not p.exists():
        return {
            "ok": False,
            "early_stop": None,
            "message": f"stdout file does not exist: {p}",
        }

    txt = p.read_text(encoding="utf-8", errors="ignore")

    explicit_epoch1_stop = "Inputs finished at epoch #1" in txt
    no_more_data = "No more data available" in txt
    processed_epochs = "Processed epoch -" in txt

    early_stop = explicit_epoch1_stop or (no_more_data and not processed_epochs)

    return {
        "ok": True,
        "early_stop": early_stop,
        "message": "Early stop detected at epoch #1" if early_stop else "No early stop detected",
    }
