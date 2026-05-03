from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


def _recursive_update(base: dict, updates: dict) -> dict:
    """Recursively update nested dictionaries without mutating inputs."""
    result = deepcopy(base)

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _recursive_update(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> dict:
    """
    Return the default global configuration for the PPP batch orchestrator.

    Notes
    -----
    - `python_path` defaults to the Python interpreter of the currently running
      Jupyter kernel (`sys.executable`).
    - If the user's system has different paths for PEA, GFZRNX, template YAML,
      or static products, they should edit the corresponding lines below.
    """
    python_path = sys.executable

    return {
        "system": {
            "pea_path": str(Path.home() / "opt" / "ginan" / "ginan-gui-linux-x64" / "_internal" / "bin" / "pea"),
            "gfzrnx_path": str(Path.home() / ".local" / "bin" / "gfzrnx"),
            "python_path": python_path,
            "downloader_python_path": str(Path.home() / "data" / "RINEX" / "ginan_batch_PPP" / "ginanenv" / "bin" / "python"),
            "auto_download_script": str(Path.home() / "data" / "RINEX" / "ginan_batch_PPP" / "auto_download_PPP.py"),
            "template_yaml_path": str(Path.home() / "opt" / "ginan" / "ginan-gui-linux-x64" / "_internal" / "scripts" / "GinanUI" / "app" / "resources" / "ppp_TG_GEIN_NOA_template.yaml"),
            "static_products_root": str(Path.home() / "opt" / "ginan" / "ginan-gui-linux-x64" / "_internal" / "scripts" / "GinanUI" / "app" / "resources" / "inputData" / "products"),
        },
        "user_inputs": {
            "raw_root": "",
            "requested_sample_rate_sec": 15,
            "provider": "COD",
            "series": "FIN",
            "project": "MGX",
            "execution_mode": "run",
            "overwrite": False,
            "limit_datasets": None,
        },
        "processing": {
            "resampled_folder_name": "RESAMPLED",
            "igs_precise_folder_name": "IGS_PRECISE",
            "ginan_process_folder_name": "GINAN_process",
            "yaml_subfolder_name": "yaml",
            "stdout_prefix": "stdout_",
            "manifest_filename": "run_manifest.json",
            "commands_filename": "run_commands.sh",
            "batch_summary_filename": "batch_summary.csv",
        },
        "download_policy": {
            "snx_mode": "IGS_OPS",
            "nav_mode": "BRDC_IGS",
            "sp3clk_mode": "PROVIDER_PROJECT_SERIES",
            "bia_mode": "PROVIDER_PROJECT_SERIES",
            "dont_replace": True,
        },
        "validation": {
            "require_pos": True,
            "require_smoothed_pos": True,
            "require_trace": True,
            "fail_on_epoch1_stop": True,
        },
    }



def get_work_paths(raw_root: str, config: dict) -> dict:
    # Return the standard processing layout derived only from user RAW_ROOT.
    # New convention: generated products are written inside RAW_ROOT,
    # independently of any external station/GNSS folder hierarchy.
    raw_root_path = Path(raw_root).expanduser().resolve()

    return {
        "raw_root": str(raw_root_path),
        "igs_precise_dir": str(raw_root_path / config["processing"]["igs_precise_folder_name"]),
        "resampled_dir": str(raw_root_path / config["processing"]["resampled_folder_name"]),
        "yaml_dir": str(raw_root_path / config["processing"]["yaml_subfolder_name"]),
        "ginan_process_dir": str(raw_root_path / config["processing"]["ginan_process_folder_name"]),
    }

def merge_user_inputs(base_config: dict, user_inputs: dict) -> dict:
    """
    Merge user-provided values into the base configuration.
    """
    if not isinstance(base_config, dict):
        raise TypeError("base_config must be a dictionary")

    if not isinstance(user_inputs, dict):
        raise TypeError("user_inputs must be a dictionary")

    return _recursive_update(base_config, user_inputs)


# === PATCH: static_products_root support START ===
# Adds user-defined static Ginan auxiliary products directory to paths_config.
# The value is intentionally not hardcoded. It must be set by the CLI/GUI/user workflow.
from pathlib import Path as _PathForStaticProducts

_paths_config_original_get_default_config = get_default_config

def get_default_config() -> dict:
    cfg = _paths_config_original_get_default_config()

    cfg.setdefault("system", {})

    # User-defined directory containing static Ginan auxiliary products.
    # Typical location, but not assumed:
    # ~/ginan-gui-linux-x64/_internal/scripts/GinanUI/app/resources/inputData/products
    cfg["system"].setdefault("static_products_root", "")

    # Hint only. The program must not assume that this path exists.
    cfg["system"].setdefault(
        "static_products_root_hint",
        str(
            _PathForStaticProducts.home()
            / "ginan-gui-linux-x64"
            / "_internal"
            / "scripts"
            / "GinanUI"
            / "app"
            / "resources"
            / "inputData"
            / "products"
        ),
    )

    return cfg
# === PATCH: static_products_root support END ===

# === PATCH: static_products_root merge START ===
_paths_config_original_merge_user_inputs_static_products = merge_user_inputs

def merge_user_inputs(base_config: dict, user_inputs: dict) -> dict:
    cfg = _paths_config_original_merge_user_inputs_static_products(base_config, user_inputs)

    static_root = None

    if isinstance(user_inputs, dict):
        if "static_products_root" in user_inputs:
            static_root = user_inputs.get("static_products_root")
        elif isinstance(user_inputs.get("system"), dict):
            static_root = user_inputs["system"].get("static_products_root")

    if static_root is not None:
        cfg.setdefault("system", {})
        cfg["system"]["static_products_root"] = str(static_root).strip()

    return cfg
# === PATCH: static_products_root merge END ===
