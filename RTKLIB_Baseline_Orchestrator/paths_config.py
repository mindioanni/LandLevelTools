
from __future__ import annotations

from pathlib import Path

APP_NAME = "RTKLIB Baseline Orchestrator"
VERSION = "0.1.0"

OUTPUT_FOLDER_NAME = "RTK_process"
REPORT_FILENAME = "bases.solution.report.html"

PRODUCT_PROVIDERS = ["EMR", "COD", "WUM", "IGS", "GFZ", "GRG"]
PRODUCT_SERIES = ["FIN", "RAP"]
PRODUCT_PROJECTS = ["MGX", "OPS"]
PRODUCT_MODES = ["broadcast", "precise"]

PROCESSING_MODES = ["static", "dynamic"]
MATCHING_STRATEGIES = ["best_overlap_per_rover", "all_valid_overlaps"]
SOLUTION_TYPES = ["forward", "backward", "combined"]
FREQUENCY_MODES = ["L1", "L1+L2", "L1+L2+L5"]
AR_MODES = ["continuous", "instantaneous", "fix-and-hold", "off"]
OUTPUT_FORMATS = ["ECEF XYZ", "LLH", "ENU baseline"]
EXECUTION_MODES = ["run", "build_only"]

DEFAULTS = {
    "product_provider": "COD",
    "product_series": "FIN",
    "product_project": "MGX",
    "product_mode": "precise",
    "processing_mode": "static",
    "minimum_overlap_minutes": 45.0,
    "matching_strategy": "best_overlap_per_rover",
    "frequency_mode": "L1+L2",
    "elevation_mask_deg": 15.0,
    "solution_type": "forward",
    "ambiguity_mode": "continuous",
    "ambiguity_threshold": 3.0,
    "nav_systems": ["G", "E", "C"],
    "output_coordinate_format": "ECEF XYZ",
    "final_window_minutes": 30.0,
    "recommended_min_final_window_minutes": 15.0,
    "q_fixed_only_for_final": True,
    "min_fixed_percent": 80.0,
    "min_ratio_for_fixed": 3.0,
    "generate_plots": True,
    "download_missing_products": True,
    "use_ionex": False,
    "use_antex": True,
    "use_blq": False,
    "use_bia_osb": False,
    "execution_mode": "run",
    "generate_report": True,
    "report_filename": REPORT_FILENAME,
    "save_run_conf": True,
    "save_run_command": True,
    "trace_level": 0,
    "overwrite_existing_outputs": False,
}

REFERENCE_RNX2RTKP_PATH = (
    Path.home()
    / "apps"
    / "rtklib_native"
    / "RTKLIB_2.4.2_p13"
    / "app"
    / "rnx2rtkp"
    / "gcc"
    / "rnx2rtkp"
)

REFERENCE_DOWNLOADER_SCRIPT = Path.home() / "data" / "RINEX" / "ginan_batch_PPP" / "auto_download_PPP.py"
REFERENCE_DOWNLOADER_PYTHON = Path.home() / "data" / "RINEX" / "ginan_batch_PPP" / "ginanenv" / "bin" / "python"


def get_default_rnx2rtkp_path() -> str:
    return str(REFERENCE_RNX2RTKP_PATH)


def get_default_downloader_script_path() -> str:
    return str(REFERENCE_DOWNLOADER_SCRIPT)


def get_default_downloader_python_path() -> str:
    return str(REFERENCE_DOWNLOADER_PYTHON)


def output_root_from_rover_root(rover_root: str | Path) -> Path:
    return Path(rover_root).expanduser().resolve() / OUTPUT_FOLDER_NAME
