
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Any


@dataclass
class UserInputs:
    project_name: str
    cors_solution_report_path: Path
    rover_rinex_root: Path
    base_rinex_root: Path
    products_root: Path
    output_root: Path
    rnx2rtkp_path: Path

    product_provider: str = "COD"
    product_series: str = "FIN"
    product_project: str = "MGX"
    product_mode: str = "precise"
    download_missing_products: bool = True
    downloader_script_path: Path | None = None
    downloader_python_path: Path | None = None
    use_ionex: bool = False
    use_antex: bool = True
    use_blq: bool = False
    use_bia_osb: bool = False

    processing_mode: str = "static"
    minimum_overlap_minutes: float = 45.0
    matching_strategy: str = "best_overlap_per_rover"
    overwrite_existing_outputs: bool = False

    frequency_mode: str = "L1+L2"
    elevation_mask_deg: float = 15.0
    solution_type: str = "forward"
    ambiguity_mode: str = "continuous"
    ambiguity_threshold: float = 3.0
    nav_systems: list[str] = field(default_factory=lambda: ["G", "E", "C"])
    output_coordinate_format: str = "ECEF XYZ"

    final_window_minutes: float = 30.0
    recommended_min_final_window_minutes: float = 15.0
    q_fixed_only_for_final: bool = True
    min_fixed_percent: float = 80.0
    min_ratio_for_fixed: float = 3.0
    min_continuous_fixed_duration_sec: float = 300.0
    generate_plots: bool = True

    execution_mode: str = "run"
    generate_report: bool = True
    report_filename: str = "bases.solution.report.html"
    save_run_conf: bool = True
    save_run_command: bool = True
    trace_level: int = 0


@dataclass
class CorsSolution:
    station_id: str
    X_m: float
    Y_m: float
    Z_m: float
    std_X_m: float | None = None
    std_Y_m: float | None = None
    std_Z_m: float | None = None
    n_solutions: int | None = None
    source_report_path: Path | None = None
    final_table: Any | None = None
    daily_table: Any | None = None


@dataclass
class RinexObsFile:
    path: Path
    filename: str
    marker_name: str = ""
    rinex_version: str = ""
    first_obs: datetime | None = None
    last_obs: datetime | None = None
    interval_sec: float | None = None
    receiver: str = ""
    antenna: str = ""
    antenna_delta_h_m: float | None = None
    antenna_delta_e_m: float | None = None
    antenna_delta_n_m: float | None = None
    duration_minutes: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BaselinePair:
    run_label: str
    rover: RinexObsFile
    base: RinexObsFile
    overlap_start: datetime
    overlap_end: datetime
    overlap_minutes: float
    matching_status: str = "ACCEPTED"


@dataclass
class ResolvedProducts:
    run_label: str
    nav_files: list[Path] = field(default_factory=list)
    sp3_files: list[Path] = field(default_factory=list)
    clk_files: list[Path] = field(default_factory=list)
    ionex_files: list[Path] = field(default_factory=list)
    antex_file: Path | None = None
    blq_file: Path | None = None
    bia_files: list[Path] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    downloaded_files: list[Path] = field(default_factory=list)
    product_status: str = "UNKNOWN"


@dataclass
class RunConfig:
    run_label: str
    run_dir: Path
    conf_path: Path
    output_pos_path: Path
    command_path: Path
    command: list[str]


@dataclass
class RunResult:
    run_label: str
    status: str
    exit_code: int | None
    output_pos_path: Path
    stdout_path: Path
    stderr_path: Path
    command_path: Path
    processing_duration_sec: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ParsedPos:
    path: Path
    header: dict[str, str]
    dataframe: Any


@dataclass
class BaselineSolution:
    run_label: str
    benchmark_id: str
    rover_file: Path
    base_file: Path
    solution_method: str
    final_window_minutes: float
    X_m: float | None
    Y_m: float | None
    Z_m: float | None
    lon_deg: float | None
    lat_deg: float | None
    h_m: float | None
    std_X_m: float | None
    std_Y_m: float | None
    std_Z_m: float | None
    std_lon_m: float | None
    std_lat_m: float | None
    std_h_m: float | None
    baseline_dX_m: float | None
    baseline_dY_m: float | None
    baseline_dZ_m: float | None
    baseline_E_m: float | None
    baseline_N_m: float | None
    baseline_U_m: float | None
    baseline_length_m: float | None
    q1_fixed_percent: float | None
    ratio_min: float | None
    ratio_mean: float | None
    ratio_max: float | None
    n_fixed_epochs_used: int | None = None
    fixed_time_start: Any | None = None
    fixed_time_end: Any | None = None
    fixed_total_duration_min: float | None = None
    longest_fixed_segment_start: Any | None = None
    longest_fixed_segment_end: Any | None = None
    longest_fixed_segment_duration_min: float | None = None
    longest_fixed_segment_epochs: int | None = None
    qc_flags: list[str] = field(default_factory=list)


def dataclass_to_dict(obj):
    out = asdict(obj)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out
