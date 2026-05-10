from __future__ import annotations

from pathlib import Path

from models import UserInputs, CorsSolution, BaselinePair, ResolvedProducts, RunConfig
from sp3_compat import convert_sp3_to_rtklib_242_subset


def _frequency_value(mode: str) -> str:
    return {
        "L1": "l1",
        "L1+L2": "l1+l2",
        "L1+L2+L5": "l1+l2+l5",
    }.get(str(mode).strip(), "l1+l2")


def _posmode_value(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "static":
        return "static"
    if mode == "dynamic":
        return "kinematic"
    if mode == "kinematic":
        return "kinematic"
    return "static"


def _soltype_value(solution_type: str) -> str:
    return {
        "forward": "forward",
        "backward": "backward",
        "combined": "combined",
    }.get(str(solution_type).strip().lower(), "forward")


def _armode_value(mode: str) -> str:
    mode = str(mode).strip().lower()
    return {
        "continuous": "continuous",
        "instantaneous": "instantaneous",
        "fix-and-hold": "fix-and-hold",
        "fix and hold": "fix-and-hold",
        "off": "off",
    }.get(mode, "continuous")


def _navsys_mask(systems: list[str] | tuple[str, ...]) -> int:
    # RTKLIB 2.4.x bit mask convention:
    # GPS=1, SBAS=2, GLONASS=4, Galileo=8, QZSS=16, BeiDou=32.
    values = {"G": 1, "S": 2, "R": 4, "E": 8, "J": 16, "C": 32}
    mask = 0
    for s in systems:
        mask |= values.get(str(s).upper(), 0)
    return mask or 1


def _solformat_value(output_coordinate_format: str) -> str:
    fmt = str(output_coordinate_format).strip().lower()

    if fmt in {"ecef xyz", "xyz", "ecef"}:
        return "xyz"

    if fmt in {"enu baseline", "enu"}:
        return "enu"

    if fmt in {"llh", "lat lon height", "geodetic"}:
        return "llh"

    return "xyz"


def _command_coordinate_flag(output_coordinate_format: str) -> str | None:
    fmt = str(output_coordinate_format).strip().lower()

    if fmt in {"ecef xyz", "xyz", "ecef"}:
        return "-e"

    if fmt in {"enu baseline", "enu"}:
        return "-a"

    return "-e"


def _sp3_compat_systems(inputs: UserInputs) -> tuple[str, ...]:
    """
    RTKLIB 2.4.2 p13 fails with modern MGEX SP3-d files with >99 satellites.
    For v0.1, precise mode uses a GPS/Galileo-compatible SP3 subset.
    """
    requested = [str(s).upper() for s in getattr(inputs, "nav_systems", ["G", "E"])]

    systems = tuple(s for s in requested if s in {"G", "E"})

    if systems:
        return systems

    return ("G", "E")


def _effective_nav_systems(inputs: UserInputs, products: ResolvedProducts) -> list[str]:
    if str(inputs.product_mode).strip().lower() == "precise" and products.sp3_files:
        return list(_sp3_compat_systems(inputs))

    return list(getattr(inputs, "nav_systems", ["G"]))


def _prepare_sp3_files_for_rtklib_242(
    inputs: UserInputs,
    products: ResolvedProducts,
) -> list[Path]:
    if str(inputs.product_mode).strip().lower() != "precise":
        return []

    if not products.sp3_files:
        return []

    compatible_dir = Path(inputs.output_root) / "compatible_products"
    compatible_dir.mkdir(parents=True, exist_ok=True)

    systems = _sp3_compat_systems(inputs)

    compatible_sp3_files = []
    for sp3 in products.sp3_files:
        compatible_sp3 = convert_sp3_to_rtklib_242_subset(
            input_sp3=sp3,
            output_dir=compatible_dir,
            systems=systems,
        )
        compatible_sp3_files.append(compatible_sp3)

    return compatible_sp3_files


def _rtklib_anttype_from_rinex_ant(antenna: str | None) -> str:
    """
    Convert RINEX ANT # / TYPE text to RTKLIB ant*-anttype text.

    Handles both:
        "HITV500"                 -> "HITV500         NONE"
        "HITV500 NONE"            -> "HITV500         NONE"
        "16578337 HITV500"        -> "HITV500         NONE"
        "15095061 LEIAR10 NONE"   -> "LEIAR10         NONE"

    RINEX ANT # / TYPE often contains:
        antenna_serial antenna_type radome
    RTKLIB ant*-anttype must contain:
        antenna_type radome
    """
    s = (antenna or "").strip()
    if not s:
        return ""

    parts = s.split()
    if not parts:
        return ""

    # If the first token is a serial number, skip it.
    if parts[0].isdigit() and len(parts) >= 2:
        parts = parts[1:]

    model = parts[0]
    radome = parts[1] if len(parts) >= 2 else "NONE"

    return f"{model:<16}{radome}"


def _append_explicit_receiver_antenna_options(lines: list[str], pair: BaselinePair) -> None:
    """
    Append explicit receiver antenna PCV/PCO options.

    The solution remains antenna-to-antenna:
    ant*-antdel* are forced to zero so no marker/BM height transfer is applied
    inside RTKLIB.
    """
    rover_ant = _rtklib_anttype_from_rinex_ant(getattr(pair.rover, "antenna", ""))
    base_ant = _rtklib_anttype_from_rinex_ant(getattr(pair.base, "antenna", ""))

    lines.append("pos1-posopt1      =on")
    lines.append("pos1-posopt2      =on")

    if rover_ant:
        lines.append(f"ant1-anttype      ={rover_ant}")
    if base_ant:
        lines.append(f"ant2-anttype      ={base_ant}")

    lines.extend([
        "ant1-antdele      =0.0000",
        "ant1-antdeln      =0.0000",
        "ant1-antdelu      =0.0000",
        "ant2-antdele      =0.0000",
        "ant2-antdeln      =0.0000",
        "ant2-antdelu      =0.0000",
    ])



def build_conf_text(
    inputs: UserInputs,
    cors: CorsSolution,
    products: ResolvedProducts,
    pair: BaselinePair,
) -> str:
    product_mode = str(inputs.product_mode).strip().lower()

    sateph = "precise" if product_mode == "precise" else "brdc"
    ionoopt = "ionex-tec" if inputs.use_ionex else "brdc"
    nav_systems = _effective_nav_systems(inputs, products)

    lines = [
        "# RTKLIB Baseline Orchestrator generated configuration",
        "# Note: for RTKLIB 2.4.2 p13, modern SP3-d products are converted to SP3-c-like subsets.",
        f"pos1-posmode       ={_posmode_value(inputs.processing_mode)}",
        f"pos1-frequency     ={_frequency_value(inputs.frequency_mode)}",
        f"pos1-soltype       ={_soltype_value(inputs.solution_type)}",
        f"pos1-elmask        ={float(inputs.elevation_mask_deg):.1f}",
        f"pos1-ionoopt       ={ionoopt}",
        "pos1-tropopt       =saas",
        f"pos1-sateph        ={sateph}",
        f"pos1-navsys        ={_navsys_mask(nav_systems)}",
        f"pos2-armode        ={_armode_value(inputs.ambiguity_mode)}",
        f"pos2-arthres       ={float(inputs.ambiguity_threshold):.1f}",
        "out-outhead        =on",
        "out-outopt         =on",
        "out-timesys        =gpst",
        "out-timeform       =hms",
        "out-timendec       =3",
        "out-solstatic      =all",
        f"out-solformat      ={_solformat_value(inputs.output_coordinate_format)}",
    ]

    if products.antex_file:
        lines.append(f"file-satantfile    ={products.antex_file}")
        lines.append(f"file-rcvantfile    ={products.antex_file}")
        _append_explicit_receiver_antenna_options(lines, pair)

    if products.blq_file:
        lines.append(f"file-blqfile       ={products.blq_file}")

    return "\n".join(lines) + "\n"


def build_run_config(
    inputs: UserInputs,
    cors: CorsSolution,
    pair: BaselinePair,
    products: ResolvedProducts,
) -> RunConfig:
    run_dir = Path(inputs.output_root) / "runs" / pair.run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    conf_path = run_dir / "run.conf"
    pos_path = run_dir / f"{pair.run_label}.pos"
    command_path = run_dir / "run_command.txt"

    compatible_sp3_files = _prepare_sp3_files_for_rtklib_242(inputs, products)

    conf_text = build_conf_text(inputs, cors, products, pair)
    conf_path.write_text(conf_text, encoding="utf-8")

    command = [
        str(inputs.rnx2rtkp_path),
        "-k", str(conf_path),
        "-r", f"{cors.X_m:.4f}", f"{cors.Y_m:.4f}", f"{cors.Z_m:.4f}",
    ]

    coord_flag = _command_coordinate_flag(inputs.output_coordinate_format)
    if coord_flag:
        command.append(coord_flag)

    command.extend(["-t", "-o", str(pos_path)])

    command.append(str(pair.rover.path))
    command.append(str(pair.base.path))

    command_files = []
    command_files.extend(products.nav_files)

    if compatible_sp3_files:
        command_files.extend(compatible_sp3_files)
    else:
        command_files.extend(products.sp3_files)

    # RTKLIB 2.4.2 p13 compatibility note:
    # Modern CLK 3.04 files are not passed in v0.1 until separately validated.
    # SP3 position+clock columns are used for the current precise compatibility path.
    command_files.extend(products.ionex_files)

    for p in command_files:
        command.append(str(p))

    if inputs.save_run_command:
        command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

    return RunConfig(
        run_label=pair.run_label,
        run_dir=run_dir,
        conf_path=conf_path,
        output_pos_path=pos_path,
        command_path=command_path,
        command=command,
    )
