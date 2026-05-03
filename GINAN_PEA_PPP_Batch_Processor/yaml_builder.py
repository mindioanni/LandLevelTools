from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re


def compute_run_label(dataset_context: dict) -> str:
    dataset_name = dataset_context["identity"]["dataset_name"]
    eff = dataset_context["resampling"]["effective_interval_sec"]

    if eff is None:
        raise ValueError("effective_interval_sec is missing in dataset_context")

    return f"{dataset_name}_{int(eff)}s"


def build_output_paths(config: dict, dataset_context: dict) -> dict:
    ctx = deepcopy(dataset_context)

    raw_dataset_dir = Path(ctx["raw"]["raw_dataset_dir"])
    raw_root = Path(ctx["raw"].get("raw_root") or raw_dataset_dir.parent).expanduser().resolve()

    ginan_process_dir = raw_root / config["processing"]["ginan_process_folder_name"]
    yaml_dir = raw_root / config["processing"]["yaml_subfolder_name"]

    run_label = compute_run_label(ctx)
    run_dir = ginan_process_dir / run_label
    yaml_path = yaml_dir / f"{run_label}.yaml"
    stdout_path = run_dir / f"{config['processing']['stdout_prefix']}{run_label}.txt"
    manifest_path = run_dir / config["processing"]["manifest_filename"]
    commands_path = run_dir / config["processing"]["commands_filename"]

    ctx["outputs"]["ginan_process_dir"] = str(ginan_process_dir)
    ctx["outputs"]["yaml_dir"] = str(yaml_dir)
    ctx["outputs"]["run_label"] = run_label
    ctx["outputs"]["run_dir"] = str(run_dir)
    ctx["outputs"]["yaml_path"] = str(yaml_path)
    ctx["outputs"]["stdout_path"] = str(stdout_path)
    ctx["outputs"]["manifest_path"] = str(manifest_path)
    ctx["outputs"]["commands_path"] = str(commands_path)

    return ctx


def _find_block_bounds(lines: list[str], block_name: str, base_indent: int) -> tuple[int, int, int]:
    prefix = " " * base_indent
    key_pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(block_name)}:\s*(?:#.*)?$")

    key_idx = None
    for i, line in enumerate(lines):
        if key_pattern.match(line):
            key_idx = i
            break

    if key_idx is None:
        raise ValueError(f"Could not find YAML block: {block_name}")

    start = key_idx + 1
    end = start

    while end < len(lines):
        line = lines[end]

        if not line.strip():
            end += 1
            continue

        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent <= base_indent:
            break

        end += 1

    return key_idx, start, end


def _extract_list_block_items(yaml_text: str, block_name: str, base_indent: int = 4) -> list[str]:
    lines = yaml_text.splitlines()
    _, start, end = _find_block_bounds(lines, block_name, base_indent)

    items = []
    item_pattern = re.compile(r"^\s*-\s*(.*?)\s*$")

    for line in lines[start:end]:
        m = item_pattern.match(line)
        if m:
            items.append(m.group(1))

    return items


def _replace_list_block(yaml_text: str, block_name: str, items: list[str], base_indent: int = 4) -> str:
    lines = yaml_text.splitlines()
    _, start, end = _find_block_bounds(lines, block_name, base_indent)

    item_indent = " " * (base_indent + 4)
    new_block_lines = [f"{item_indent}- {item}" for item in items]

    new_lines = lines[:start] + new_block_lines + lines[end:]
    return "\n".join(new_lines) + "\n"


def _product_basename(item: str) -> str:
    clean = str(item).strip().strip("'").strip('"')
    return Path(clean).name


def _looks_like_dynamic_crd_snx(item: str) -> bool:
    name = _product_basename(item).upper()

    if name.startswith("IGS0OPSSNX_") and name.endswith("_CRD.SNX"):
        return True

    if re.match(r"^IGS0OPSSNX_\d{11}_01D_01D_CRD\.SNX$", name):
        return True

    return False


def _first_dynamic_crd_snx_basename(dynamic_snx_files: list[str]) -> str:
    for item in dynamic_snx_files:
        if _looks_like_dynamic_crd_snx(item):
            return _product_basename(item)

    raise ValueError("No dynamic IGS0OPSSNX_*_CRD.SNX product found in dataset_context products")


def _merge_template_static_snx_with_gui_style_dynamic_snx(
    yaml_text: str,
    dynamic_snx_files: list[str],
) -> list[str]:
    template_snx_items = _extract_list_block_items(yaml_text, "snx_files", base_indent=4)

    static_items = [
        item for item in template_snx_items
        if not _looks_like_dynamic_crd_snx(item)
    ]

    dynamic_item = _first_dynamic_crd_snx_basename(dynamic_snx_files)

    merged = []
    seen = set()

    for item in static_items + [dynamic_item]:
        key = item.strip()
        if key not in seen:
            merged.append(item)
            seen.add(key)

    return merged


def _gui_style_nav_files() -> list[str]:
    return ["BRDC*"]


def _gui_style_clk_files() -> list[str]:
    return ["'*.CLK'"]


def _gui_style_bsx_files() -> list[str]:
    return ["'*.BIA'"]


def _gui_style_sp3_files() -> list[str]:
    return ["'*.SP3'"]


def _replace_line_value(yaml_text: str, key: str, new_value: str) -> str:
    lines = yaml_text.splitlines()

    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(key)}:\s*.*$", line)
        if m:
            indent = m.group(1)
            lines[i] = f"{indent}{key}: {new_value}"
            return "\n".join(lines) + "\n"

    raise ValueError(f"Could not replace YAML key: {key}")


def _replace_three_value_block(yaml_text: str, key: str, values: list[float]) -> str:
    if len(values) != 3:
        raise ValueError(f"{key} requires exactly three values")

    lines = yaml_text.splitlines()

    key_idx = None
    key_indent = None

    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(key)}:\s*(?:#.*)?$", line)
        if m:
            key_idx = i
            key_indent = len(m.group(1))
            break

    if key_idx is None:
        raise ValueError(f"Could not replace three-value YAML block: {key}")

    start = key_idx + 1
    end = start

    while end < len(lines):
        line = lines[end]

        if not line.strip():
            end += 1
            continue

        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent <= key_indent:
            break

        end += 1

    item_indent = " " * (key_indent + 4)
    replacement_lines = [
        f"{item_indent}- {values[0]}",
        f"{item_indent}- {values[1]}",
        f"{item_indent}- {values[2]}",
    ]

    new_lines = lines[:start] + replacement_lines + lines[end:]
    return "\n".join(new_lines) + "\n"


def _replace_receiver_station_key(yaml_text: str, marker_name: str) -> str:
    lines = yaml_text.splitlines()

    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Z0-9_]{4}):\s*#change to header name of RNX.*$", line)
        if m:
            indent = m.group(1)
            lines[i] = f"{indent}{marker_name}: #change to header name of RNX"
            return "\n".join(lines) + "\n"

    raise ValueError("Could not replace receiver_options station key")


def render_yaml_from_template(config: dict, dataset_context: dict) -> str:
    template_path = Path(config["system"]["template_yaml_path"]).expanduser().resolve()
    yaml_text = template_path.read_text(encoding="utf-8")

    marker_name = dataset_context["header"]["marker_name"]
    receiver_type = dataset_context["header"]["receiver_type"]
    antenna_type = dataset_context["header"]["antenna_type"]
    approx_xyz = dataset_context["header"]["approx_position_xyz"]
    delta_hen = dataset_context["header"]["antenna_delta_hen"]

    start_epoch = dataset_context["time_window"]["start_epoch"]
    end_epoch = dataset_context["time_window"]["end_epoch"]
    interval = dataset_context["resampling"]["effective_interval_sec"]

    obs_root = dataset_context["resampling"]["resampled_dataset_dir"]
    obs_filename = dataset_context["resampling"]["resampled_rinex_filename"]
    outputs_root = dataset_context["outputs"]["run_dir"]

    products = dataset_context.get("products", {})
    products_root = products.get("igs_precise_dir")
    dynamic_snx_files = products.get("snx_files", [])

    if not products_root:
        raise ValueError("products igs_precise_dir is missing")
    if not marker_name:
        raise ValueError("marker_name is missing")
    if not receiver_type:
        raise ValueError("receiver_type is missing")
    if not antenna_type:
        raise ValueError("antenna_type is missing")
    if not start_epoch or not end_epoch:
        raise ValueError("start/end epoch is missing")
    if interval is None:
        raise ValueError("effective_interval_sec is missing")
    if not obs_root or not obs_filename:
        raise ValueError("observation path information is missing")
    if not outputs_root:
        raise ValueError("outputs_root is missing")

    snx_files = _merge_template_static_snx_with_gui_style_dynamic_snx(
        yaml_text,
        dynamic_snx_files,
    )

    yaml_text = _replace_line_value(yaml_text, "inputs_root", f"{products_root} #USER_SET")

    yaml_text = _replace_list_block(yaml_text, "snx_files", snx_files, base_indent=4)
    yaml_text = _replace_list_block(yaml_text, "nav_files", _gui_style_nav_files(), base_indent=8)
    yaml_text = _replace_list_block(yaml_text, "clk_files", _gui_style_clk_files(), base_indent=8)
    yaml_text = _replace_list_block(yaml_text, "bsx_files", _gui_style_bsx_files(), base_indent=8)
    yaml_text = _replace_list_block(yaml_text, "sp3_files", _gui_style_sp3_files(), base_indent=8)

    yaml_text = _replace_line_value(yaml_text, "gnss_observations_root", f"{obs_root} #USER_SET")
    yaml_text = _replace_list_block(yaml_text, "rnx_inputs", [obs_filename], base_indent=8)

    yaml_text = _replace_line_value(yaml_text, "outputs_root", f"{outputs_root} #USER_SET")
    yaml_text = _replace_line_value(yaml_text, "start_epoch", f"'{start_epoch}' #USER_SET")
    yaml_text = _replace_line_value(yaml_text, "end_epoch", f"'{end_epoch}' #USER_SET")
    yaml_text = _replace_line_value(yaml_text, "epoch_interval", f"{int(interval)}    #USER_SET")

    yaml_text = _replace_line_value(yaml_text, "receiver_type", f"{receiver_type} #USER_SET (string)")
    yaml_text = _replace_line_value(yaml_text, "antenna_type", f"{antenna_type} #USER_SET (string)")

    yaml_text = _replace_receiver_station_key(yaml_text, marker_name)
    yaml_text = _replace_three_value_block(yaml_text, "apriori_position", approx_xyz)

    h, e, n = delta_hen
    yaml_text = _replace_three_value_block(yaml_text, "offset", [e, n, h])

    return yaml_text


def write_yaml_file(yaml_text: str, yaml_path: str) -> dict:
    p = Path(yaml_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml_text, encoding="utf-8")

    return {
        "ok": True,
        "yaml_path": str(p),
        "message": f"YAML written to {p}",
    }
