from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import shutil
import sys


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


PRESERVED_FILENAMES = {
    "timeseries.out",
    "timeseries.report",
    "timeseries.report.html",
    "timeseries.html",
}

DEFAULT_REMOVE_RESAMPLED_RINEX = True
DEFAULT_REMOVE_GENERATED_YAML = True
DEFAULT_REMOVE_PEA_RUN_OUTPUTS = True
DEFAULT_REMOVE_DOWNLOADED_PRODUCTS = False


@dataclass(frozen=True)
class CleanupLayout:
    raw_root: Path
    metrica_dir: Path
    gnss_dir: Path
    resampled_dirs: list[Path]
    ginan_process_dir: Path
    yaml_dir: Path
    products_dir: Path


@dataclass(frozen=True)
class CleanupItem:
    category: str
    path: Path
    kind: str
    size_bytes: int


@dataclass
class CleanupPlan:
    layout: CleanupLayout
    items: list[CleanupItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preserved: list[Path] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    @property
    def n_items(self) -> int:
        return len(self.items)


def _load_default_config() -> dict:
    try:
        import paths_config
        return paths_config.get_default_config()
    except Exception:
        return {}


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{size} B"


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        _safe_resolve(path).relative_to(_safe_resolve(root))
        return True
    except ValueError:
        return False


def _is_preserved_path(path: Path) -> bool:
    return path.name in PRESERVED_FILENAMES


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    if path.is_symlink():
        return 0

    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                continue
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass

    return total


def _kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "missing"


def _iter_children(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []

    return sorted(path.iterdir(), key=lambda p: str(p))


def resolve_cleanup_layout(
    raw_root: str | Path | None = None,
    ginan_process_dir: str | Path | None = None,
    products_dir: str | Path | None = None,
) -> CleanupLayout:
    config = _load_default_config()
    user_inputs = config.get("user_inputs", {})

    if raw_root is None:
        raw_root = user_inputs.get("raw_root", "")

    if not raw_root:
        raise ValueError("raw_root is required and could not be inferred from paths_config.py")

    raw_root_path = _safe_resolve(Path(raw_root))

    # New layout: RAW_ROOT is the only anchor. Generated outputs are direct
    # children of RAW_ROOT, independent of any external station/GNSS hierarchy.
    metrica_dir = raw_root_path
    gnss_dir = raw_root_path

    resampled_dirs = [
        raw_root_path / "RESAMPLED",
    ]

    if ginan_process_dir is None:
        ginan_process_dir_path = raw_root_path / "GINAN_process"
    else:
        ginan_process_dir_path = _safe_resolve(Path(ginan_process_dir))

    if products_dir is None:
        products_dir_path = raw_root_path / "IGS_PRECISE"
    else:
        products_dir_path = _safe_resolve(Path(products_dir))

    yaml_dir = raw_root_path / "yaml"

    return CleanupLayout(
        raw_root=raw_root_path,
        metrica_dir=_safe_resolve(metrica_dir),
        gnss_dir=_safe_resolve(gnss_dir),
        resampled_dirs=[_safe_resolve(p) for p in resampled_dirs],
        ginan_process_dir=_safe_resolve(ginan_process_dir_path),
        yaml_dir=_safe_resolve(yaml_dir),
        products_dir=_safe_resolve(products_dir_path),
    )


def _add_candidate(
    plan: CleanupPlan,
    category: str,
    path: Path,
    allowed_root: Path,
    seen: set[Path],
) -> None:
    path = _safe_resolve(path)
    allowed_root = _safe_resolve(allowed_root)

    if path in seen:
        return

    if not path.exists():
        return

    if path == allowed_root:
        plan.warnings.append(f"Skipped root directory itself: {path}")
        return

    if not _is_within(path, allowed_root):
        plan.warnings.append(f"Skipped path outside allowed root: {path}")
        return

    if _is_preserved_path(path):
        plan.preserved.append(path)
        return

    if path.is_symlink():
        plan.warnings.append(f"Skipped symbolic link for safety: {path}")
        return

    item = CleanupItem(
        category=category,
        path=path,
        kind=_kind(path),
        size_bytes=_path_size_bytes(path),
    )

    plan.items.append(item)
    seen.add(path)


def build_cleanup_plan(
    raw_root: str | Path | None = None,
    ginan_process_dir: str | Path | None = None,
    products_dir: str | Path | None = None,
    remove_resampled_rinex: bool = DEFAULT_REMOVE_RESAMPLED_RINEX,
    remove_generated_yaml: bool = DEFAULT_REMOVE_GENERATED_YAML,
    remove_pea_run_outputs: bool = DEFAULT_REMOVE_PEA_RUN_OUTPUTS,
    remove_downloaded_products: bool = DEFAULT_REMOVE_DOWNLOADED_PRODUCTS,
) -> CleanupPlan:
    layout = resolve_cleanup_layout(
        raw_root=raw_root,
        ginan_process_dir=ginan_process_dir,
        products_dir=products_dir,
    )

    plan = CleanupPlan(layout=layout)
    seen: set[Path] = set()

    if not layout.raw_root.exists():
        plan.warnings.append(f"RAW root does not exist: {layout.raw_root}")

    if remove_resampled_rinex:
        for resampled_dir in layout.resampled_dirs:
            if not resampled_dir.exists():
                continue

            for child in _iter_children(resampled_dir):
                _add_candidate(
                    plan=plan,
                    category="resampled_rinex",
                    path=child,
                    allowed_root=resampled_dir,
                    seen=seen,
                )

    if remove_generated_yaml:
        if layout.yaml_dir.exists():
            for child in _iter_children(layout.yaml_dir):
                _add_candidate(
                    plan=plan,
                    category="generated_yaml",
                    path=child,
                    allowed_root=layout.yaml_dir,
                    seen=seen,
                )

    if remove_pea_run_outputs:
        if layout.ginan_process_dir.exists():
            for child in _iter_children(layout.ginan_process_dir):
                if child == layout.yaml_dir:
                    continue

                if _is_preserved_path(child):
                    plan.preserved.append(child)
                    continue

                _add_candidate(
                    plan=plan,
                    category="pea_run_outputs",
                    path=child,
                    allowed_root=layout.ginan_process_dir,
                    seen=seen,
                )

    if remove_downloaded_products:
        if layout.products_dir.exists():
            for child in _iter_children(layout.products_dir):
                _add_candidate(
                    plan=plan,
                    category="downloaded_products",
                    path=child,
                    allowed_root=layout.products_dir,
                    seen=seen,
                )

    return plan


def summarize_cleanup_plan(plan: CleanupPlan) -> dict:
    by_category: dict[str, dict[str, int]] = {}

    for item in plan.items:
        if item.category not in by_category:
            by_category[item.category] = {
                "n_items": 0,
                "size_bytes": 0,
            }

        by_category[item.category]["n_items"] += 1
        by_category[item.category]["size_bytes"] += item.size_bytes

    return {
        "n_items": plan.n_items,
        "total_size_bytes": plan.total_size_bytes,
        "total_size_human": _format_bytes(plan.total_size_bytes),
        "by_category": by_category,
        "n_warnings": len(plan.warnings),
        "n_preserved": len(plan.preserved),
    }


def print_cleanup_plan(plan: CleanupPlan, max_items_per_category: int = 20) -> None:
    summary = summarize_cleanup_plan(plan)

    print("Cleanup plan")
    print("------------")
    print(f"RAW root              : {plan.layout.raw_root}")
    print(f"GINAN process dir     : {plan.layout.ginan_process_dir}")
    print(f"Products dir          : {plan.layout.products_dir}")
    print(f"Total candidate items : {summary['n_items']}")
    print(f"Estimated size        : {summary['total_size_human']}")
    print()

    if summary["by_category"]:
        print("By category")
        print("-----------")
        for category, data in summary["by_category"].items():
            print(
                f"{category}: "
                f"{data['n_items']} items, "
                f"{_format_bytes(data['size_bytes'])}"
            )
        print()

    if plan.items:
        print("Candidate paths")
        print("---------------")

        categories = []
        for item in plan.items:
            if item.category not in categories:
                categories.append(item.category)

        for category in categories:
            category_items = [item for item in plan.items if item.category == category]

            print(f"[{category}]")
            for item in category_items[:max_items_per_category]:
                print(f"  - {item.kind:9s} {_format_bytes(item.size_bytes):>10s}  {item.path}")

            remaining = len(category_items) - max_items_per_category
            if remaining > 0:
                print(f"  ... {remaining} more items")
            print()

    if plan.preserved:
        print("Preserved report/timeseries files")
        print("---------------------------------")
        for path in sorted(set(plan.preserved)):
            print(f"  - {path}")
        print()

    if plan.warnings:
        print("Warnings")
        print("--------")
        for warning in plan.warnings:
            print(f"  - {warning}")
        print()


def execute_cleanup_plan(plan: CleanupPlan, execute: bool = False) -> dict:
    if not execute:
        return {
            "ok": True,
            "executed": False,
            "deleted": [],
            "failed": [],
            "message": "Dry-run only. No files were deleted.",
        }

    deleted = []
    failed = []

    for item in plan.items:
        path = item.path

        try:
            if not path.exists():
                continue

            if path.is_symlink():
                failed.append((str(path), "symbolic link skipped"))
                continue

            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
            else:
                failed.append((str(path), "unsupported path type"))
                continue

            deleted.append(str(path))

        except Exception as exc:
            failed.append((str(path), str(exc)))

    return {
        "ok": len(failed) == 0,
        "executed": True,
        "deleted": deleted,
        "failed": failed,
        "message": f"Deleted {len(deleted)} items; failed {len(failed)} items.",
    }


def preview_cleanup(**kwargs) -> CleanupPlan:
    plan = build_cleanup_plan(**kwargs)
    print_cleanup_plan(plan)
    return plan


def clean_generated_files(
    execute: bool = False,
    raw_root: str | Path | None = None,
    ginan_process_dir: str | Path | None = None,
    products_dir: str | Path | None = None,
    remove_resampled_rinex: bool = DEFAULT_REMOVE_RESAMPLED_RINEX,
    remove_generated_yaml: bool = DEFAULT_REMOVE_GENERATED_YAML,
    remove_pea_run_outputs: bool = DEFAULT_REMOVE_PEA_RUN_OUTPUTS,
    remove_downloaded_products: bool = DEFAULT_REMOVE_DOWNLOADED_PRODUCTS,
) -> dict:
    plan = build_cleanup_plan(
        raw_root=raw_root,
        ginan_process_dir=ginan_process_dir,
        products_dir=products_dir,
        remove_resampled_rinex=remove_resampled_rinex,
        remove_generated_yaml=remove_generated_yaml,
        remove_pea_run_outputs=remove_pea_run_outputs,
        remove_downloaded_products=remove_downloaded_products,
    )

    print_cleanup_plan(plan)

    result = execute_cleanup_plan(plan, execute=execute)

    print("Cleanup execution")
    print("-----------------")
    print(result["message"])

    if not execute:
        print("No files were deleted because execute=False.")

    if result["failed"]:
        print("Failures:")
        for path, reason in result["failed"]:
            print(f"  - {path}: {reason}")

    return result


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Preview or execute cleanup of reproducible PPP Batch Orchestrator outputs."
    )

    parser.add_argument(
        "mode",
        choices=["preview", "clean"],
        help="preview: dry-run only; clean: delete selected files only when --execute is also provided",
    )

    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--ginan-process-dir", default=None)
    parser.add_argument("--products-dir", default=None)

    parser.add_argument("--no-resampled-rinex", action="store_true")
    parser.add_argument("--no-generated-yaml", action="store_true")
    parser.add_argument("--no-pea-run-outputs", action="store_true")
    parser.add_argument("--downloaded-products", action="store_true")

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, even clean mode is dry-run only.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    execute = bool(args.execute and args.mode == "clean")

    result = clean_generated_files(
        execute=execute,
        raw_root=args.raw_root,
        ginan_process_dir=args.ginan_process_dir,
        products_dir=args.products_dir,
        remove_resampled_rinex=not args.no_resampled_rinex,
        remove_generated_yaml=not args.no_generated_yaml,
        remove_pea_run_outputs=not args.no_pea_run_outputs,
        remove_downloaded_products=args.downloaded_products,
    )

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
