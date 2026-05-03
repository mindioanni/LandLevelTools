
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from models import UserInputs, BaselinePair, ResolvedProducts


def _date_range(start: datetime, end: datetime):
    d = start.date()
    last = end.date()
    while d <= last:
        yield d
        d += timedelta(days=1)


def _yyyyddd(d) -> str:
    return f"{d.year}{d.timetuple().tm_yday:03d}"


def _find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _find_all(root: Path, patterns: list[str]) -> list[Path]:
    out = []
    seen = set()
    for pattern in patterns:
        for p in sorted(root.rglob(pattern)):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def resolve_products_for_pair(pair: BaselinePair, inputs: UserInputs) -> ResolvedProducts:
    root = Path(inputs.products_root).expanduser().resolve()
    result = ResolvedProducts(run_label=pair.run_label)

    provider = inputs.product_provider.upper()
    series = inputs.product_series.upper()
    project = inputs.product_project.upper()

    for d in _date_range(pair.overlap_start, pair.overlap_end):
        yyyyddd = _yyyyddd(d)

        nav = _find_first(root, [
            f"BRDC00IGS_R_{yyyyddd}0000_01D_MN.rnx",
            f"BRDC*{yyyyddd}*.rnx",
            f"brdc*{yyyyddd}*.rnx",
            f"*{yyyyddd}*MN.rnx",
        ])
        if nav:
            result.nav_files.append(nav)
        else:
            result.missing_files.append(f"BRDC NAV for {yyyyddd}")

        if inputs.product_mode == "precise":
            sp3 = _find_first(root, [
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_05M_ORB.SP3",
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_15M_ORB.SP3",
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_*_ORB.SP3",
                f"*{yyyyddd}*_ORB.SP3",
                f"*{yyyyddd}*.SP3",
            ])
            if sp3:
                result.sp3_files.append(sp3)
            else:
                result.missing_files.append(f"SP3 orbit for {provider}/{series}/{project} {yyyyddd}")

            clk = _find_first(root, [
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_30S_CLK.CLK",
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_*_CLK.CLK",
                f"*{yyyyddd}*_CLK.CLK",
                f"*{yyyyddd}*.CLK",
            ])
            if clk:
                result.clk_files.append(clk)
            else:
                result.missing_files.append(f"CLK clock for {provider}/{series}/{project} {yyyyddd}")

        if inputs.use_ionex:
            ionex = _find_all(root, [f"*{yyyyddd}*.i", f"*{yyyyddd}*.I", f"*{yyyyddd}*.inx", f"*{yyyyddd}*.INX"])
            if ionex:
                result.ionex_files.extend(ionex)
            else:
                result.missing_files.append(f"IONEX for {yyyyddd}")

        if inputs.use_bia_osb:
            bia = _find_all(root, [
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_01D_OSB.BIA",
                f"{provider}0{project}{series}_{yyyyddd}0000_01D_*_OSB.BIA",
                f"*{yyyyddd}*_OSB.BIA",
                f"*{yyyyddd}*.BIA",
            ])
            if bia:
                result.bia_files.extend(bia)
            else:
                result.missing_files.append(f"BIA/OSB for {provider}/{series}/{project} {yyyyddd}")

    if inputs.use_antex:
        result.antex_file = _find_first(root, ["*.atx", "*.ATX"])

    if inputs.use_blq:
        result.blq_file = _find_first(root, ["*.BLQ", "*.blq"])

    if inputs.product_mode == "broadcast":
        required_ok = len(result.nav_files) > 0
    else:
        required_ok = len(result.nav_files) > 0 and len(result.sp3_files) > 0 and len(result.clk_files) > 0

    result.product_status = "OK" if required_ok and not result.missing_files else ("PARTIAL" if required_ok else "MISSING_REQUIRED")

    # The exact downloader CLI is intentionally isolated for later verification.
    # v0.1 reports missing products; download integration can be enabled after CLI contract is confirmed.
    return result
