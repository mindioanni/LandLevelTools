
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


@dataclass
class VelocityDetectionConfig:
    columns: tuple[str, ...] = ("E_m", "N_m", "U_m")
    time_column: str = "time_mean_all_epochs_utc"

    min_series_days: float = 365.0
    min_segment_days: float = 180.0
    min_points_per_segment: int = 30

    max_breaks_per_component: int = 5

    min_bic_improvement: float = 10.0
    min_velocity_change_mm_per_year: float = 5.0

    robust_max_iter: int = 4
    robust_outlier_sigma: float = 4.0
    sigma_floor_m: float = 0.001


def _decimal_year_from_times(times: pd.Series) -> pd.Series:
    t = pd.to_datetime(times, errors="coerce", utc=True)
    out = []

    for item in t:
        if pd.isna(item):
            out.append(math.nan)
            continue

        start = pd.Timestamp(year=int(item.year), month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=int(item.year) + 1, month=1, day=1, tz="UTC")
        dec = int(item.year) + (item - start).total_seconds() / (end - start).total_seconds()
        out.append(float(dec))

    return pd.Series(out, index=times.index)


def _nearest_time_for_decimal_year(df: pd.DataFrame, x: pd.Series, t0: float, time_column: str) -> str:
    if len(df) == 0 or not math.isfinite(t0):
        return ""

    dx = (x - t0).abs()
    idx = dx.idxmin()

    if time_column not in df.columns:
        return ""

    return str(df.loc[idx, time_column])


def _mad_sigma(values: np.ndarray, floor: float = 0.001) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(floor)

    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma):
        sigma = floor

    return max(float(sigma), float(floor))


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def _robust_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 4,
    outlier_sigma: float = 4.0,
    sigma_floor_m: float = 0.001,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 3:
        return {
            "ok": False,
            "n": int(x.size),
            "slope_m_per_year": math.nan,
            "intercept_m": math.nan,
            "rss_m2": math.nan,
            "sigma_m": math.nan,
        }

    # Center x for numerical stability.
    x0 = float(np.median(x))
    xc = x - x0

    keep = np.ones_like(y, dtype=bool)

    for _ in range(max_iter):
        if keep.sum() < 3:
            break

        slope, intercept_centered = _ols_fit(xc[keep], y[keep])
        residuals = y - (slope * xc + intercept_centered)
        sigma = _mad_sigma(residuals[keep], floor=sigma_floor_m)

        new_keep = np.abs(residuals) <= outlier_sigma * sigma

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep

    if keep.sum() < 3:
        keep = np.ones_like(y, dtype=bool)

    slope, intercept_centered = _ols_fit(xc[keep], y[keep])
    residuals = y[keep] - (slope * xc[keep] + intercept_centered)
    rss = float(np.sum(residuals ** 2))
    sigma = _mad_sigma(residuals, floor=sigma_floor_m)

    # Convert centered intercept to ordinary intercept y = slope*x + intercept.
    intercept = intercept_centered - slope * x0

    return {
        "ok": True,
        "n": int(keep.sum()),
        "n_raw": int(y.size),
        "slope_m_per_year": float(slope),
        "intercept_m": float(intercept),
        "rss_m2": max(rss, 1e-18),
        "sigma_m": float(sigma),
        "x_start": float(np.min(x)),
        "x_end": float(np.max(x)),
    }


def _bic(rss: float, n: int, k: int) -> float:
    if n <= 0:
        return math.inf

    rss = max(float(rss), 1e-18)
    return float(n * math.log(rss / n) + k * math.log(n))


def _fit_segment(x: np.ndarray, y: np.ndarray, cfg: VelocityDetectionConfig) -> dict:
    fit = _robust_line_fit(
        x,
        y,
        max_iter=cfg.robust_max_iter,
        outlier_sigma=cfg.robust_outlier_sigma,
        sigma_floor_m=cfg.sigma_floor_m,
    )

    if not fit["ok"]:
        fit["bic"] = math.inf
        return fit

    fit["bic"] = _bic(fit["rss_m2"], fit["n"], k=2)
    return fit


def _best_break(
    x: np.ndarray,
    y: np.ndarray,
    cfg: VelocityDetectionConfig,
) -> dict:
    n = len(x)

    base = _fit_segment(x, y, cfg)

    if not base["ok"]:
        return {"ok": False, "reason": "base fit failed"}

    best = {
        "ok": False,
        "reason": "no valid break",
        "base_fit": base,
    }

    min_years = float(cfg.min_segment_days) / 365.25

    for split in range(cfg.min_points_per_segment, n - cfg.min_points_per_segment):
        x_left = x[:split]
        y_left = y[:split]
        x_right = x[split:]
        y_right = y[split:]

        if (x_left[-1] - x_left[0]) < min_years:
            continue

        if (x_right[-1] - x_right[0]) < min_years:
            continue

        fit_left = _fit_segment(x_left, y_left, cfg)
        fit_right = _fit_segment(x_right, y_right, cfg)

        if not fit_left["ok"] or not fit_right["ok"]:
            continue

        bic_two = fit_left["bic"] + fit_right["bic"]
        bic_improvement = base["bic"] - bic_two

        v1 = fit_left["slope_m_per_year"] * 1000.0
        v2 = fit_right["slope_m_per_year"] * 1000.0
        delta_v = abs(v2 - v1)

        if bic_improvement < cfg.min_bic_improvement:
            continue

        if delta_v < cfg.min_velocity_change_mm_per_year:
            continue

        candidate = {
            "ok": True,
            "split_index": int(split),
            "break_decimal_year": float(x[split]),
            "bic_improvement": float(bic_improvement),
            "velocity_before_mm_per_year": float(v1),
            "velocity_after_mm_per_year": float(v2),
            "delta_velocity_mm_per_year": float(delta_v),
            "fit_left": fit_left,
            "fit_right": fit_right,
            "base_fit": base,
        }

        if (not best["ok"]) or candidate["bic_improvement"] > best["bic_improvement"]:
            best = candidate

    return best


def _recursive_segment(
    x: np.ndarray,
    y: np.ndarray,
    cfg: VelocityDetectionConfig,
    *,
    component: str,
    depth: int,
    max_depth: int,
    out_breaks: list[dict],
) -> list[dict]:
    fit = _fit_segment(x, y, cfg)

    if len(x) < 2 or not fit["ok"]:
        return []

    if depth >= max_depth:
        return [{
            "component": component,
            "start_decimal_year": float(x[0]),
            "end_decimal_year": float(x[-1]),
            "n_points": int(len(x)),
            "velocity_mm_per_year": fit["slope_m_per_year"] * 1000.0,
            "intercept_m": fit["intercept_m"],
            "sigma_m": fit["sigma_m"],
            "bic": fit["bic"],
        }]

    br = _best_break(x, y, cfg)

    if not br["ok"]:
        return [{
            "component": component,
            "start_decimal_year": float(x[0]),
            "end_decimal_year": float(x[-1]),
            "n_points": int(len(x)),
            "velocity_mm_per_year": fit["slope_m_per_year"] * 1000.0,
            "intercept_m": fit["intercept_m"],
            "sigma_m": fit["sigma_m"],
            "bic": fit["bic"],
        }]

    split = br["split_index"]

    out_breaks.append({
        "component": component,
        "break_decimal_year": br["break_decimal_year"],
        "bic_improvement": br["bic_improvement"],
        "velocity_before_mm_per_year": br["velocity_before_mm_per_year"],
        "velocity_after_mm_per_year": br["velocity_after_mm_per_year"],
        "delta_velocity_mm_per_year": br["delta_velocity_mm_per_year"],
    })

    left_segments = _recursive_segment(
        x[:split],
        y[:split],
        cfg,
        component=component,
        depth=depth + 1,
        max_depth=max_depth,
        out_breaks=out_breaks,
    )

    right_segments = _recursive_segment(
        x[split:],
        y[split:],
        cfg,
        component=component,
        depth=depth + 1,
        max_depth=max_depth,
        out_breaks=out_breaks,
    )

    return left_segments + right_segments


def detect_velocity_changes(df: pd.DataFrame, cfg: VelocityDetectionConfig | None = None) -> dict:
    if cfg is None:
        cfg = VelocityDetectionConfig()

    if cfg.time_column not in df.columns:
        raise ValueError(f"Missing time column: {cfg.time_column}")

    x_all = _decimal_year_from_times(df[cfg.time_column])

    segments_all = []
    breaks_all = []

    series_start = float(np.nanmin(x_all.values))
    series_end = float(np.nanmax(x_all.values))

    if (series_end - series_start) * 365.25 < cfg.min_series_days:
        return {
            "ok": False,
            "reason": "series shorter than min_series_days",
            "segments": pd.DataFrame(),
            "breaks": pd.DataFrame(),
        }

    for column in cfg.columns:
        if column not in df.columns:
            continue

        y_all = pd.to_numeric(df[column], errors="coerce")
        valid = np.isfinite(x_all.values) & np.isfinite(y_all.values)

        x = x_all.values[valid]
        y = y_all.values[valid]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if len(x) < 2 * cfg.min_points_per_segment:
            continue

        local_breaks = []

        segments = _recursive_segment(
            x,
            y,
            cfg,
            component=column,
            depth=0,
            max_depth=cfg.max_breaks_per_component,
            out_breaks=local_breaks,
        )

        for item in segments:
            item["duration_days"] = (item["end_decimal_year"] - item["start_decimal_year"]) * 365.25

        for item in local_breaks:
            item["time_utc"] = _nearest_time_for_decimal_year(df, x_all, item["break_decimal_year"], cfg.time_column)

        segments_all.extend(segments)
        breaks_all.extend(local_breaks)

    segments_df = pd.DataFrame(segments_all)
    breaks_df = pd.DataFrame(breaks_all)

    if len(breaks_df) > 0:
        breaks_df = breaks_df.sort_values(["break_decimal_year", "component"]).reset_index(drop=True)

    if len(segments_df) > 0:
        segments_df = segments_df.sort_values(["component", "start_decimal_year"]).reset_index(drop=True)

    return {
        "ok": True,
        "reason": "",
        "segments": segments_df,
        "breaks": breaks_df,
    }


def load_timeseries(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser().resolve(), sep="\t")


# === PATCH: meta-cluster velocity windows START ===

@dataclass
class MetaClusterVelocityWindowConfig:
    columns: tuple[str, ...] = ("E_m", "N_m", "U_m")
    time_column: str = "time_mean_all_epochs_utc"

    min_points_per_window: int = 5
    min_duration_days_for_stable_rate: float = 365.25

    apply_gaussian_smoothing: bool = True
    gaussian_width_days: float = 28.0

    outlier_rejection_enabled: bool = True
    robust_max_iter: int = 4
    robust_outlier_sigma: float = 4.0
    sigma_floor_m: float = 0.001


def _quality_flag_for_velocity_window(n_points: int, duration_days: float, cfg: MetaClusterVelocityWindowConfig) -> str:
    if n_points < cfg.min_points_per_window:
        return "insufficient_points"

    if duration_days < cfg.min_duration_days_for_stable_rate:
        return "short_duration"

    return "ok"



def _pre_smoothing_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    cfg: MetaClusterVelocityWindowConfig,
) -> np.ndarray:
    if not cfg.outlier_rejection_enabled:
        return np.ones_like(y, dtype=bool)

    if len(y) < 5:
        return np.ones_like(y, dtype=bool)

    fit = _robust_line_fit(
        x,
        y,
        max_iter=cfg.robust_max_iter,
        outlier_sigma=cfg.robust_outlier_sigma,
        sigma_floor_m=cfg.sigma_floor_m,
    )

    if not fit.get("ok", False):
        return np.ones_like(y, dtype=bool)

    y_model = fit["slope_m_per_year"] * x + fit["intercept_m"]
    residuals = y - y_model
    sigma = _mad_sigma(residuals, floor=cfg.sigma_floor_m)

    if not math.isfinite(sigma) or sigma <= 0:
        return np.ones_like(y, dtype=bool)

    return np.abs(residuals) <= cfg.robust_outlier_sigma * sigma


def _gaussian_smooth_by_decimal_year(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    gaussian_width_days: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if len(x) == 0:
        return y.copy()

    sigma_years = float(gaussian_width_days) / 365.25

    if not math.isfinite(sigma_years) or sigma_years <= 0:
        return y.copy()

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 3:
        return y.copy()

    out = np.full_like(y, np.nan, dtype=float)

    # Use truncated Gaussian support at ±3 sigma for numerical stability.
    support_years = 3.0 * sigma_years

    for i, xi in enumerate(x):
        local = np.abs(x_valid - xi) <= support_years

        if local.sum() < 3:
            out[i] = y[i]
            continue

        dx = x_valid[local] - xi
        weights = np.exp(-0.5 * (dx / sigma_years) ** 2)

        sw = float(np.sum(weights))

        if sw <= 0 or not math.isfinite(sw):
            out[i] = y[i]
            continue

        out[i] = float(np.sum(weights * y_valid[local]) / sw)

    # Fallback: retain original y wherever smoothing could not be computed.
    fallback = ~np.isfinite(out)
    out[fallback] = y[fallback]

    return out

def _fit_single_velocity_window(
    df: pd.DataFrame,
    x_all: pd.Series,
    component: str,
    mask: pd.Series,
    cfg: MetaClusterVelocityWindowConfig,
) -> dict:
    work = pd.DataFrame({
        "x": x_all,
        "y": pd.to_numeric(df[component], errors="coerce"),
    })

    work = work.loc[mask].copy()
    work = work[np.isfinite(work["x"]) & np.isfinite(work["y"])]

    n_points = int(len(work))

    if n_points == 0:
        return {
            "n_points": 0,
            "start_decimal_year": math.nan,
            "end_decimal_year": math.nan,
            "duration_days": math.nan,
            "velocity_mm_per_year": math.nan,
            "intercept_m": math.nan,
            "sigma_m": math.nan,
            "quality_flag": "no_data",
        }

    x = work["x"].to_numpy(dtype=float)
    y_raw = work["y"].to_numpy(dtype=float)

    start_dec = float(np.nanmin(x))
    end_dec = float(np.nanmax(x))
    duration_days = float((end_dec - start_dec) * 365.25)

    quality_flag = _quality_flag_for_velocity_window(n_points, duration_days, cfg)

    if n_points < 3:
        return {
            "n_points": n_points,
            "start_decimal_year": start_dec,
            "end_decimal_year": end_dec,
            "duration_days": duration_days,
            "velocity_mm_per_year": math.nan,
            "intercept_m": math.nan,
            "sigma_m": math.nan,
            "quality_flag": quality_flag,
            "series_used_for_fit": "insufficient_data",
            "gaussian_width_days": cfg.gaussian_width_days if cfg.apply_gaussian_smoothing else math.nan,
            "n_points_used_for_smoothing": 0,
        }

    if cfg.apply_gaussian_smoothing:
        keep_for_smoothing = _pre_smoothing_outlier_mask(x, y_raw, cfg)
        y_fit = _gaussian_smooth_by_decimal_year(
            x=x,
            y=y_raw,
            valid_mask=keep_for_smoothing,
            gaussian_width_days=cfg.gaussian_width_days,
        )
        series_used_for_fit = "gaussian_smoothed"
        n_points_used_for_smoothing = int(np.sum(keep_for_smoothing))
    else:
        y_fit = y_raw
        series_used_for_fit = "raw"
        n_points_used_for_smoothing = int(len(y_raw))

    fit = _robust_line_fit(
        x,
        y_fit,
        max_iter=cfg.robust_max_iter if cfg.outlier_rejection_enabled else 0,
        outlier_sigma=cfg.robust_outlier_sigma,
        sigma_floor_m=cfg.sigma_floor_m,
    )

    if not fit.get("ok", False):
        return {
            "n_points": n_points,
            "start_decimal_year": start_dec,
            "end_decimal_year": end_dec,
            "duration_days": duration_days,
            "velocity_mm_per_year": math.nan,
            "intercept_m": math.nan,
            "sigma_m": math.nan,
            "quality_flag": "fit_failed",
            "series_used_for_fit": series_used_for_fit,
            "gaussian_width_days": cfg.gaussian_width_days if cfg.apply_gaussian_smoothing else math.nan,
            "n_points_used_for_smoothing": n_points_used_for_smoothing,
        }

    return {
        "n_points": n_points,
        "start_decimal_year": start_dec,
        "end_decimal_year": end_dec,
        "duration_days": duration_days,
        "velocity_mm_per_year": fit["slope_m_per_year"] * 1000.0,
        "intercept_m": fit["intercept_m"],
        "sigma_m": fit["sigma_m"],
        "quality_flag": quality_flag,
        "series_used_for_fit": series_used_for_fit,
        "gaussian_width_days": cfg.gaussian_width_days if cfg.apply_gaussian_smoothing else math.nan,
        "n_points_used_for_smoothing": n_points_used_for_smoothing,
    }


def fit_velocity_windows_around_meta_clusters(
    df: pd.DataFrame,
    meta_clusters: pd.DataFrame,
    cfg: MetaClusterVelocityWindowConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = MetaClusterVelocityWindowConfig()

    if meta_clusters is None or len(meta_clusters) == 0:
        return pd.DataFrame()

    if cfg.time_column not in df.columns:
        raise ValueError(f"Missing time column: {cfg.time_column}")

    missing = [col for col in cfg.columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing component columns: {missing}")

    required_meta = [
        "meta_cluster_id",
        "meta_start_decimal_year",
        "meta_end_decimal_year",
    ]

    missing_meta = [col for col in required_meta if col not in meta_clusters.columns]
    if missing_meta:
        raise ValueError(f"Missing meta-cluster columns: {missing_meta}")

    x_all = _decimal_year_from_times(df[cfg.time_column])

    rows = []

    for _, meta in meta_clusters.iterrows():
        try:
            meta_id = int(meta.get("meta_cluster_id"))
        except Exception:
            meta_id = meta.get("meta_cluster_id", "")

        try:
            meta_start = float(meta.get("meta_start_decimal_year"))
            meta_end = float(meta.get("meta_end_decimal_year"))
        except Exception:
            continue

        if not math.isfinite(meta_start) or not math.isfinite(meta_end):
            continue

        lo = min(meta_start, meta_end)
        hi = max(meta_start, meta_end)

        window_masks = {
            "before_meta_cluster": x_all < lo,
            "during_meta_cluster": (x_all >= lo) & (x_all <= hi),
            "after_meta_cluster": x_all > hi,
        }

        for component in cfg.columns:
            for window_label, mask in window_masks.items():
                fit = _fit_single_velocity_window(
                    df=df,
                    x_all=x_all,
                    component=component,
                    mask=mask,
                    cfg=cfg,
                )

                # === PATCH: velocity-window rate class and displacement START ===
                duration_days = fit.get("duration_days", math.nan)
                velocity_mm_per_year = fit.get("velocity_mm_per_year", math.nan)

                if window_label == "during_meta_cluster":
                    rate_class = "transition_rate"
                else:
                    rate_class = "stable_window_velocity"

                if (
                    isinstance(duration_days, (int, float))
                    and isinstance(velocity_mm_per_year, (int, float))
                    and math.isfinite(duration_days)
                    and math.isfinite(velocity_mm_per_year)
                ):
                    estimated_displacement_mm = velocity_mm_per_year * (duration_days / 365.25)
                else:
                    estimated_displacement_mm = math.nan
                # === PATCH: velocity-window rate class and displacement END ===

                if window_label == "during_meta_cluster":
                    component_jump_map = {
                        "E_m": "E_net_jump_mm",
                        "N_m": "N_net_jump_mm",
                        "U_m": "U_net_jump_mm",
                    }

                    jump_col = component_jump_map.get(component)
                    transition_displacement_mm = math.nan

                    if jump_col is not None:
                        try:
                            transition_displacement_mm = float(meta.get(jump_col, math.nan))
                        except Exception:
                            transition_displacement_mm = math.nan

                    meta_duration_days = (hi - lo) * 365.25

                    if (
                        isinstance(transition_displacement_mm, (int, float))
                        and math.isfinite(transition_displacement_mm)
                        and meta_duration_days > 0
                    ):
                        transition_rate_mm_per_year = transition_displacement_mm / (meta_duration_days / 365.25)
                    else:
                        transition_rate_mm_per_year = math.nan

                    fit["velocity_mm_per_year"] = transition_rate_mm_per_year
                    fit["estimated_displacement_mm"] = transition_displacement_mm
                    fit["series_used_for_fit"] = "meta_cluster_net_jump"
                    fit["quality_flag"] = "transition_interval"
                    estimated_displacement_mm = transition_displacement_mm

                row = {
                    "meta_cluster_id": meta_id,
                    "strict_cluster_ids": meta.get("strict_cluster_ids", ""),
                    "component": component,
                    "window_label": window_label,
                    "rate_class": rate_class,
                    "meta_start_decimal_year": lo,
                    "meta_end_decimal_year": hi,
                    "meta_duration_days": (hi - lo) * 365.25,
                    "estimated_displacement_mm": estimated_displacement_mm,
                }
                row.update(fit)

                rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["meta_cluster_id", "component", "window_label"]
        ).reset_index(drop=True)

    return out

# === PATCH: meta-cluster velocity windows END ===



# === PATCH: rolling velocity diagnostics START ===

@dataclass
class RollingVelocityDiagnosticConfig:
    columns: tuple[str, ...] = ("E_m", "N_m", "U_m")
    time_column: str = "time_mean_all_epochs_utc"

    window_days: float = 182.0
    step_days: float = 7.0
    comparison_lag_days: float | None = None

    min_points_per_window: int = 20

    apply_gaussian_smoothing: bool = True
    gaussian_width_days: float = 28.0

    outlier_rejection_enabled: bool = True
    robust_max_iter: int = 4
    robust_outlier_sigma: float = 4.0
    sigma_floor_m: float = 0.001

    significance_z_threshold: float = 4.0
    min_abs_delta_velocity_mm_per_year: float = 1.0
    minimum_persistence_fraction_of_window: float = 0.5
    horizontal_coherence_required: bool = True
    meta_association_window_days: float | None = None

    include_horizontal_vector: bool = True


def _decimal_year_from_timeseries_dataframe(
    df: pd.DataFrame,
    time_column: str = "time_mean_all_epochs_utc",
) -> pd.Series:
    if "decimal_year" in df.columns:
        return pd.to_numeric(df["decimal_year"], errors="coerce")

    if time_column not in df.columns:
        raise KeyError(f"Time column not found and decimal_year is absent: {time_column}")

    t = pd.to_datetime(df[time_column], errors="coerce", utc=True)

    out = []

    for item in t:
        if pd.isna(item):
            out.append(math.nan)
            continue

        start = pd.Timestamp(year=int(item.year), month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=int(item.year) + 1, month=1, day=1, tz="UTC")

        out.append(
            float(int(item.year) + (item - start).total_seconds() / (end - start).total_seconds())
        )

    return pd.Series(out, index=df.index, dtype="float64")


def _rolling_velocity_prepare_component_series(
    x: np.ndarray,
    y_raw: np.ndarray,
    cfg: RollingVelocityDiagnosticConfig,
) -> tuple[np.ndarray, str, int]:
    if not cfg.apply_gaussian_smoothing:
        return y_raw.copy(), "raw", int(np.isfinite(y_raw).sum())

    try:
        smooth_cfg = MetaClusterVelocityWindowConfig(
            columns=("tmp",),
            min_points_per_window=cfg.min_points_per_window,
            min_duration_days_for_stable_rate=365.25,
            apply_gaussian_smoothing=True,
            gaussian_width_days=cfg.gaussian_width_days,
            outlier_rejection_enabled=cfg.outlier_rejection_enabled,
            robust_max_iter=cfg.robust_max_iter,
            robust_outlier_sigma=cfg.robust_outlier_sigma,
            sigma_floor_m=cfg.sigma_floor_m,
        )

        finite = np.isfinite(x) & np.isfinite(y_raw)

        if finite.sum() < 3:
            return y_raw.copy(), "raw_insufficient_for_smoothing", int(finite.sum())

        keep_local = _pre_smoothing_outlier_mask(
            x[finite],
            y_raw[finite],
            smooth_cfg,
        )

        valid_mask = np.zeros_like(y_raw, dtype=bool)
        valid_indices = np.where(finite)[0]
        valid_mask[valid_indices[keep_local]] = True

        y_smooth = _gaussian_smooth_by_decimal_year(
            x=x,
            y=y_raw,
            valid_mask=valid_mask,
            gaussian_width_days=cfg.gaussian_width_days,
        )

        return y_smooth, "gaussian_smoothed", int(valid_mask.sum())

    except Exception:
        return y_raw.copy(), "raw_smoothing_failed", int(np.isfinite(y_raw).sum())


def _fit_rolling_velocity_window(
    x: np.ndarray,
    y: np.ndarray,
    cfg: RollingVelocityDiagnosticConfig,
) -> dict:
    finite = np.isfinite(x) & np.isfinite(y)
    xw = x[finite]
    yw = y[finite]

    n_points = int(len(xw))

    if n_points < cfg.min_points_per_window:
        return {
            "n_points": n_points,
            "velocity_mm_per_year": math.nan,
            "sigma_velocity_mm_per_year": math.nan,
            "sigma_m": math.nan,
            "quality_flag": "insufficient_points",
        }

    try:
        fit = _robust_line_fit(
            xw,
            yw,
            max_iter=cfg.robust_max_iter if cfg.outlier_rejection_enabled else 0,
            outlier_sigma=cfg.robust_outlier_sigma,
            sigma_floor_m=cfg.sigma_floor_m,
        )
    except Exception:
        return {
            "n_points": n_points,
            "velocity_mm_per_year": math.nan,
            "sigma_velocity_mm_per_year": math.nan,
            "sigma_m": math.nan,
            "quality_flag": "fit_failed",
        }

    if not fit.get("ok", False):
        return {
            "n_points": n_points,
            "velocity_mm_per_year": math.nan,
            "sigma_velocity_mm_per_year": math.nan,
            "sigma_m": math.nan,
            "quality_flag": "fit_failed",
        }

    x_mean = float(np.nanmean(xw))
    sxx = float(np.nansum((xw - x_mean) ** 2))

    sigma_m = float(fit.get("sigma_m", math.nan))

    if math.isfinite(sigma_m) and math.isfinite(sxx) and sxx > 0:
        sigma_velocity_mm_per_year = 1000.0 * sigma_m / math.sqrt(sxx)
    else:
        sigma_velocity_mm_per_year = math.nan

    return {
        "n_points": n_points,
        "velocity_mm_per_year": 1000.0 * float(fit["slope_m_per_year"]),
        "sigma_velocity_mm_per_year": sigma_velocity_mm_per_year,
        "sigma_m": sigma_m,
        "quality_flag": "ok",
    }


def compute_rolling_velocity_diagnostics(
    df: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = RollingVelocityDiagnosticConfig()

    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    x_series = _decimal_year_from_timeseries_dataframe(df, cfg.time_column)
    x_all = x_series.to_numpy(dtype=float)

    finite_x = x_all[np.isfinite(x_all)]

    if len(finite_x) == 0:
        return pd.DataFrame(), pd.DataFrame()

    window_years = float(cfg.window_days) / 365.25
    step_years = float(cfg.step_days) / 365.25

    if not math.isfinite(window_years) or window_years <= 0:
        raise ValueError("Rolling velocity window_days must be positive.")

    if not math.isfinite(step_years) or step_years <= 0:
        raise ValueError("Rolling velocity step_days must be positive.")

    x_min = float(np.nanmin(finite_x))
    x_max = float(np.nanmax(finite_x))

    first_center = x_min + 0.5 * window_years
    last_center = x_max - 0.5 * window_years

    rows = []

    if last_center < first_center:
        return pd.DataFrame(), pd.DataFrame()

    centers = []
    c = first_center

    while c <= last_center + 1.0e-12:
        centers.append(float(c))
        c += step_years

    for component in cfg.columns:
        if component not in df.columns:
            continue

        y_raw = pd.to_numeric(df[component], errors="coerce").to_numpy(dtype=float)
        y_used, series_used, n_points_used_for_smoothing = _rolling_velocity_prepare_component_series(
            x=x_all,
            y_raw=y_raw,
            cfg=cfg,
        )

        for center in centers:
            start_dec = center - 0.5 * window_years
            end_dec = center + 0.5 * window_years

            mask = (x_all >= start_dec) & (x_all <= end_dec)
            fit = _fit_rolling_velocity_window(
                x=x_all[mask],
                y=y_used[mask],
                cfg=cfg,
            )

            rows.append({
                "component": component,
                "center_decimal_year": center,
                "start_decimal_year": start_dec,
                "end_decimal_year": end_dec,
                "window_days": float(cfg.window_days),
                "step_days": float(cfg.step_days),
                "series_used_for_fit": series_used,
                "n_points_used_for_smoothing": n_points_used_for_smoothing,
                **fit,
            })

    rolling = pd.DataFrame(rows)

    if len(rolling) == 0:
        return rolling, pd.DataFrame()

    changes = compute_rolling_velocity_changes(rolling, cfg)

    if cfg.include_horizontal_vector:
        horizontal_rolling, horizontal_changes = compute_horizontal_rolling_velocity(rolling, cfg)

        if len(horizontal_rolling) > 0:
            rolling = pd.concat([rolling, horizontal_rolling], ignore_index=True)

        if len(horizontal_changes) > 0:
            changes = pd.concat([changes, horizontal_changes], ignore_index=True)

    return rolling, changes


def compute_rolling_velocity_changes(
    rolling: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = RollingVelocityDiagnosticConfig()

    if rolling is None or len(rolling) == 0:
        return pd.DataFrame()

    lag_days = cfg.comparison_lag_days

    if lag_days is None:
        lag_days = 0.5 * float(cfg.window_days)

    lag_years = float(lag_days) / 365.25

    rows = []

    for component, group in rolling.groupby("component"):
        work = group.copy()
        work["center_decimal_year"] = pd.to_numeric(work["center_decimal_year"], errors="coerce")
        work = work.sort_values("center_decimal_year").reset_index(drop=True)

        for i, row in work.iterrows():
            center = float(row.get("center_decimal_year", math.nan))

            if not math.isfinite(center):
                continue

            previous = work[work["center_decimal_year"] <= center - lag_years]

            if len(previous) == 0:
                continue

            prev = previous.iloc[-1]

            try:
                v1 = float(prev.get("velocity_mm_per_year", math.nan))
                v2 = float(row.get("velocity_mm_per_year", math.nan))
                s1 = float(prev.get("sigma_velocity_mm_per_year", math.nan))
                s2 = float(row.get("sigma_velocity_mm_per_year", math.nan))
            except Exception:
                continue

            delta_v = v2 - v1

            if math.isfinite(s1) and math.isfinite(s2):
                sigma_delta = math.sqrt(s1 ** 2 + s2 ** 2)
            else:
                sigma_delta = math.nan

            if math.isfinite(sigma_delta) and sigma_delta > 0:
                z_score = abs(delta_v) / sigma_delta
            else:
                z_score = math.nan

            rows.append({
                "component": component,
                "center_decimal_year": center,
                "reference_center_decimal_year": float(prev.get("center_decimal_year", math.nan)),
                "comparison_lag_days": float(lag_days),
                "velocity_reference_mm_per_year": v1,
                "velocity_current_mm_per_year": v2,
                "delta_velocity_mm_per_year": delta_v,
                "sigma_delta_velocity_mm_per_year": sigma_delta,
                "velocity_change_z": z_score,
                "significant_velocity_change": bool(
                    math.isfinite(z_score) and z_score >= float(cfg.significance_z_threshold)
                ),
                "z_threshold": float(cfg.significance_z_threshold),
            })

    return pd.DataFrame(rows)


def compute_horizontal_rolling_velocity(
    rolling: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = RollingVelocityDiagnosticConfig()

    if rolling is None or len(rolling) == 0:
        return pd.DataFrame(), pd.DataFrame()

    required_components = {"E_m", "N_m"}

    if not required_components.issubset(set(rolling["component"].astype(str))):
        return pd.DataFrame(), pd.DataFrame()

    e = rolling[rolling["component"] == "E_m"].copy()
    n = rolling[rolling["component"] == "N_m"].copy()

    key_cols = [
        "center_decimal_year",
        "start_decimal_year",
        "end_decimal_year",
        "window_days",
        "step_days",
    ]

    merged = pd.merge(
        e,
        n,
        on=key_cols,
        suffixes=("_E", "_N"),
        how="inner",
    )

    rows = []

    for _, item in merged.iterrows():
        vE = float(item.get("velocity_mm_per_year_E", math.nan))
        vN = float(item.get("velocity_mm_per_year_N", math.nan))
        sE = float(item.get("sigma_velocity_mm_per_year_E", math.nan))
        sN = float(item.get("sigma_velocity_mm_per_year_N", math.nan))

        vh = math.sqrt(vE ** 2 + vN ** 2) if math.isfinite(vE) and math.isfinite(vN) else math.nan
        az = math.degrees(math.atan2(vE, vN)) if math.isfinite(vE) and math.isfinite(vN) else math.nan

        if math.isfinite(vh) and vh > 0 and math.isfinite(sE) and math.isfinite(sN):
            sigma_vh = math.sqrt((vE / vh * sE) ** 2 + (vN / vh * sN) ** 2)
        else:
            sigma_vh = math.nan

        qE = str(item.get("quality_flag_E", ""))
        qN = str(item.get("quality_flag_N", ""))

        quality = "ok" if qE == "ok" and qN == "ok" else "component_quality_warning"

        rows.append({
            "component": "H_magnitude",
            "center_decimal_year": float(item["center_decimal_year"]),
            "start_decimal_year": float(item["start_decimal_year"]),
            "end_decimal_year": float(item["end_decimal_year"]),
            "window_days": float(item["window_days"]),
            "step_days": float(item["step_days"]),
            "series_used_for_fit": "derived_from_E_N",
            "n_points": min(int(item.get("n_points_E", 0)), int(item.get("n_points_N", 0))),
            "n_points_used_for_smoothing": min(
                int(item.get("n_points_used_for_smoothing_E", 0)),
                int(item.get("n_points_used_for_smoothing_N", 0)),
            ),
            "velocity_mm_per_year": vh,
            "sigma_velocity_mm_per_year": sigma_vh,
            "horizontal_azimuth_deg": az,
            "sigma_m": math.nan,
            "quality_flag": quality,
        })

    horizontal_rolling = pd.DataFrame(rows)
    horizontal_changes = compute_rolling_velocity_changes(horizontal_rolling, cfg)

    return horizontal_rolling, horizontal_changes


# === PATCH: rolling velocity diagnostics END ===



# === PATCH: persistent velocity change V1 START ===

def _velocity_change_minimum_persistence_days(cfg: RollingVelocityDiagnosticConfig) -> float:
    return float(cfg.minimum_persistence_fraction_of_window) * float(cfg.window_days)


def _velocity_change_minimum_centers(cfg: RollingVelocityDiagnosticConfig) -> int:
    if float(cfg.step_days) <= 0:
        return 1

    min_days = _velocity_change_minimum_persistence_days(cfg)

    # Duration covered by N centers is (N - 1) * step_days.
    return int(math.ceil(min_days / float(cfg.step_days))) + 1


def cluster_persistent_velocity_changes_v1(
    changes: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = RollingVelocityDiagnosticConfig()

    if changes is None or len(changes) == 0:
        return pd.DataFrame()

    required = {
        "component",
        "center_decimal_year",
        "delta_velocity_mm_per_year",
        "sigma_delta_velocity_mm_per_year",
        "velocity_change_z",
        "significant_velocity_change",
    }

    if not required.issubset(set(changes.columns)):
        return pd.DataFrame()

    work = changes.copy()
    work["component"] = work["component"].astype(str)
    work["center_decimal_year"] = pd.to_numeric(work["center_decimal_year"], errors="coerce")
    work["delta_velocity_mm_per_year"] = pd.to_numeric(work["delta_velocity_mm_per_year"], errors="coerce")
    work["sigma_delta_velocity_mm_per_year"] = pd.to_numeric(work["sigma_delta_velocity_mm_per_year"], errors="coerce")
    work["velocity_change_z"] = pd.to_numeric(work["velocity_change_z"], errors="coerce")

    work = work[
        (work["significant_velocity_change"] == True)
        & work["center_decimal_year"].notna()
        & work["delta_velocity_mm_per_year"].notna()
        & work["velocity_change_z"].notna()
        & (work["velocity_change_z"].abs() >= float(cfg.significance_z_threshold))
        & (work["delta_velocity_mm_per_year"].abs() >= float(cfg.min_abs_delta_velocity_mm_per_year))
    ].copy()

    if len(work) == 0:
        return pd.DataFrame()

    min_persistence_days = _velocity_change_minimum_persistence_days(cfg)
    min_centers = _velocity_change_minimum_centers(cfg)
    consecutive_gap_years = 1.5 * float(cfg.step_days) / 365.25

    clusters = []
    cluster_id = 0

    for component, group in work.groupby("component"):
        group = group.sort_values("center_decimal_year").reset_index(drop=True)

        current = []

        for _, row in group.iterrows():
            if not current:
                current = [row]
                continue

            previous_center = float(current[-1]["center_decimal_year"])
            current_center = float(row["center_decimal_year"])

            if current_center - previous_center <= consecutive_gap_years:
                current.append(row)
            else:
                summary = _summarize_persistent_velocity_change_cluster_v1(
                    cluster_id=cluster_id + 1,
                    component=component,
                    rows=current,
                    cfg=cfg,
                    min_persistence_days=min_persistence_days,
                    min_centers=min_centers,
                )

                if summary is not None:
                    cluster_id += 1
                    clusters.append(summary)

                current = [row]

        if current:
            summary = _summarize_persistent_velocity_change_cluster_v1(
                cluster_id=cluster_id + 1,
                component=component,
                rows=current,
                cfg=cfg,
                min_persistence_days=min_persistence_days,
                min_centers=min_centers,
            )

            if summary is not None:
                cluster_id += 1
                clusters.append(summary)

    if not clusters:
        return pd.DataFrame()

    out = pd.DataFrame(clusters)

    out = _classify_horizontal_coherence_v1(out, cfg)

    return out


def _summarize_persistent_velocity_change_cluster_v1(
    cluster_id: int,
    component: str,
    rows: list,
    cfg: RollingVelocityDiagnosticConfig,
    min_persistence_days: float,
    min_centers: int,
) -> dict | None:
    group = pd.DataFrame(rows).copy()

    if len(group) == 0:
        return None

    start_dec = float(group["center_decimal_year"].min())
    end_dec = float(group["center_decimal_year"].max())
    duration_days = float((end_dec - start_dec) * 365.25)

    n_centers = int(len(group))

    if n_centers < int(min_centers):
        return None

    if duration_days + 1.0e-9 < float(min_persistence_days):
        return None

    group["abs_z"] = group["velocity_change_z"].abs()
    group["abs_delta"] = group["delta_velocity_mm_per_year"].abs()

    rep_idx = group["abs_z"].idxmax()
    rep = group.loc[rep_idx]

    if component == "U_m":
        preliminary_class = "vertical_diagnostic_only"
    elif component == "H_magnitude":
        preliminary_class = "horizontal_candidate"
    elif component in {"E_m", "N_m"}:
        preliminary_class = "horizontal_component_support"
    else:
        preliminary_class = "diagnostic_only"

    return {
        "velocity_change_cluster_id": int(cluster_id),
        "component": str(component),
        "cluster_start_decimal_year": start_dec,
        "cluster_end_decimal_year": end_dec,
        "cluster_duration_days": duration_days,
        "representative_center_decimal_year": float(rep["center_decimal_year"]),
        "representative_reference_center_decimal_year": float(rep.get("reference_center_decimal_year", math.nan)),
        "n_consecutive_centers": n_centers,
        "minimum_required_centers": int(min_centers),
        "minimum_required_persistence_days": float(min_persistence_days),
        "representative_delta_velocity_mm_per_year": float(rep["delta_velocity_mm_per_year"]),
        "representative_sigma_delta_velocity_mm_per_year": float(rep.get("sigma_delta_velocity_mm_per_year", math.nan)),
        "max_abs_delta_velocity_mm_per_year": float(group["abs_delta"].max()),
        "max_velocity_change_z": float(group["abs_z"].max()),
        "mean_delta_velocity_mm_per_year": float(group["delta_velocity_mm_per_year"].mean()),
        "z_threshold": float(cfg.significance_z_threshold),
        "min_abs_delta_velocity_mm_per_year": float(cfg.min_abs_delta_velocity_mm_per_year),
        "persistence_class": "persistent",
        "preliminary_class": preliminary_class,
        "horizontal_coherence": False,
        "report_grade": False,
        "final_class": preliminary_class,
    }


def _intervals_overlap_v1(a0: float, a1: float, b0: float, b1: float) -> bool:
    if not all(math.isfinite(x) for x in [a0, a1, b0, b1]):
        return False

    lo_a, hi_a = min(a0, a1), max(a0, a1)
    lo_b, hi_b = min(b0, b1), max(b0, b1)

    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _classify_horizontal_coherence_v1(
    clusters: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig,
) -> pd.DataFrame:
    if clusters is None or len(clusters) == 0:
        return clusters

    out = clusters.copy()

    support = out[out["component"].isin(["E_m", "N_m"])].copy()

    for idx, row in out.iterrows():
        component = str(row.get("component", ""))

        if component != "H_magnitude":
            continue

        h0 = float(row.get("cluster_start_decimal_year", math.nan))
        h1 = float(row.get("cluster_end_decimal_year", math.nan))

        overlapping_support = []

        for _, item in support.iterrows():
            s0 = float(item.get("cluster_start_decimal_year", math.nan))
            s1 = float(item.get("cluster_end_decimal_year", math.nan))

            if _intervals_overlap_v1(h0, h1, s0, s1):
                overlapping_support.append(str(item.get("component", "")))

        overlapping_support = sorted(set(overlapping_support))

        if bool(cfg.horizontal_coherence_required):
            coherent = len(overlapping_support) >= 1
        else:
            coherent = True

        out.loc[idx, "horizontal_coherence"] = bool(coherent)
        out.loc[idx, "horizontal_support_components"] = ",".join(overlapping_support)

        if coherent:
            out.loc[idx, "report_grade"] = True
            out.loc[idx, "final_class"] = "report_grade_horizontal_velocity_change"
        else:
            out.loc[idx, "report_grade"] = False
            out.loc[idx, "final_class"] = "horizontal_candidate_without_component_support"

    # Explicitly keep U-only changes diagnostic.
    u_mask = out["component"].astype(str) == "U_m"
    out.loc[u_mask, "report_grade"] = False
    out.loc[u_mask, "final_class"] = "vertical_diagnostic_only"

    return out


# === PATCH: persistent velocity change V1 END ===



# === PATCH: velocity change meta classification V1 START ===

def _velocity_change_meta_association_window_days(
    cfg: RollingVelocityDiagnosticConfig,
) -> float:
    value = getattr(cfg, "meta_association_window_days", None)

    if value is None:
        return 0.5 * float(cfg.window_days)

    try:
        value = float(value)
    except Exception:
        return 0.5 * float(cfg.window_days)

    if not math.isfinite(value) or value < 0:
        return 0.5 * float(cfg.window_days)

    return value


def _extract_meta_cluster_interval_v1(row) -> dict:
    def _first_finite(*names):
        for name in names:
            try:
                value = float(row.get(name, math.nan))
            except Exception:
                value = math.nan

            if math.isfinite(value):
                return value

        return math.nan

    try:
        meta_id = row.get("meta_cluster_id", row.get("cluster_id", math.nan))
    except Exception:
        meta_id = math.nan

    start_dec = _first_finite(
        "meta_start_decimal_year",
        "cluster_start_decimal_year",
        "representative_decimal_year",
    )

    end_dec = _first_finite(
        "meta_end_decimal_year",
        "cluster_end_decimal_year",
        "representative_decimal_year",
    )

    representative_dec = _first_finite(
        "representative_decimal_year",
        "representative_center_decimal_year",
        "meta_start_decimal_year",
        "cluster_start_decimal_year",
    )

    if not math.isfinite(start_dec):
        start_dec = representative_dec

    if not math.isfinite(end_dec):
        end_dec = representative_dec

    if math.isfinite(start_dec) and math.isfinite(end_dec) and end_dec < start_dec:
        start_dec, end_dec = end_dec, start_dec

    return {
        "meta_cluster_id": meta_id,
        "meta_start_decimal_year": start_dec,
        "meta_end_decimal_year": end_dec,
        "meta_representative_decimal_year": representative_dec,
    }


def _interval_overlap_and_gap_days_v1(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> tuple[float, float, str]:
    if not all(math.isfinite(x) for x in [a0, a1, b0, b1]):
        return math.nan, math.nan, "undefined"

    lo_a, hi_a = min(a0, a1), max(a0, a1)
    lo_b, hi_b = min(b0, b1), max(b0, b1)

    overlap_years = min(hi_a, hi_b) - max(lo_a, lo_b)

    if overlap_years >= 0:
        return float(overlap_years * 365.25), 0.0, "overlaps_meta_cluster"

    if hi_a < lo_b:
        gap_days = float((lo_b - hi_a) * 365.25)
        return 0.0, gap_days, "before_meta_cluster"

    gap_days = float((lo_a - hi_b) * 365.25)
    return 0.0, gap_days, "after_meta_cluster"


def classify_velocity_change_clusters_against_meta_clusters_v1(
    velocity_clusters: pd.DataFrame,
    meta_clusters: pd.DataFrame,
    cfg: RollingVelocityDiagnosticConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = RollingVelocityDiagnosticConfig()

    if velocity_clusters is None or len(velocity_clusters) == 0:
        return pd.DataFrame()

    out = velocity_clusters.copy()

    association_window_days = _velocity_change_meta_association_window_days(cfg)

    default_columns = {
        "nearest_meta_cluster_id": math.nan,
        "nearest_meta_relation": "no_meta_cluster_available",
        "nearest_meta_gap_days": math.nan,
        "nearest_meta_overlap_days": math.nan,
        "nearest_meta_start_decimal_year": math.nan,
        "nearest_meta_end_decimal_year": math.nan,
        "meta_association_window_days": association_window_days,
        "shift_related_velocity_change": False,
        "shift_context_class": "independent_or_background_velocity_change",
    }

    for key, value in default_columns.items():
        out[key] = value

    if meta_clusters is None or len(meta_clusters) == 0:
        return out

    meta_intervals = [
        _extract_meta_cluster_interval_v1(row)
        for _, row in meta_clusters.iterrows()
    ]

    meta_intervals = [
        item for item in meta_intervals
        if math.isfinite(float(item["meta_start_decimal_year"]))
        and math.isfinite(float(item["meta_end_decimal_year"]))
    ]

    if len(meta_intervals) == 0:
        return out

    for idx, row in out.iterrows():
        try:
            v0 = float(row.get("cluster_start_decimal_year", math.nan))
            v1 = float(row.get("cluster_end_decimal_year", math.nan))
        except Exception:
            continue

        if not math.isfinite(v0) or not math.isfinite(v1):
            continue

        best = None

        for meta in meta_intervals:
            overlap_days, gap_days, relation = _interval_overlap_and_gap_days_v1(
                v0,
                v1,
                float(meta["meta_start_decimal_year"]),
                float(meta["meta_end_decimal_year"]),
            )

            if not math.isfinite(gap_days):
                continue

            score = -overlap_days if overlap_days > 0 else gap_days

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "meta": meta,
                    "overlap_days": overlap_days,
                    "gap_days": gap_days,
                    "relation": relation,
                }

        if best is None:
            continue

        relation = best["relation"]
        overlap_days = float(best["overlap_days"])
        gap_days = float(best["gap_days"])

        shift_related = bool(
            relation == "overlaps_meta_cluster"
            or (
                math.isfinite(gap_days)
                and gap_days <= association_window_days
            )
        )

        component = str(row.get("component", ""))
        report_grade = bool(row.get("report_grade", False))

        if shift_related and report_grade and component == "H_magnitude":
            shift_context_class = "shift_related_report_grade_horizontal_velocity_change"
        elif shift_related and component == "H_magnitude":
            shift_context_class = "shift_related_horizontal_velocity_diagnostic"
        elif shift_related and component in {"E_m", "N_m"}:
            shift_context_class = "shift_related_horizontal_component_support"
        elif shift_related and component == "U_m":
            shift_context_class = "shift_related_vertical_diagnostic_only"
        elif component == "U_m":
            shift_context_class = "independent_vertical_diagnostic_only"
        elif report_grade and component == "H_magnitude":
            shift_context_class = "background_report_grade_horizontal_velocity_change"
        else:
            shift_context_class = "independent_or_background_velocity_change"

        meta = best["meta"]

        out.loc[idx, "nearest_meta_cluster_id"] = meta["meta_cluster_id"]
        out.loc[idx, "nearest_meta_relation"] = relation
        out.loc[idx, "nearest_meta_gap_days"] = gap_days
        out.loc[idx, "nearest_meta_overlap_days"] = overlap_days
        out.loc[idx, "nearest_meta_start_decimal_year"] = meta["meta_start_decimal_year"]
        out.loc[idx, "nearest_meta_end_decimal_year"] = meta["meta_end_decimal_year"]
        out.loc[idx, "meta_association_window_days"] = association_window_days
        out.loc[idx, "shift_related_velocity_change"] = shift_related
        out.loc[idx, "shift_context_class"] = shift_context_class

    return out


# === PATCH: velocity change meta classification V1 END ===
