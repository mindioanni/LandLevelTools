from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShiftDetectionConfig:
    columns: tuple[str, ...] = ("E_m", "N_m", "U_m")
    min_series_days: float = 100.0
    window_days: float = 30.0
    step_days: float = 1.0
    min_points_per_window: int = 10
    mad_sigma_floor_m: float = 0.001
    min_abs_jump_m: float = 0.005
    min_jump_sigma: float = 5.0
    min_model_improvement_percent: float = 20.0
    min_confidence: float = 0.70
    noise_method: str = "diff_mad"


def timestamp_to_decimal_year(ts) -> float:
    t = pd.to_datetime(ts, errors="coerce", utc=True)

    if pd.isna(t):
        return math.nan

    year = int(t.year)
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")

    return year + (t - start).total_seconds() / (end - start).total_seconds()


def _robust_sigma(values: Iterable[float], sigma_floor: float) -> float:
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)

    if a.size < 2:
        return math.nan

    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    sigma = 1.4826 * mad

    if not math.isfinite(sigma) or sigma < sigma_floor:
        sigma = sigma_floor

    return sigma


def _robust_sigma_from_first_differences(values: Iterable[float], sigma_floor: float) -> float:
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)

    if a.size < 3:
        return math.nan

    d = np.diff(a)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma = 1.4826 * mad / math.sqrt(2.0)

    if not math.isfinite(sigma) or sigma < sigma_floor:
        sigma = sigma_floor

    return sigma


def _rss_around_level(values: np.ndarray, level: float) -> float:
    residuals = values - level
    return float(np.sum(residuals * residuals))


def _confidence_from_evidence(jump_sigma: float, improvement_percent: float, cfg: ShiftDetectionConfig) -> float:
    if not math.isfinite(jump_sigma) or not math.isfinite(improvement_percent):
        return 0.0

    jump_score = min(1.0, max(0.0, (jump_sigma - cfg.min_jump_sigma) / cfg.min_jump_sigma))
    improvement_score = min(
        1.0,
        max(0.0, (improvement_percent - cfg.min_model_improvement_percent) / cfg.min_model_improvement_percent),
    )

    confidence = 0.65 * jump_score + 0.35 * improvement_score
    return float(max(0.0, min(1.0, confidence)))


def detect_shifts(df: pd.DataFrame, config: ShiftDetectionConfig | None = None) -> pd.DataFrame:
    cfg = config or ShiftDetectionConfig()

    if "time_mean_all_epochs_utc" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["__time"] = pd.to_datetime(work["time_mean_all_epochs_utc"], errors="coerce", utc=True)
    work = work.dropna(subset=["__time"]).sort_values("__time").reset_index(drop=True)

    if len(work) < 2:
        return pd.DataFrame()

    duration_days = (work["__time"].max() - work["__time"].min()).total_seconds() / 86400.0

    if duration_days < cfg.min_series_days:
        return pd.DataFrame()

    available_columns = [c for c in cfg.columns if c in work.columns]

    if not available_columns:
        return pd.DataFrame()

    start_time = work["__time"].min() + pd.Timedelta(days=cfg.window_days)
    end_time = work["__time"].max() - pd.Timedelta(days=cfg.window_days)

    if start_time >= end_time:
        return pd.DataFrame()

    candidate_times = pd.date_range(
        start=start_time,
        end=end_time,
        freq=pd.Timedelta(days=cfg.step_days),
    )

    rows = []

    for col in available_columns:
        y_all = pd.to_numeric(work[col], errors="coerce")
        component_sigma_mad = _robust_sigma(y_all, cfg.mad_sigma_floor_m)
        component_sigma_diff = _robust_sigma_from_first_differences(y_all, cfg.mad_sigma_floor_m)

        if cfg.noise_method == "diff_mad":
            component_sigma = component_sigma_diff
        elif cfg.noise_method == "series_mad":
            component_sigma = component_sigma_mad
        else:
            raise ValueError(f"Unsupported shift-detection noise_method: {cfg.noise_method}")

        if not math.isfinite(component_sigma):
            continue

        for t0 in candidate_times:
            pre_mask = (
                (work["__time"] >= t0 - pd.Timedelta(days=cfg.window_days))
                & (work["__time"] < t0)
            )
            post_mask = (
                (work["__time"] >= t0)
                & (work["__time"] < t0 + pd.Timedelta(days=cfg.window_days))
            )

            pre = pd.to_numeric(work.loc[pre_mask, col], errors="coerce").dropna().to_numpy(dtype=float)
            post = pd.to_numeric(work.loc[post_mask, col], errors="coerce").dropna().to_numpy(dtype=float)

            if pre.size < cfg.min_points_per_window or post.size < cfg.min_points_per_window:
                continue

            pre_median = float(np.median(pre))
            post_median = float(np.median(post))
            jump_m = post_median - pre_median
            abs_jump_m = abs(jump_m)
            jump_sigma = abs_jump_m / component_sigma

            combined = np.concatenate([pre, post])
            single_level = float(np.median(combined))

            rss_single = _rss_around_level(combined, single_level)
            rss_split = _rss_around_level(pre, pre_median) + _rss_around_level(post, post_median)

            if rss_single <= 0:
                improvement_percent = 0.0
            else:
                improvement_percent = 100.0 * (rss_single - rss_split) / rss_single

            confidence = _confidence_from_evidence(jump_sigma, improvement_percent, cfg)

            if (
                abs_jump_m >= cfg.min_abs_jump_m
                and jump_sigma >= cfg.min_jump_sigma
                and improvement_percent >= cfg.min_model_improvement_percent
                and confidence >= cfg.min_confidence
            ):
                rows.append({
                    "component": col,
                    "event_type": "shift",
                    "time_utc": t0.isoformat(),
                    "decimal_year": timestamp_to_decimal_year(t0),
                    "pre_median_m": pre_median,
                    "post_median_m": post_median,
                    "jump_m": jump_m,
                    "jump_mm": jump_m * 1000.0,
                    "abs_jump_m": abs_jump_m,
                    "component_sigma_m": component_sigma,
                    "component_sigma_mad_m": component_sigma_mad,
                    "component_sigma_diff_m": component_sigma_diff,
                    "noise_method": cfg.noise_method,
                    "jump_sigma": jump_sigma,
                    "jump_sigma_mad": abs_jump_m / component_sigma_mad if math.isfinite(component_sigma_mad) else math.nan,
                    "jump_sigma_diff": abs_jump_m / component_sigma_diff if math.isfinite(component_sigma_diff) else math.nan,
                    "rss_single": rss_single,
                    "rss_split": rss_split,
                    "model_improvement_percent": improvement_percent,
                    "confidence": confidence,
                    "n_pre": int(pre.size),
                    "n_post": int(post.size),
                    "window_days": cfg.window_days,
                })

    if not rows:
        return pd.DataFrame()

    events = pd.DataFrame(rows)

    events = events.sort_values(
        ["component", "confidence", "abs_jump_m"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return events


def select_top_shift_events(events: pd.DataFrame, min_separation_days: float = 30.0) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()

    work = events.copy()
    work["__time"] = pd.to_datetime(work["time_utc"], errors="coerce", utc=True)
    work = work.dropna(subset=["__time"])

    selected = []

    for component, group in work.groupby("component"):
        group = group.sort_values(["confidence", "abs_jump_m"], ascending=[False, False])

        accepted_times = []

        for _, row in group.iterrows():
            t = row["__time"]

            too_close = any(
                abs((t - prev).total_seconds()) / 86400.0 < min_separation_days
                for prev in accepted_times
            )

            if too_close:
                continue

            selected.append(row.drop(labels=["__time"]).to_dict())
            accepted_times.append(t)

    if not selected:
        return pd.DataFrame()

    return pd.DataFrame(selected).sort_values(
        ["decimal_year", "component"]
    ).reset_index(drop=True)

def cluster_shift_events(
    events: pd.DataFrame,
    cluster_window_days: float = 90.0,
    min_components: int = 1,
) -> pd.DataFrame:
    """
    Cluster nearby shift candidates into broader event groups.

    The detector may produce many adjacent candidate break epochs when the
    physical signal is a transition/deformation episode rather than a single
    instantaneous step. This function groups candidates that are close in time
    and returns one representative event per cluster.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of detect_shifts().
    cluster_window_days : float
        Maximum time separation from the current cluster representative for
        assigning a candidate to the same event group.
    min_components : int
        Minimum number of distinct components required for retaining a cluster.

    Returns
    -------
    pandas.DataFrame
        One row per clustered event.
    """
    if events is None or len(events) == 0:
        return pd.DataFrame()

    work = events.copy()
    work["__time"] = pd.to_datetime(work["time_utc"], errors="coerce", utc=True)
    work = work.dropna(subset=["__time"]).sort_values("__time").reset_index(drop=True)

    if len(work) == 0:
        return pd.DataFrame()

    clusters = []

    for _, row in work.iterrows():
        t = row["__time"]

        assigned = False

        for cluster in clusters:
            center = cluster["center_time"]
            dt_days = abs((t - center).total_seconds()) / 86400.0

            if dt_days <= cluster_window_days:
                cluster["rows"].append(row)
                # Keep the cluster center tied to the highest-confidence member.
                best = max(
                    cluster["rows"],
                    key=lambda r: (
                        float(r.get("confidence", 0.0)),
                        float(r.get("abs_jump_m", 0.0)),
                    ),
                )
                cluster["center_time"] = best["__time"]
                assigned = True
                break

        if not assigned:
            clusters.append({
                "center_time": t,
                "rows": [row],
            })

    out_rows = []

    for idx, cluster in enumerate(clusters, start=1):
        rows = cluster["rows"]

        components = sorted({str(r.get("component", "")) for r in rows if str(r.get("component", ""))})
        if len(components) < min_components:
            continue

        best = max(
            rows,
            key=lambda r: (
                float(r.get("confidence", 0.0)),
                float(r.get("abs_jump_m", 0.0)),
            ),
        )

        cluster_times = [r["__time"] for r in rows]
        start_time = min(cluster_times)
        end_time = max(cluster_times)

        jump_by_component = {}
        confidence_by_component = {}

        for component in components:
            comp_rows = [r for r in rows if str(r.get("component", "")) == component]
            if not comp_rows:
                continue

            comp_best = max(
                comp_rows,
                key=lambda r: (
                    float(r.get("confidence", 0.0)),
                    float(r.get("abs_jump_m", 0.0)),
                ),
            )

            jump_by_component[component] = float(comp_best.get("jump_mm", math.nan))
            confidence_by_component[component] = float(comp_best.get("confidence", math.nan))

        out_rows.append({
            "cluster_id": idx,
            "event_type": "shift_cluster",
            "representative_component": best.get("component", ""),
            "representative_time_utc": best.get("time_utc", ""),
            "representative_decimal_year": float(best.get("decimal_year", math.nan)),
            "cluster_start_utc": start_time.isoformat(),
            "cluster_end_utc": end_time.isoformat(),
            "cluster_start_decimal_year": timestamp_to_decimal_year(start_time),
            "cluster_end_decimal_year": timestamp_to_decimal_year(end_time),
            "n_candidates": int(len(rows)),
            "n_components": int(len(components)),
            "components": ",".join(components),
            "max_abs_jump_mm": float(max(abs(float(r.get("jump_mm", 0.0))) for r in rows)),
            "max_jump_sigma": float(max(float(r.get("jump_sigma", 0.0)) for r in rows)),
            "max_model_improvement_percent": float(max(float(r.get("model_improvement_percent", 0.0)) for r in rows)),
            "max_confidence": float(max(float(r.get("confidence", 0.0)) for r in rows)),
            "E_jump_mm": jump_by_component.get("E_m", math.nan),
            "N_jump_mm": jump_by_component.get("N_m", math.nan),
            "U_jump_mm": jump_by_component.get("U_m", math.nan),
            "E_confidence": confidence_by_component.get("E_m", math.nan),
            "N_confidence": confidence_by_component.get("N_m", math.nan),
            "U_confidence": confidence_by_component.get("U_m", math.nan),
        })

    if not out_rows:
        return pd.DataFrame()

    return pd.DataFrame(out_rows).sort_values(
        ["representative_decimal_year", "max_confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)

def select_report_shift_clusters(
    clusters: pd.DataFrame,
    min_confidence: float = 0.90,
    min_abs_jump_mm: float = 20.0,
    min_components: int = 1,
    max_events: int | None = None,
) -> pd.DataFrame:
    """
    Select shift clusters suitable for report annotation.

    This is a conservative post-filter applied after cluster_shift_events().
    It keeps only clusters with sufficient confidence, amplitude and component
    support. Large single-component events are retained when they exceed the
    amplitude and confidence thresholds.
    """
    if clusters is None or len(clusters) == 0:
        return pd.DataFrame()

    work = clusters.copy()

    required = [
        "max_confidence",
        "max_abs_jump_mm",
        "n_components",
        "representative_decimal_year",
    ]

    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError("Missing required cluster columns: " + ", ".join(missing))

    work["max_confidence"] = pd.to_numeric(work["max_confidence"], errors="coerce")
    work["max_abs_jump_mm"] = pd.to_numeric(work["max_abs_jump_mm"], errors="coerce")
    work["n_components"] = pd.to_numeric(work["n_components"], errors="coerce")
    work["representative_decimal_year"] = pd.to_numeric(
        work["representative_decimal_year"],
        errors="coerce",
    )

    selected = work[
        (work["max_confidence"] >= min_confidence)
        & (work["max_abs_jump_mm"] >= min_abs_jump_mm)
        & (work["n_components"] >= min_components)
    ].copy()

    if len(selected) == 0:
        return pd.DataFrame()

    selected["selection_min_confidence"] = min_confidence
    selected["selection_min_abs_jump_mm"] = min_abs_jump_mm
    selected["selection_min_components"] = min_components
    selected["selection_reason"] = (
        "max_confidence >= "
        + str(min_confidence)
        + "; max_abs_jump_mm >= "
        + str(min_abs_jump_mm)
        + "; n_components >= "
        + str(min_components)
    )

    if max_events is not None:
        selected = selected.sort_values(
            ["max_confidence", "max_abs_jump_mm"],
            ascending=[False, False],
        ).head(int(max_events))

    return selected.sort_values(
        ["representative_decimal_year", "max_confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)



# === PATCH: strict-cluster meta-clustering support START ===
from dataclasses import dataclass as _meta_dataclass
import math as _meta_math
import pandas as _meta_pd


@_meta_dataclass
class MetaClusteringConfig:
    enabled: bool = True

    # Temporal criterion
    max_gap_days: float = 14.0

    # Direction-similarity criterion
    enable_direction_similarity: bool = True
    direction_mode: str = "horizontal"  # "horizontal" or "total"
    max_direction_change_deg: float = 45.0

    # Magnitude-compatibility criterion
    enable_magnitude_compatibility: bool = False
    max_magnitude_ratio: float = 3.0

    # Numerical safeguard
    min_vector_norm_mm: float = 1.0


def _meta_to_float(value, default=_meta_math.nan):
    try:
        out = float(value)
    except Exception:
        return default

    if not _meta_math.isfinite(out):
        return default

    return out


def _meta_cluster_vector_mm(row, mode: str = "horizontal"):
    mode = str(mode).strip().lower()

    e = _meta_to_float(row.get("E_jump_mm", 0.0), 0.0)
    n = _meta_to_float(row.get("N_jump_mm", 0.0), 0.0)
    u = _meta_to_float(row.get("U_jump_mm", 0.0), 0.0)

    if mode == "total":
        return [e, n, u]

    return [e, n]


def _meta_vector_norm_mm(vector) -> float:
    return _meta_math.sqrt(sum(float(v) * float(v) for v in vector))


def _meta_angle_deg(vector_a, vector_b) -> float:
    norm_a = _meta_vector_norm_mm(vector_a)
    norm_b = _meta_vector_norm_mm(vector_b)

    if norm_a <= 0 or norm_b <= 0:
        return _meta_math.nan

    dot = sum(float(a) * float(b) for a, b in zip(vector_a, vector_b))
    cosang = max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    return _meta_math.degrees(_meta_math.acos(cosang))


def _meta_horizontal_azimuth_deg(vector) -> float:
    if len(vector) < 2:
        return _meta_math.nan

    e = float(vector[0])
    n = float(vector[1])

    if _meta_vector_norm_mm([e, n]) <= 0:
        return _meta_math.nan

    az = _meta_math.degrees(_meta_math.atan2(e, n))
    if az < 0:
        az += 360.0

    return az


def _meta_signed_angle_diff_deg(angle_a, angle_b) -> float:
    if not _meta_math.isfinite(angle_a) or not _meta_math.isfinite(angle_b):
        return _meta_math.nan

    return ((angle_b - angle_a + 180.0) % 360.0) - 180.0


def _meta_cluster_gap_days(previous_row, next_row) -> float:
    previous_end = _meta_to_float(previous_row.get("cluster_end_decimal_year"))
    next_start = _meta_to_float(next_row.get("cluster_start_decimal_year"))

    if not _meta_math.isfinite(previous_end) or not _meta_math.isfinite(next_start):
        return _meta_math.nan

    return (next_start - previous_end) * 365.25


def _meta_can_merge_adjacent(previous_row, next_row, config: MetaClusteringConfig) -> tuple[bool, dict]:
    diagnostics = {
        "gap_days": _meta_math.nan,
        "direction_change_deg": _meta_math.nan,
        "magnitude_ratio": _meta_math.nan,
        "failed_reason": "",
    }

    gap_days = _meta_cluster_gap_days(previous_row, next_row)
    diagnostics["gap_days"] = gap_days

    if not _meta_math.isfinite(gap_days):
        diagnostics["failed_reason"] = "invalid temporal gap"
        return False, diagnostics

    if gap_days < 0:
        diagnostics["failed_reason"] = "overlapping or unordered clusters"
        return False, diagnostics

    if gap_days > float(config.max_gap_days):
        diagnostics["failed_reason"] = "temporal gap too large"
        return False, diagnostics

    vector_a = _meta_cluster_vector_mm(previous_row, config.direction_mode)
    vector_b = _meta_cluster_vector_mm(next_row, config.direction_mode)

    norm_a = _meta_vector_norm_mm(vector_a)
    norm_b = _meta_vector_norm_mm(vector_b)

    if config.enable_direction_similarity:
        if norm_a < float(config.min_vector_norm_mm) or norm_b < float(config.min_vector_norm_mm):
            diagnostics["failed_reason"] = "vector norm too small for direction test"
            return False, diagnostics

        angle = _meta_angle_deg(vector_a, vector_b)
        diagnostics["direction_change_deg"] = angle

        if not _meta_math.isfinite(angle):
            diagnostics["failed_reason"] = "invalid direction angle"
            return False, diagnostics

        if angle > float(config.max_direction_change_deg):
            diagnostics["failed_reason"] = "direction change too large"
            return False, diagnostics

    if config.enable_magnitude_compatibility:
        if norm_a < float(config.min_vector_norm_mm) or norm_b < float(config.min_vector_norm_mm):
            diagnostics["failed_reason"] = "vector norm too small for magnitude test"
            return False, diagnostics

        ratio = max(norm_a, norm_b) / min(norm_a, norm_b)
        diagnostics["magnitude_ratio"] = ratio

        if ratio > float(config.max_magnitude_ratio):
            diagnostics["failed_reason"] = "magnitude ratio too large"
            return False, diagnostics

    return True, diagnostics


def _meta_direction_behaviour(cluster_rows, direction_mode: str = "horizontal") -> dict:
    azimuths = []

    for row in cluster_rows:
        vector = _meta_cluster_vector_mm(row, direction_mode)
        az = _meta_horizontal_azimuth_deg(vector)
        if _meta_math.isfinite(az):
            azimuths.append(az)

    if len(azimuths) < 2:
        return {
            "start_direction_deg": azimuths[0] if azimuths else _meta_math.nan,
            "end_direction_deg": azimuths[0] if azimuths else _meta_math.nan,
            "cumulative_rotation_deg": 0.0 if azimuths else _meta_math.nan,
            "max_adjacent_rotation_deg": 0.0 if azimuths else _meta_math.nan,
            "direction_behaviour": "single-direction estimate" if azimuths else "undefined",
        }

    signed_diffs = [
        _meta_signed_angle_diff_deg(azimuths[i], azimuths[i + 1])
        for i in range(len(azimuths) - 1)
    ]
    signed_diffs = [v for v in signed_diffs if _meta_math.isfinite(v)]

    cumulative = sum(signed_diffs) if signed_diffs else _meta_math.nan
    max_adjacent = max(abs(v) for v in signed_diffs) if signed_diffs else _meta_math.nan

    if not signed_diffs:
        behaviour = "undefined"
    elif abs(cumulative) < 10.0:
        behaviour = "stable direction"
    elif cumulative > 0:
        behaviour = "gradual clockwise rotation"
    else:
        behaviour = "gradual counter-clockwise rotation"

    return {
        "start_direction_deg": azimuths[0],
        "end_direction_deg": azimuths[-1],
        "cumulative_rotation_deg": cumulative,
        "max_adjacent_rotation_deg": max_adjacent,
        "direction_behaviour": behaviour,
    }


def create_meta_clusters(strict_clusters, config: MetaClusteringConfig | None = None):
    if config is None:
        config = MetaClusteringConfig()

    if strict_clusters is None or len(strict_clusters) == 0:
        return _meta_pd.DataFrame()

    if not config.enabled:
        work = strict_clusters.copy()
        work["meta_cluster_id"] = range(1, len(work) + 1)
        return work

    required = [
        "cluster_id",
        "cluster_start_decimal_year",
        "cluster_end_decimal_year",
        "representative_decimal_year",
    ]

    missing = [col for col in required if col not in strict_clusters.columns]
    if missing:
        raise ValueError(f"Cannot create meta-clusters. Missing strict-cluster columns: {missing}")

    work = strict_clusters.copy()
    work["cluster_start_decimal_year"] = _meta_pd.to_numeric(
        work["cluster_start_decimal_year"], errors="coerce"
    )
    work["cluster_end_decimal_year"] = _meta_pd.to_numeric(
        work["cluster_end_decimal_year"], errors="coerce"
    )
    work["representative_decimal_year"] = _meta_pd.to_numeric(
        work["representative_decimal_year"], errors="coerce"
    )

    work = work.sort_values(
        ["cluster_start_decimal_year", "cluster_end_decimal_year", "representative_decimal_year"]
    ).reset_index(drop=True)

    groups = []
    current = []
    current_pair_diagnostics = []

    for _, row in work.iterrows():
        row_dict = row.to_dict()

        if not current:
            current = [row_dict]
            current_pair_diagnostics = []
            continue

        can_merge, diagnostics = _meta_can_merge_adjacent(current[-1], row_dict, config)

        if can_merge:
            current.append(row_dict)
            current_pair_diagnostics.append(diagnostics)
        else:
            groups.append((current, current_pair_diagnostics))
            current = [row_dict]
            current_pair_diagnostics = []

    if current:
        groups.append((current, current_pair_diagnostics))

    meta_rows = []

    for meta_id, (cluster_rows, pair_diagnostics) in enumerate(groups, start=1):
        cluster_ids = [str(int(_meta_to_float(row.get("cluster_id"), 0))) for row in cluster_rows]

        starts = [_meta_to_float(row.get("cluster_start_decimal_year")) for row in cluster_rows]
        ends = [_meta_to_float(row.get("cluster_end_decimal_year")) for row in cluster_rows]
        reps = [_meta_to_float(row.get("representative_decimal_year")) for row in cluster_rows]

        starts = [v for v in starts if _meta_math.isfinite(v)]
        ends = [v for v in ends if _meta_math.isfinite(v)]
        reps = [v for v in reps if _meta_math.isfinite(v)]

        meta_start = min(starts) if starts else _meta_math.nan
        meta_end = max(ends) if ends else _meta_math.nan
        meta_duration_days = (meta_end - meta_start) * 365.25 if (
            _meta_math.isfinite(meta_start) and _meta_math.isfinite(meta_end)
        ) else _meta_math.nan

        # Representative strict cluster: strongest available evidence.
        def strength_key(row):
            return (
                _meta_to_float(row.get("max_confidence"), -1.0),
                _meta_to_float(row.get("max_abs_jump_mm"), -1.0),
                _meta_to_float(row.get("max_jump_sigma"), -1.0),
            )

        representative_row = max(cluster_rows, key=strength_key)
        representative_decimal_year = _meta_to_float(
            representative_row.get("representative_decimal_year")
        )
        representative_time_utc = representative_row.get("representative_time_utc", "")

        components = []
        for row in cluster_rows:
            for comp in str(row.get("components", "")).split(","):
                comp = comp.strip()
                if comp and comp not in components:
                    components.append(comp)

        e_net = sum(_meta_to_float(row.get("E_jump_mm"), 0.0) for row in cluster_rows)
        n_net = sum(_meta_to_float(row.get("N_jump_mm"), 0.0) for row in cluster_rows)
        u_net = sum(_meta_to_float(row.get("U_jump_mm"), 0.0) for row in cluster_rows)

        h_net = _meta_vector_norm_mm([e_net, n_net])
        total_net = _meta_vector_norm_mm([e_net, n_net, u_net])

        gap_days_values = [
            _meta_to_float(item.get("gap_days"))
            for item in pair_diagnostics
            if _meta_math.isfinite(_meta_to_float(item.get("gap_days")))
        ]
        direction_values = [
            _meta_to_float(item.get("direction_change_deg"))
            for item in pair_diagnostics
            if _meta_math.isfinite(_meta_to_float(item.get("direction_change_deg")))
        ]
        magnitude_values = [
            _meta_to_float(item.get("magnitude_ratio"))
            for item in pair_diagnostics
            if _meta_math.isfinite(_meta_to_float(item.get("magnitude_ratio")))
        ]

        direction_info = _meta_direction_behaviour(cluster_rows, config.direction_mode)

        meta_rows.append({
            "meta_cluster_id": meta_id,
            "strict_cluster_ids": ",".join(cluster_ids),
            "n_strict_clusters": len(cluster_rows),
            "meta_start_decimal_year": meta_start,
            "meta_end_decimal_year": meta_end,
            "meta_duration_days": meta_duration_days,
            "representative_decimal_year": representative_decimal_year,
            "representative_time_utc": representative_time_utc,
            "components": ",".join(components),
            "E_net_jump_mm": e_net,
            "N_net_jump_mm": n_net,
            "U_net_jump_mm": u_net,
            "horizontal_net_jump_mm": h_net,
            "total_net_jump_mm": total_net,
            "max_abs_jump_mm": max(
                _meta_to_float(row.get("max_abs_jump_mm"), 0.0) for row in cluster_rows
            ),
            "max_jump_sigma": max(
                _meta_to_float(row.get("max_jump_sigma"), 0.0) for row in cluster_rows
            ),
            "max_model_improvement_percent": max(
                _meta_to_float(row.get("max_model_improvement_percent"), 0.0)
                for row in cluster_rows
            ),
            "max_confidence": max(
                _meta_to_float(row.get("max_confidence"), 0.0) for row in cluster_rows
            ),
            "max_gap_days": max(gap_days_values) if gap_days_values else 0.0,
            "max_adjacent_direction_change_deg": max(direction_values) if direction_values else 0.0,
            "max_magnitude_ratio": max(magnitude_values) if magnitude_values else _meta_math.nan,
            "direction_mode": config.direction_mode,
            "start_direction_deg": direction_info["start_direction_deg"],
            "end_direction_deg": direction_info["end_direction_deg"],
            "cumulative_rotation_deg": direction_info["cumulative_rotation_deg"],
            "max_adjacent_rotation_deg": direction_info["max_adjacent_rotation_deg"],
            "direction_behaviour": direction_info["direction_behaviour"],
        })

    return _meta_pd.DataFrame(meta_rows)
# === PATCH: strict-cluster meta-clustering support END ===
