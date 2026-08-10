"""Evaluation and result-saving utilities for VIFOS experiments.

Every test run can write reproducible artifacts for overall, per-station,
per-lead-time, and heavy-rainfall analyses.  The public ``test_func`` and
``test_func_quantile`` signatures remain backward compatible: all new
arguments are optional.
"""

import csv
import glob
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import utils
from src.utils.loss import get_station_from_grid


# Coordinates and identifiers reported in Appendix 6.4 of the manuscript.
# A station name is assigned only when the coordinates in the batch match.
DEFAULT_STATIONS = (
    {"station_id": "S5", "station_name": "TUAN GIAO", "lat": 21.58, "lon": 103.42},
    {"station_id": "S9", "station_name": "TAM DUONG", "lat": 22.42, "lon": 103.48},
    {"station_id": "S11", "station_name": "MUONG TE", "lat": 22.37, "lon": 102.83},
    {"station_id": "S13", "station_name": "SIN HO", "lat": 22.37, "lon": 103.23},
    {"station_id": "S25", "station_name": "PHA DIN", "lat": 21.57, "lon": 103.52},
    {"station_id": "S29", "station_name": "DIEN BIEN", "lat": 21.37, "lon": 103.00},
    {"station_id": "S30", "station_name": "SON LA", "lat": 21.33, "lon": 103.90},
    {"station_id": "S32", "station_name": "BAC YEN", "lat": 21.25, "lon": 104.42},
    {"station_id": "S37", "station_name": "CO NOI", "lat": 21.13, "lon": 104.15},
    {"station_id": "S38", "station_name": "SONG MA", "lat": 21.07, "lon": 103.73},
    {"station_id": "S39", "station_name": "YEN CHAU", "lat": 21.05, "lon": 104.30},
    {"station_id": "S44", "station_name": "MOC CHAU", "lat": 20.83, "lon": 104.67},
)

SUMMARY_METRICS = (
    "mae",
    "mse",
    "mape",
    "rmse",
    "r2",
    "r2_pooled",
    "corr",
    "bias",
)


def to_float(x, device):
    if isinstance(x, list):
        return [x_i.to(device).float() for x_i in x]
    return x.to(device).float()


def _finite_pairs(y_prd, y_grt):
    y_prd = np.asarray(y_prd, dtype=np.float64)
    y_grt = np.asarray(y_grt, dtype=np.float64)
    if y_prd.shape != y_grt.shape:
        raise ValueError(
            f"Prediction and ground-truth shapes differ: {y_prd.shape} vs {y_grt.shape}"
        )
    valid = np.isfinite(y_prd) & np.isfinite(y_grt)
    return y_prd, y_grt, valid


def _safe_corr(y_prd, y_grt):
    y_prd, y_grt, valid = _finite_pairs(y_prd, y_grt)
    prediction = y_prd[valid]
    observation = y_grt[valid]
    if prediction.size < 2 or np.std(prediction) == 0 or np.std(observation) == 0:
        return float("nan")
    return float(np.corrcoef(observation, prediction)[0, 1])


def _safe_r2(y_prd, y_grt, pooled=False):
    y_prd, y_grt, valid = _finite_pairs(y_prd, y_grt)
    if not np.any(valid):
        return float("nan")

    if pooled or y_prd.ndim == 1:
        prediction = y_prd[valid]
        observation = y_grt[valid]
        if observation.size < 2 or np.var(observation) == 0:
            return float("nan")
        return float(r2_score(observation, prediction))

    if y_grt.shape[0] < 2:
        return float("nan")

    # Preserve the legacy result used by the current paper: sklearn averages
    # the R2 values of all station columns for a two-dimensional input.
    if np.all(valid):
        try:
            return float(r2_score(y_grt, y_prd))
        except ValueError:
            return float("nan")
    return _safe_r2(y_prd, y_grt, pooled=True)


def cal_acc(y_prd, y_grt):
    """Return the six legacy metrics in their original order.

    ``r2`` intentionally retains sklearn's multi-output averaging when the
    inputs are two-dimensional.  ``metric_dict`` additionally reports a
    pooled R2 that matches the single-vector formula shown in the manuscript.
    """
    y_prd, y_grt, valid = _finite_pairs(y_prd, y_grt)
    prediction = y_prd[valid]
    observation = y_grt[valid]
    if prediction.size == 0:
        return (float("nan"),) * 6

    mae = float(mean_absolute_error(observation, prediction))
    mse = float(mean_squared_error(observation, prediction))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(mean_absolute_percentage_error(observation, prediction))
    rmse = float(np.sqrt(mse))
    r2 = _safe_r2(y_prd, y_grt, pooled=False)
    corr = _safe_corr(y_prd, y_grt)
    return mae, mse, mape, rmse, r2, corr


def metric_dict(y_prd, y_grt):
    mae, mse, mape, rmse, r2, corr = cal_acc(y_prd, y_grt)
    y_prd, y_grt, valid = _finite_pairs(y_prd, y_grt)
    bias = float(np.mean(y_prd[valid] - y_grt[valid])) if np.any(valid) else float("nan")
    return {
        "n": int(np.sum(valid)),
        "mae": mae,
        "mse": mse,
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
        "r2_pooled": _safe_r2(y_prd, y_grt, pooled=True),
        "corr": corr,
        "bias": bias,
    }


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_2d(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.reshape(1, -1)
    return values


def _extract_lead_times(lead_time, batch_size):
    values = _to_numpy(lead_time)
    if values.ndim == 0:
        return np.repeat(float(values), batch_size)
    values = values.reshape(batch_size, -1)
    if values.shape[1] == 1:
        return values[:, 0].astype(float)

    # Support one-hot lead-time encodings without changing the model input.
    is_one_hot = np.all((values == 0) | (values == 1)) and np.allclose(
        values.sum(axis=1), 1
    )
    if is_one_hot:
        indices = np.argmax(values, axis=1).astype(float)
        return indices + 7 if values.shape[1] == 40 else indices
    return values[:, 0].astype(float)


def _extract_coords(y_with_metadata):
    values = _to_numpy(y_with_metadata)
    if values.ndim >= 3 and values.shape[-1] >= 3:
        # Dataset convention used by test_func_quantile: [rain, lon, lat].
        return values[:, :, 1].astype(float), values[:, :, 2].astype(float)
    return None, None


def _match_station(station_index, lon, lat, tolerance=0.06):
    fallback = {
        "station_index": station_index,
        "station_id": f"station_{station_index:02d}",
        "station_name": f"Station {station_index}",
        "longitude": _optional_float(lon),
        "latitude": _optional_float(lat),
    }
    if lon is None or lat is None or not np.isfinite(lon) or not np.isfinite(lat):
        return fallback

    nearest = min(
        DEFAULT_STATIONS,
        key=lambda station: (station["lat"] - lat) ** 2 + (station["lon"] - lon) ** 2,
    )
    distance = math.sqrt(
        (nearest["lat"] - lat) ** 2 + (nearest["lon"] - lon) ** 2
    )
    if distance > tolerance:
        return fallback
    return {
        "station_index": station_index,
        "station_id": nearest["station_id"],
        "station_name": nearest["station_name"],
        "longitude": float(lon),
        "latitude": float(lat),
    }


def _optional_float(value):
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, ensure_ascii=False)


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _normalise_thresholds(extreme_thresholds):
    if not extreme_thresholds:
        return {}
    output = {}
    for label, value in extreme_thresholds.items():
        value = _optional_float(value)
        if value is not None:
            output[str(label)] = value
    return dict(sorted(output.items(), key=lambda item: item[1]))


def load_extreme_thresholds(path=None, manual_thresholds=None):
    """Load named heavy-rainfall thresholds from train-derived JSON or a mapping."""
    thresholds = {}
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        thresholds.update(payload.get("percentiles", payload))
    if manual_thresholds:
        thresholds.update(manual_thresholds)
    return _normalise_thresholds(thresholds)


def compute_training_percentiles(
    train_dataset,
    config,
    output_scaler,
    output_path,
    percentiles=(90, 95, 99),
):
    """Compute rainfall thresholds once from training observations only."""
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUMBER_WORKERS,
        collate_fn=utils.custom_collate_fn,
    )
    observations = []
    for data in tqdm(dataloader, desc="Training rainfall percentiles"):
        y_grt = _to_numpy(data["y"])
        if y_grt.ndim < 3:
            raise ValueError(f"Expected y with shape [B, station, feature], got {y_grt.shape}")
        rain = y_grt[:, :, 0]
        if config.TRAIN.OUTPUT_NORM:
            rain = output_scaler.inverse_transform(rain)
        rain = np.clip(rain, 0, config.DATA.RAIN_THRESHOLD)
        observations.append(rain.reshape(-1))

    if not observations:
        raise ValueError("Training dataset is empty; cannot compute rainfall percentiles.")
    observations = np.concatenate(observations)
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError("Training rainfall contains no finite values.")

    percentile_values = {
        f"p{int(percentile)}": float(np.percentile(observations, percentile))
        for percentile in percentiles
    }
    payload = {
        "source_split": "train",
        "unit": "mm/week",
        "n_observations": int(observations.size),
        "minimum": float(np.min(observations)),
        "maximum": float(np.max(observations)),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "percentiles": percentile_values,
    }
    _write_json(output_path, payload)
    return percentile_values


def _summary_rows(sources, observation, seed):
    rows = []
    for source_name, prediction in sources.items():
        rows.append(
            {
                "seed": seed,
                "forecast_source": source_name,
                **metric_dict(prediction, observation),
            }
        )
    return rows


def _station_rows(sources, observation, longitudes, latitudes, seed):
    rows = []
    station_count = observation.shape[1]
    for station_index in range(station_count):
        lon = None if longitudes is None else np.nanmedian(longitudes[:, station_index])
        lat = None if latitudes is None else np.nanmedian(latitudes[:, station_index])
        station = _match_station(station_index, lon, lat)
        for source_name, prediction in sources.items():
            rows.append(
                {
                    "seed": seed,
                    "forecast_source": source_name,
                    **station,
                    **metric_dict(
                        prediction[:, station_index], observation[:, station_index]
                    ),
                }
            )
    return rows


def _lead_time_rows(sources, observation, lead_times, seed):
    rows = []
    for lead_time in np.unique(lead_times):
        selected = np.isclose(lead_times, lead_time)
        for source_name, prediction in sources.items():
            rows.append(
                {
                    "seed": seed,
                    "forecast_source": source_name,
                    "lead_time": float(lead_time),
                    "n_samples": int(np.sum(selected)),
                    **metric_dict(prediction[selected], observation[selected]),
                }
            )
    return rows


def _extreme_rows(sources, observation, thresholds, seed):
    rows = []
    thresholds = _normalise_thresholds(thresholds)
    for threshold_label, threshold in thresholds.items():
        observed_event = observation >= threshold
        for source_name, prediction in sources.items():
            predicted_event = prediction >= threshold
            hits = int(np.sum(observed_event & predicted_event))
            misses = int(np.sum(observed_event & ~predicted_event))
            false_alarms = int(np.sum(~observed_event & predicted_event))
            correct_negatives = int(np.sum(~observed_event & ~predicted_event))
            pod_denominator = hits + misses
            far_denominator = hits + false_alarms
            csi_denominator = hits + misses + false_alarms
            rows.append(
                {
                    "seed": seed,
                    "forecast_source": source_name,
                    "threshold_label": threshold_label,
                    "threshold_mm_week": threshold,
                    "n": int(observation.size),
                    "observed_events": pod_denominator,
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "correct_negatives": correct_negatives,
                    "pod": hits / pod_denominator if pod_denominator else float("nan"),
                    "far": false_alarms / far_denominator if far_denominator else float("nan"),
                    "csi": hits / csi_denominator if csi_denominator else float("nan"),
                    "frequency_bias": (
                        (hits + false_alarms) / pod_denominator
                        if pod_denominator
                        else float("nan")
                    ),
                }
            )
    return rows


def _intensity_rows(sources, observation, thresholds, seed):
    thresholds = list(_normalise_thresholds(thresholds).items())
    if not thresholds:
        return []
    rows = []
    lower_value = -np.inf
    lower_label = "min"
    classes = []
    for label, value in thresholds:
        classes.append((f"{lower_label}_to_{label}", lower_value, value))
        lower_value = value
        lower_label = label
    classes.append((f"{lower_label}_and_above", lower_value, np.inf))

    for class_name, lower, upper in classes:
        selected = (observation >= lower) & (observation < upper)
        if not np.any(selected):
            continue
        for source_name, prediction in sources.items():
            rows.append(
                {
                    "seed": seed,
                    "forecast_source": source_name,
                    "intensity_class": class_name,
                    "lower_mm_week": None if not np.isfinite(lower) else lower,
                    "upper_mm_week": None if not np.isfinite(upper) else upper,
                    **metric_dict(prediction[selected], observation[selected]),
                }
            )
    return rows


def _prediction_rows(
    sources,
    observation,
    lead_times,
    longitudes,
    latitudes,
    seed,
):
    rows = []
    sample_count, station_count = observation.shape
    for station_index in range(station_count):
        lon = None if longitudes is None else np.nanmedian(longitudes[:, station_index])
        lat = None if latitudes is None else np.nanmedian(latitudes[:, station_index])
        station = _match_station(station_index, lon, lat)
        for sample_index in range(sample_count):
            row = {
                "seed": seed,
                "sample_index": sample_index,
                "lead_time": float(lead_times[sample_index]),
                **station,
                "observation": float(observation[sample_index, station_index]),
            }
            for source_name, prediction in sources.items():
                row[f"prediction_{source_name}"] = float(
                    prediction[sample_index, station_index]
                )
            rows.append(row)
    return rows


def save_evaluation_results(
    prediction,
    observation,
    ecmwf,
    lead_times,
    result_dir,
    seed,
    prediction_name="model",
    longitudes=None,
    latitudes=None,
    extreme_thresholds=None,
    run_metadata=None,
):
    """Save all artifacts required by the reviewer analyses."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction = _ensure_2d(prediction).astype(np.float64)
    observation = _ensure_2d(observation).astype(np.float64)
    ecmwf = _ensure_2d(ecmwf).astype(np.float64)
    lead_times = np.asarray(lead_times, dtype=np.float64).reshape(-1)
    if not (prediction.shape == observation.shape == ecmwf.shape):
        raise ValueError(
            "Prediction, observation, and ECMWF arrays must have the same shape: "
            f"{prediction.shape}, {observation.shape}, {ecmwf.shape}"
        )
    if lead_times.size != observation.shape[0]:
        raise ValueError(
            f"Expected {observation.shape[0]} lead times, got {lead_times.size}."
        )

    if longitudes is not None:
        longitudes = _ensure_2d(longitudes).astype(np.float64)
    if latitudes is not None:
        latitudes = _ensure_2d(latitudes).astype(np.float64)
    sources = {prediction_name: prediction, "ecmwf_s2s": ecmwf}
    thresholds = _normalise_thresholds(extreme_thresholds)

    summary_rows = _summary_rows(sources, observation, seed)
    station_rows = _station_rows(sources, observation, longitudes, latitudes, seed)
    lead_rows = _lead_time_rows(sources, observation, lead_times, seed)
    extreme_rows = _extreme_rows(sources, observation, thresholds, seed)
    intensity_rows = _intensity_rows(sources, observation, thresholds, seed)
    prediction_rows = _prediction_rows(
        sources,
        observation,
        lead_times,
        longitudes,
        latitudes,
        seed,
    )

    _write_csv(result_dir / "metrics_summary.csv", summary_rows)
    _write_csv(result_dir / "per_station_metrics.csv", station_rows)
    _write_csv(result_dir / "per_lead_time_metrics.csv", lead_rows)
    _write_csv(result_dir / "predictions_long.csv", prediction_rows)
    if extreme_rows:
        _write_csv(result_dir / "extreme_metrics.csv", extreme_rows)
        _write_csv(result_dir / "intensity_class_metrics.csv", intensity_rows)

    np.savez_compressed(
        result_dir / "predictions_arrays.npz",
        prediction=prediction,
        observation=observation,
        ecmwf_s2s=ecmwf,
        lead_time=lead_times,
        longitude=np.array([]) if longitudes is None else longitudes,
        latitude=np.array([]) if latitudes is None else latitudes,
    )
    metrics_payload = {
        "seed": seed,
        "prediction_name": prediction_name,
        "thresholds_mm_week": thresholds,
        "overall": {row["forecast_source"]: row for row in summary_rows},
    }
    _write_json(result_dir / "metrics_summary.json", metrics_payload)

    metadata = dict(run_metadata or {})
    metadata.update(
        {
            "seed": seed,
            "prediction_name": prediction_name,
            "result_dir": str(result_dir),
            "n_test_samples": int(observation.shape[0]),
            "n_stations": int(observation.shape[1]),
            "lead_times": sorted(float(value) for value in np.unique(lead_times)),
            "extreme_thresholds_mm_week": thresholds,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(result_dir / "run_metadata.json", metadata)
    return {
        "summary": summary_rows,
        "metadata": metadata,
        "result_dir": str(result_dir),
    }


def _student_t_critical_95(degrees_of_freedom):
    # Two-sided t critical values for the seed counts normally used here.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    eligible = [key for key in table if key <= degrees_of_freedom]
    return table[max(eligible)] if eligible else 1.96


def _aggregate_values(values):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(values))
    if values.size == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = float(np.std(values, ddof=1))
    half_width = _student_t_critical_95(values.size - 1) * std / math.sqrt(values.size)
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate_detail_files(
    experiment_dir,
    input_filename,
    output_filename,
    group_keys,
    metric_names,
):
    all_rows = []
    for path in sorted(Path(experiment_dir).glob(f"seed_*/{input_filename}")):
        all_rows.extend(_read_csv(path))
    if not all_rows:
        return

    grouped = {}
    for row in all_rows:
        key = tuple(row.get(column, "") for column in group_keys)
        grouped.setdefault(key, []).append(row)

    output_rows = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        base = dict(zip(group_keys, key))
        if base.get("forecast_source") == "ecmwf_s2s":
            # ECMWF S2S is deterministic here. Repeating the same baseline in
            # every seed must not create a false zero-width seed confidence interval.
            rows = rows[:1]
        for metric_name in metric_names:
            stats = _aggregate_values([_as_float(row.get(metric_name)) for row in rows])
            output_rows.append({**base, "metric": metric_name, **stats})
    _write_csv(Path(experiment_dir) / output_filename, output_rows)


def aggregate_seed_results(experiment_dir):
    """Update mean, standard deviation, and 95% CI after every completed seed."""
    experiment_dir = Path(experiment_dir)
    seed_rows = []
    for path in sorted(experiment_dir.glob("seed_*/metrics_summary.csv")):
        seed_rows.extend(_read_csv(path))
    if not seed_rows:
        return None

    seed_rows.sort(key=lambda row: (row.get("forecast_source", ""), int(float(row["seed"]))))
    _write_csv(experiment_dir / "seed_metrics.csv", seed_rows)

    aggregate_rows = []
    sources = sorted({row["forecast_source"] for row in seed_rows})
    for source in sources:
        source_rows = [row for row in seed_rows if row["forecast_source"] == source]
        if source == "ecmwf_s2s":
            source_rows = source_rows[:1]
        for metric_name in SUMMARY_METRICS:
            stats = _aggregate_values([_as_float(row.get(metric_name)) for row in source_rows])
            aggregate_rows.append(
                {"forecast_source": source, "metric": metric_name, **stats}
            )
    _write_csv(experiment_dir / "aggregate_metrics.csv", aggregate_rows)

    _aggregate_detail_files(
        experiment_dir,
        "per_station_metrics.csv",
        "aggregate_per_station.csv",
        ("forecast_source", "station_index", "station_id", "station_name", "longitude", "latitude"),
        ("mae", "rmse", "r2", "r2_pooled", "corr", "bias"),
    )
    _aggregate_detail_files(
        experiment_dir,
        "per_lead_time_metrics.csv",
        "aggregate_per_lead_time.csv",
        ("forecast_source", "lead_time"),
        ("mae", "rmse", "r2", "r2_pooled", "corr", "bias"),
    )
    _aggregate_detail_files(
        experiment_dir,
        "extreme_metrics.csv",
        "aggregate_extreme_metrics.csv",
        ("forecast_source", "threshold_label", "threshold_mm_week"),
        ("pod", "far", "csi", "frequency_bias"),
    )
    return str(experiment_dir / "aggregate_metrics.csv")


def compare_experiments(group_dir, experiment_a, experiment_b):
    """Create paired seed tests after both experiments have matching seeds."""
    group_dir = Path(group_dir)
    rows_by_experiment = {}
    for experiment in (experiment_a, experiment_b):
        path = group_dir / experiment / "seed_metrics.csv"
        if not path.exists():
            return None
        rows_by_experiment[experiment] = {
            int(float(row["seed"])): row
            for row in _read_csv(path)
            if row.get("forecast_source") not in ("ecmwf_s2s", "s2s")
        }

    common_seeds = sorted(
        set(rows_by_experiment[experiment_a])
        & set(rows_by_experiment[experiment_b])
    )
    if len(common_seeds) < 2:
        return None

    try:
        from scipy.stats import ttest_rel
    except ImportError:
        ttest_rel = None

    comparison_rows = []
    for metric_name in ("mae", "rmse", "r2", "r2_pooled", "corr", "bias"):
        values_a = np.asarray(
            [_as_float(rows_by_experiment[experiment_a][seed][metric_name]) for seed in common_seeds]
        )
        values_b = np.asarray(
            [_as_float(rows_by_experiment[experiment_b][seed][metric_name]) for seed in common_seeds]
        )
        valid = np.isfinite(values_a) & np.isfinite(values_b)
        differences = values_a[valid] - values_b[valid]
        stats = _aggregate_values(differences.tolist())
        test = ttest_rel(values_a[valid], values_b[valid]) if ttest_rel and np.sum(valid) >= 2 else None
        comparison_rows.append(
            {
                "experiment_a": experiment_a,
                "experiment_b": experiment_b,
                "metric": metric_name,
                "paired_seeds": ",".join(str(seed) for seed in np.asarray(common_seeds)[valid]),
                "n_pairs": int(np.sum(valid)),
                "mean_a_minus_b": stats["mean"],
                "std_difference": stats["std"],
                "ci95_low_difference": stats["ci95_low"],
                "ci95_high_difference": stats["ci95_high"],
                "paired_t_statistic": None if test is None else float(test.statistic),
                "paired_t_pvalue": None if test is None else float(test.pvalue),
            }
        )

    comparison_dir = group_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_path = comparison_dir / f"{experiment_a}_vs_{experiment_b}.csv"
    _write_csv(output_path, comparison_rows)
    return str(output_path)


def _log_to_wandb(summary_rows, prediction, observation, ecmwf):
    if not wandb.run:
        return
    for row in summary_rows:
        source = row["forecast_source"]
        wandb.log(
            {
                f"{source}/{metric}": value
                for metric, value in row.items()
                if metric in SUMMARY_METRICS and value is not None
            }
        )

    for index in range(min(10, prediction.shape[0])):
        figure = plt.figure(figsize=(12, 4))
        plt.plot(prediction[index], label="Prediction", marker="o")
        plt.plot(observation[index], label="Ground truth", marker="x")
        plt.plot(ecmwf[index], label="ECMWF S2S", marker="s")
        plt.xlabel("Station index")
        plt.ylabel("Weekly rainfall (mm)")
        plt.title(f"Test sample {index}")
        plt.legend()
        plt.grid(True)
        wandb.log({f"Output/Image{index}": wandb.Image(figure)})
        plt.close(figure)


def test_func(
    model,
    test_dataset,
    criterion,
    config,
    input_scaler,
    output_scaler,
    device,
    result_dir=None,
    run_metadata=None,
    extreme_thresholds=None,
    prediction_name="model",
):
    model.eval()
    list_prd = []
    list_grt = []
    list_ecmwf = []
    list_lead_time = []
    list_lon = []
    list_lat = []
    epoch_loss = 0.0
    batch_count = 0
    model_forward_seconds = 0.0
    model.to(device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUMBER_WORKERS,
        collate_fn=utils.custom_collate_fn,
    )

    print("********** Starting testing process **********")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            input_data = data["x"].to(device)
            lead_time = data["lead_time"].to(device)
            y_with_metadata = data["y"].to(device)
            ecmwf = data["ecmwf"].to(device)
            batch_size = y_with_metadata.shape[0]
            list_lead_time.append(_extract_lead_times(lead_time, batch_size))
            lon, lat = _extract_coords(y_with_metadata)
            if lon is not None:
                list_lon.append(lon)
                list_lat.append(lat)

            ecmwf = ecmwf[:, 12, -config.MODEL.ECMWF_TIME_STEP :, :, :]
            ecmwf = torch.sum(ecmwf, dim=1)
            ecmwf = torch.unsqueeze(ecmwf, dim=-1)

            h = data.get("h")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            y_prd = (
                model([input_data, lead_time, h.to(device)])
                if h is not None
                else model([input_data, lead_time])
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_forward_seconds += time.perf_counter() - forward_start
            y_prd = get_station_from_grid(y_prd, y_with_metadata, config)[:, :, 0]
            ecmwf = get_station_from_grid(ecmwf, y_with_metadata, config)[:, :, 0]
            y_grt = y_with_metadata[:, :, 0]

            batch_loss = criterion(y_prd, y_grt)
            y_prd = _to_numpy(y_prd)
            y_grt = _to_numpy(y_grt)
            ecmwf = _to_numpy(ecmwf)

            if config.TRAIN.OUTPUT_NORM:
                y_prd = output_scaler.inverse_transform(y_prd)
                y_grt = output_scaler.inverse_transform(y_grt)

            y_prd = np.clip(y_prd, 0, config.DATA.RAIN_THRESHOLD)
            y_grt = np.clip(y_grt, 0, config.DATA.RAIN_THRESHOLD)
            ecmwf = np.clip(ecmwf, 0, config.DATA.RAIN_THRESHOLD)
            list_prd.append(_ensure_2d(y_prd))
            list_grt.append(_ensure_2d(y_grt))
            list_ecmwf.append(_ensure_2d(ecmwf))
            epoch_loss += float(batch_loss.item())
            batch_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_to_end_seconds = time.perf_counter() - start_time
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    list_prd = np.concatenate(list_prd, axis=0)
    list_grt = np.concatenate(list_grt, axis=0)
    list_ecmwf = np.concatenate(list_ecmwf, axis=0)
    lead_times = np.concatenate(list_lead_time, axis=0)
    longitudes = np.concatenate(list_lon, axis=0) if list_lon else None
    latitudes = np.concatenate(list_lat, axis=0) if list_lat else None

    metadata = dict(run_metadata or {})
    metadata.update(
        {
            "test_loss_mean_batch": epoch_loss / batch_count if batch_count else None,
            "test_batches": batch_count,
            "test_end_to_end_seconds": end_to_end_seconds,
            "model_forward_seconds": model_forward_seconds,
            "model_forward_ms_per_sample": (
                1000 * model_forward_seconds / len(list_grt)
            ),
            "test_peak_gpu_memory_mb": peak_gpu_memory_mb,
        }
    )
    if result_dir is None:
        result_dir = Path("experiment_results") / "untracked_run"
    saved = save_evaluation_results(
        prediction=list_prd,
        observation=list_grt,
        ecmwf=list_ecmwf,
        lead_times=lead_times,
        result_dir=result_dir,
        seed=int(config.MODEL.SEED),
        prediction_name=prediction_name,
        longitudes=longitudes,
        latitudes=latitudes,
        extreme_thresholds=extreme_thresholds,
        run_metadata=metadata,
    )
    if config.WANDB.STATUS:
        _log_to_wandb(saved["summary"], list_prd, list_grt, list_ecmwf)

    model_metrics = saved["summary"][0]
    s2s_metrics = saved["summary"][1]
    print(
        "Model - "
        f"MAE: {model_metrics['mae']:.4f}, RMSE: {model_metrics['rmse']:.4f}, "
        f"R2: {model_metrics['r2']:.4f}, Corr: {model_metrics['corr']:.4f}"
    )
    print(
        "ECMWF S2S - "
        f"MAE: {s2s_metrics['mae']:.4f}, RMSE: {s2s_metrics['rmse']:.4f}, "
        f"R2: {s2s_metrics['r2']:.4f}, Corr: {s2s_metrics['corr']:.4f}"
    )
    print(f"Saved test results to: {saved['result_dir']}")
    return saved


def load_quantile_distribution(grid_dir, gauge_dir):
    grid_distribution = {}
    for file_path in glob.glob(os.path.join(grid_dir, "*.npy")):
        key = os.path.basename(file_path).replace(".npy", "")
        grid_distribution[key] = np.load(file_path)

    gauge_distribution = {}
    for file_path in glob.glob(os.path.join(gauge_dir, "*.npz")):
        key = os.path.basename(file_path).replace(".npz", "")
        data = np.load(file_path)
        gauge_distribution[key] = (data["quantiles"], data["levels"])
    return grid_distribution, gauge_distribution


def quantile_mapping(x, grid_quantiles, gauge_quantiles, levels):
    x = np.clip(x, grid_quantiles[0], grid_quantiles[-1])
    probability = np.interp(x, grid_quantiles, levels)
    return np.interp(probability, levels, gauge_quantiles)


def get_grid_index(lat, lon, config, step=0.125):
    lat_idx = int(np.floor((config.DATA.LAT_START - lat) / step))
    lon_idx = int(np.floor((lon - config.DATA.LON_START) / step))
    lat_idx = np.clip(lat_idx, 0, config.DATA.HEIGHT - 1)
    lon_idx = np.clip(lon_idx, 0, config.DATA.WIDTH - 1)
    return lat_idx, lon_idx


def test_func_quantile(
    test_dataset,
    criterion,
    config,
    input_scaler,
    output_scaler,
    device,
    result_dir=None,
    run_metadata=None,
    extreme_thresholds=None,
):
    del criterion, input_scaler, device  # Kept in the signature for compatibility.
    list_prd = []
    list_grt = []
    list_ecmwf = []
    list_lead_time = []
    list_lon = []
    list_lat = []
    grid_distribution, gauge_distribution = load_quantile_distribution(
        "s2s/distribution", "gauss/distribution"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUMBER_WORKERS,
        collate_fn=utils.custom_collate_fn,
    )

    print("********** Starting quantile-mapping testing process **********")
    start_time = time.perf_counter()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            lead_time = data["lead_time"]
            y_with_metadata = data["y"]
            ecmwf = data["ecmwf"]
            batch_size = y_with_metadata.shape[0]
            list_lead_time.append(_extract_lead_times(lead_time, batch_size))
            lon, lat = _extract_coords(y_with_metadata)
            if lon is not None:
                list_lon.append(lon)
                list_lat.append(lat)

            ecmwf = ecmwf[:, 12, -config.MODEL.ECMWF_TIME_STEP :, :, :]
            ecmwf = torch.sum(ecmwf, dim=1)
            ecmwf = torch.unsqueeze(ecmwf, dim=-1)
            ecmwf_station = get_station_from_grid(ecmwf, y_with_metadata, config)
            ecmwf_station = ecmwf_station[:, :, 0].cpu().numpy()
            coordinates = y_with_metadata.cpu().numpy()
            batch_size, station_count = ecmwf_station.shape
            y_prd = np.zeros((batch_size, station_count), dtype=np.float32)

            for batch_index in range(batch_size):
                for station_index in range(station_count):
                    longitude = coordinates[batch_index, station_index, 1]
                    latitude = coordinates[batch_index, station_index, 2]
                    row_index, column_index = get_grid_index(
                        latitude, longitude, config
                    )
                    key = f"{row_index:02d}_{column_index:02d}"
                    raw_value = ecmwf_station[batch_index, station_index]
                    if key not in grid_distribution or key not in gauge_distribution:
                        y_prd[batch_index, station_index] = raw_value
                        continue
                    grid_quantiles = grid_distribution[key]
                    gauge_quantiles, levels = gauge_distribution[key]
                    y_prd[batch_index, station_index] = quantile_mapping(
                        raw_value, grid_quantiles, gauge_quantiles, levels
                    )

            y_grt = y_with_metadata[:, :, 0].cpu().numpy()
            if config.TRAIN.OUTPUT_NORM:
                y_grt = output_scaler.inverse_transform(y_grt)
            y_prd = np.clip(y_prd, 0, config.DATA.RAIN_THRESHOLD)
            y_grt = np.clip(y_grt, 0, config.DATA.RAIN_THRESHOLD)
            ecmwf_station = np.clip(ecmwf_station, 0, config.DATA.RAIN_THRESHOLD)
            list_prd.append(_ensure_2d(y_prd))
            list_grt.append(_ensure_2d(y_grt))
            list_ecmwf.append(_ensure_2d(ecmwf_station))

    prediction = np.concatenate(list_prd, axis=0)
    observation = np.concatenate(list_grt, axis=0)
    ecmwf = np.concatenate(list_ecmwf, axis=0)
    lead_times = np.concatenate(list_lead_time, axis=0)
    longitudes = np.concatenate(list_lon, axis=0) if list_lon else None
    latitudes = np.concatenate(list_lat, axis=0) if list_lat else None
    inference_seconds = time.perf_counter() - start_time
    metadata = dict(run_metadata or {})
    metadata.update(
        {
            "test_end_to_end_seconds": inference_seconds,
            "quantile_mapping_ms_per_sample": 1000 * inference_seconds / len(observation),
            "test_peak_gpu_memory_mb": 0.0,
        }
    )
    if result_dir is None:
        result_dir = Path("experiment_results") / "quantile_mapping" / f"seed_{config.MODEL.SEED}"
    saved = save_evaluation_results(
        prediction=prediction,
        observation=observation,
        ecmwf=ecmwf,
        lead_times=lead_times,
        result_dir=result_dir,
        seed=int(config.MODEL.SEED),
        prediction_name="quantile_mapping",
        longitudes=longitudes,
        latitudes=latitudes,
        extreme_thresholds=extreme_thresholds,
        run_metadata=metadata,
    )
    if config.WANDB.STATUS:
        _log_to_wandb(saved["summary"], prediction, observation, ecmwf)
    print(f"Saved quantile-mapping results to: {saved['result_dir']}")
    return saved
