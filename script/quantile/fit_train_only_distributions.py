"""Fit Quantile Mapping distributions using training samples only.

The train index defines every allowed forecast case. For each row, this script
extracts the seven-day ECMWF precipitation accumulation and the matching
seven-day gauge target. Validation and test rows are never read.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ecmwf_path(recorded_path, ecmwf_root):
    recorded_path = Path(recorded_path)
    if recorded_path.is_file():
        return recorded_path
    if ecmwf_root:
        candidate = Path(ecmwf_root) / recorded_path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"ECMWF file not found: {recorded_path}. "
        "Pass --ecmwf-root to replace the directory recorded in train.csv."
    )


def assemble_base_dates(train):
    """Match CustomDataset3's fallback for invalid dates such as non-leap Feb 29."""
    base_dates = pd.to_datetime(
        {"year": train["year"], "month": train["month"], "day": train["day"]},
        errors="coerce",
    )
    invalid = base_dates.isna()
    if invalid.any():
        invalid_month_day = train.loc[invalid, ["month", "day"]].drop_duplicates()
        if not ((invalid_month_day["month"] == 2) & (invalid_month_day["day"] == 29)).all():
            raise ValueError(
                "Invalid train dates other than February 29 were found: "
                f"{invalid_month_day.to_dict(orient='records')}"
            )
        fallback = pd.to_datetime(
            {
                "year": train.loc[invalid, "year"],
                "month": 2,
                "day": 28,
            }
        )
        base_dates.loc[invalid] = fallback.to_numpy()
    return base_dates, invalid


def prepare_output(root, overwrite):
    root = Path(root)
    grid_dir = root / "s2s"
    gauge_dir = root / "gauge"
    existing = list(grid_dir.glob("*.npy")) + list(gauge_dir.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already contains {len(existing)} distribution files: {root}. "
            "Use --overwrite only when intentionally replacing them."
        )
    grid_dir.mkdir(parents=True, exist_ok=True)
    gauge_dir.mkdir(parents=True, exist_ok=True)
    return root, grid_dir, gauge_dir


def fit_grid_distributions(train, args, levels, grid_dir):
    sample_count = len(train)
    samples = np.empty((sample_count, args.height, args.width), dtype=np.float32)
    cursor = 0

    for recorded_path, group in train.groupby("pathECMWF", sort=False):
        path = resolve_ecmwf_path(recorded_path, args.ecmwf_root)
        array = np.load(path, mmap_mode="r")
        if array.ndim != 5:
            raise ValueError(f"Expected [year, feature, lead, H, W] in {path}, got {array.shape}")
        if array.shape[-2:] != (args.height, args.width):
            raise ValueError(
                f"Grid in {path} is {array.shape[-2:]}, expected {(args.height, args.width)}"
            )

        for row in group.itertuples(index=False):
            year_index = int(row.year) - args.year_origin
            lead = int(row.leadTime)
            # Match CustomDataset3.get_ecmwf exactly: [lead-window, lead).
            start = lead - args.window_days
            stop = lead
            if year_index < 0 or year_index >= array.shape[0]:
                raise IndexError(f"Year {row.year} is outside {path} with shape {array.shape}")
            if start < 0 or stop > array.shape[2]:
                raise IndexError(f"Lead {lead} cannot form a {args.window_days}-day window in {path}")
            daily_precipitation = np.asarray(
                array[year_index, args.precip_feature_index, start:stop], dtype=np.float32
            )
            samples[cursor] = np.clip(
                daily_precipitation, 0, args.rain_threshold
            ).sum(axis=0)
            cursor += 1

    if cursor != sample_count:
        raise RuntimeError(f"Extracted {cursor} ECMWF samples, expected {sample_count}")

    quantiles = np.quantile(samples, levels, axis=0).astype(np.float32)
    for row_index in range(args.height):
        for column_index in range(args.width):
            np.save(
                grid_dir / f"{row_index:02d}_{column_index:02d}.npy",
                quantiles[:, row_index, column_index],
            )
    return {
        "n_samples": sample_count,
        "minimum": float(samples.min()),
        "maximum": float(samples.max()),
    }


def fit_gauge_distributions(train, args, levels, gauge_dir):
    gauge = pd.read_csv(args.gauge_csv)
    required = {"Day", "Station", "R", "Lon", "Lat"}
    if not required.issubset(gauge.columns):
        raise ValueError(f"Gauge CSV is missing columns: {sorted(required - set(gauge.columns))}")
    gauge["Day"] = pd.to_datetime(gauge["Day"])
    gauge["R"] = (
        pd.to_numeric(gauge["R"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=args.rain_threshold)
    )

    station_info = gauge[["Station", "Lon", "Lat"]].drop_duplicates("Station")
    daily = gauge.pivot_table(
        index="Day", columns="Station", values="R", aggfunc="sum", fill_value=0.0
    ).sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    weekly = daily.rolling(args.window_days, min_periods=args.window_days).sum()
    fallback_weekly = daily.rolling(
        args.window_days + 1, min_periods=args.window_days + 1
    ).sum()

    base_dates, invalid_base_dates = assemble_base_dates(train)
    target_dates = base_dates + pd.to_timedelta(train["leadTime"], unit="D")
    if target_dates.min() < weekly.index.min() or target_dates.max() > weekly.index.max():
        raise ValueError("Gauge CSV does not cover all train target windows")
    target_values = weekly.reindex(target_dates).reset_index(drop=True)
    if invalid_base_dates.any():
        # CustomDataset3 uses [Feb-28 + lead-window, Feb-28 + lead] for its
        # invalid-date fallback, which is an inclusive window of window+1 days.
        invalid_targets = target_dates[invalid_base_dates]
        target_values.loc[invalid_base_dates.to_numpy(), :] = (
            fallback_weekly.reindex(invalid_targets).to_numpy()
        )
    if target_values.isna().any().any():
        raise ValueError("Gauge extraction produced missing train targets")

    values_by_cell = {}
    station_count_by_cell = {}
    for station in station_info.itertuples(index=False):
        row_index = int(np.clip(np.floor((args.lat_start - float(station.Lat)) / args.step), 0, args.height - 1))
        column_index = int(np.clip(np.floor((float(station.Lon) - args.lon_start) / args.step), 0, args.width - 1))
        key = (row_index, column_index)
        values_by_cell.setdefault(key, []).append(
            target_values[station.Station].to_numpy(dtype=np.float32)
        )
        station_count_by_cell[key] = station_count_by_cell.get(key, 0) + 1

    for (row_index, column_index), chunks in values_by_cell.items():
        values = np.concatenate(chunks)
        quantiles = np.quantile(values, levels).astype(np.float32)
        np.savez_compressed(
            gauge_dir / f"{row_index:02d}_{column_index:02d}.npz",
            quantiles=quantiles,
            levels=levels.astype(np.float32),
        )
    return {
        "n_train_rows": int(len(train)),
        "n_stations": int(len(station_info)),
        "n_station_cells": int(len(values_by_cell)),
        "target_date_min": str(target_dates.min().date()),
        "target_date_max": str(target_dates.max().date()),
        "invalid_base_date_rows": int(invalid_base_dates.sum()),
        "invalid_base_date_policy": (
            "Match CustomDataset3: replace non-leap Feb 29 by Feb 28; "
            "gauge fallback uses an inclusive window_days+1 interval"
        ),
        "stations_per_cell": {
            f"{row:02d}_{column:02d}": count
            for (row, column), count in sorted(station_count_by_cell.items())
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--gauge-csv", required=True, type=Path)
    parser.add_argument("--ecmwf-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("quantile_distributions/train_only"))
    parser.add_argument("--year-origin", type=int, default=2004)
    parser.add_argument("--precip-feature-index", type=int, default=12)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--height", type=int, default=25)
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--lat-start", type=float, default=23.25)
    parser.add_argument("--lon-start", type=float, default=102.25)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--rain-threshold", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train = pd.read_csv(args.train_index)
    required = {"pathECMWF", "leadTime", "year", "month", "day"}
    if not required.issubset(train.columns):
        raise ValueError(f"Train index is missing columns: {sorted(required - set(train.columns))}")
    levels = np.linspace(0, 1, args.levels, dtype=np.float32)
    base_dates, invalid_base_dates = assemble_base_dates(train)
    root, grid_dir, gauge_dir = prepare_output(args.output_root, args.overwrite)

    grid_metadata = fit_grid_distributions(train, args, levels, grid_dir)
    gauge_metadata = fit_gauge_distributions(train, args, levels, gauge_dir)
    metadata = {
        "fit_split": "train_only",
        "train_index": str(args.train_index.resolve()),
        "train_index_sha256": sha256_file(args.train_index),
        "gauge_csv": str(args.gauge_csv.resolve()),
        "gauge_csv_sha256": sha256_file(args.gauge_csv),
        "ecmwf_root_override": None if args.ecmwf_root is None else str(args.ecmwf_root.resolve()),
        "train_base_date_min": str(base_dates.min().date()),
        "train_base_date_max": str(base_dates.max().date()),
        "train_base_year_min": int(train.year.min()),
        "train_base_year_max": int(train.year.max()),
        "quantile_levels": int(args.levels),
        "window_days": int(args.window_days),
        "precip_feature_index": int(args.precip_feature_index),
        "rain_threshold_mm_day": float(args.rain_threshold),
        "invalid_base_date_rows": int(invalid_base_dates.sum()),
        "grid": grid_metadata,
        "gauge": gauge_metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(f"Saved train-only distributions to {root}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
