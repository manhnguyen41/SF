import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils import test_func_only as evaluation
from src.utils.evaluation_io import require_checkpoint, resolve_checkpoint_path


def test_perfect_prediction_metrics():
    observation = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    metrics = evaluation.metric_dict(observation.copy(), observation)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["bias"] == 0
    assert metrics["r2_station_mean"] == pytest.approx(1)
    assert metrics["r2_pooled"] == pytest.approx(1)
    assert metrics["corr"] == pytest.approx(1)


def test_constant_observation_and_no_events_return_nan():
    observation = np.ones((4, 2))
    metrics = evaluation.metric_dict(np.ones((4, 2)), observation)
    assert math.isnan(metrics["r2_station_mean"])
    assert math.isnan(metrics["r2_pooled"])
    assert math.isnan(metrics["corr"])
    row = evaluation._extreme_rows(
        {"model": np.zeros((4, 2))}, observation, {"p99": 10}, 52
    )[0]
    assert math.isnan(row["pod"])
    assert math.isnan(row["far"])
    assert math.isnan(row["csi"])


def test_extreme_hit_miss_false_alarm_example():
    observation = np.array([[10.0, 0.0, 10.0, 0.0]])
    prediction = np.array([[10.0, 10.0, 0.0, 0.0]])
    row = evaluation._extreme_rows(
        {"model": prediction}, observation, {"threshold": 5}, 52
    )[0]
    assert (row["hits"], row["misses"], row["false_alarms"]) == (1, 1, 1)
    assert row["pod"] == pytest.approx(0.5)
    assert row["far"] == pytest.approx(0.5)
    assert row["csi"] == pytest.approx(1 / 3)


def test_five_seed_student_t_aggregate_uses_sample_std():
    stats = evaluation._aggregate_values([1, 2, 3, 4, 5])
    expected_std = np.std([1, 2, 3, 4, 5], ddof=1)
    half_width = 2.776 * expected_std / math.sqrt(5)
    assert stats["n"] == 5
    assert stats["std"] == pytest.approx(expected_std)
    assert stats["ci95_low"] == pytest.approx(3 - half_width)
    assert stats["ci95_high"] == pytest.approx(3 + half_width)


def test_paired_comparison_rejects_mismatched_sample_checksum(tmp_path):
    group = tmp_path / "group"
    fields = {
        "forecast_source": "model", "mae": 1, "mse": 1, "rmse": 1,
        "r2_station_mean": 0.5, "r2_pooled": 0.5, "corr": 0.5, "bias": 0,
    }
    for experiment, checksum in (("full_vifos", "same"), ("cnn_lstm", "different")):
        rows = []
        for seed in (52, 62):
            rows.append({"seed": seed, **fields})
            seed_dir = group / experiment / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "run_metadata.json").write_text(
                json.dumps({"test_sample_checksum": checksum}), encoding="utf-8"
            )
        evaluation._write_csv(group / experiment / "seed_metrics.csv", rows)
    with pytest.raises(ValueError, match="checksums differ"):
        evaluation.compare_experiments(group, "full_vifos", "cnn_lstm")


def test_lead_time_output_contains_7_through_46():
    lead_times = np.arange(7, 47)
    observation = np.arange(40, dtype=float).reshape(-1, 1)
    rows = evaluation._lead_time_rows(
        {"model": observation, "ecmwf_s2s": observation},
        observation,
        lead_times,
        52,
    )
    assert {int(row["lead_time"]) for row in rows} == set(range(7, 47))
    assert len(rows) == 80


def test_station_name_is_not_assigned_when_coordinates_do_not_match():
    station = evaluation._match_station(0, lon=0.0, lat=0.0)
    assert station["station_id"] == "station_00"
    assert station["station_name"] == "Station 0"


def test_missing_checkpoint_fails_before_training(tmp_path, monkeypatch):
    config = SimpleNamespace(
        MODEL=SimpleNamespace(SEED=52),
        WANDB=SimpleNamespace(GROUP_NAME="missing", SESSION_NAME="missing-session"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIFOS_CHECKPOINT_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="Paths tried"):
        require_checkpoint(config)


def test_strans_v4b_resolves_legacy_and_new_session_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIFOS_CHECKPOINT_PATH", raising=False)
    session = "prefix_Modelv1_PS-3_Seed-52_suffix"
    config = SimpleNamespace(
        MODEL=SimpleNamespace(SEED=52, NAME="strans-v4b"),
        WANDB=SimpleNamespace(GROUP_NAME="group", SESSION_NAME=session),
    )
    checkpoint_dir = tmp_path / "saved_checkpoints" / "group"
    checkpoint_dir.mkdir(parents=True)
    expected = checkpoint_dir / session.replace("_Modelv1_", "_Strans-V4b_")
    expected = expected.with_suffix(".pt")
    expected.touch()
    resolved, _ = resolve_checkpoint_path(config)
    assert resolved == expected.resolve()


def test_thresholds_read_only_train_index_and_gauge_files(tmp_path):
    import pandas as pd

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    pd.DataFrame(
        {
            "pathECMWF": ["unused.npy"], "pathESP": ["unused.npy"],
            "leadTime": [7], "year": [2020], "month": [1], "day": [1],
        }
    ).to_csv(index_dir / "train.csv", index=False)
    gauge_path = tmp_path / "gauge.csv"
    pd.DataFrame(
        {
            "Day": pd.date_range("2020-01-02", periods=7),
            "Station": ["A"] * 7,
            "R": [1, 2, 3, 4, 5, 6, 7],
        }
    ).to_csv(gauge_path, index=False)
    config = SimpleNamespace(
        DATA=SimpleNamespace(
            DATA_IDX_DIR=str(index_dir), GAUGE_DATA_PATH=str(gauge_path),
            RAIN_THRESHOLD=300,
        ),
        MODEL=SimpleNamespace(ECMWF_TIME_STEP=7),
    )
    output_path = tmp_path / "thresholds.json"
    values = evaluation.compute_training_percentiles_from_files(config, output_path)
    assert values == {"p90": 28.0, "p95": 28.0, "p99": 28.0}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_split"] == "train"
    assert payload["n_observations"] == 1
