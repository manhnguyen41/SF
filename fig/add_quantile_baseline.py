"""Rebuild separate paper panels with six lead-time methods and four time series.

Learned-model lead-time curves are five-seed means from the current aggregate
artifacts. S2S and Quantile Mapping are deterministic baselines on the same
test set. Time-series panels use aligned seed-52 artifacts for station S11 at
lead day 7.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "fig"
GROUP_DIR = ROOT / "experiment_results" / "data3-r1-test-vit-tiny-all-weekly"
FULL_PREDICTIONS = GROUP_DIR / "full_vifos" / "seed_52" / "predictions_long.csv"
CNN_PREDICTIONS = GROUP_DIR / "cnn_lstm" / "seed_52" / "predictions_long.csv"
UNET_PREDICTIONS = GROUP_DIR / "unet" / "seed_52" / "predictions_long.csv"
S2S_ONLY_PREDICTIONS = GROUP_DIR / "without_gsmap" / "seed_52" / "predictions_long.csv"
STATION_ID = "S11"
LEAD_TIME = 7


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_distributions():
    default_root = ROOT / "quantile_distributions" / "train_only"
    grid_dir = Path(os.getenv("VIFOS_QUANTILE_GRID_DIR", str(default_root / "s2s")))
    gauge_dir = Path(os.getenv("VIFOS_QUANTILE_GAUGE_DIR", str(default_root / "gauge")))
    grid = {path.stem: np.load(path) for path in grid_dir.glob("*.npy")}
    gauge = {}
    for path in gauge_dir.glob("*.npz"):
        data = np.load(path)
        gauge[path.stem] = (data["quantiles"], data["levels"])
    if not grid or not gauge:
        raise FileNotFoundError(
            "Train-only Quantile Mapping distributions are missing. "
            f"Expected grid files in {grid_dir} and gauge files in {gauge_dir}."
        )
    return grid, gauge


def quantile_prediction(row, grid, gauge):
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    row_index = int(np.clip(np.floor((23.25 - latitude) / 0.125), 0, 24))
    column_index = int(np.clip(np.floor((longitude - 102.25) / 0.125), 0, 24))
    key = f"{row_index:02d}_{column_index:02d}"
    raw = np.float32(float(row["prediction_ecmwf_s2s"]))
    grid_quantiles = grid[key]
    gauge_quantiles, levels = gauge[key]
    probability = np.interp(np.clip(raw, grid_quantiles[0], grid_quantiles[-1]), grid_quantiles, levels)
    return float(np.float32(np.clip(np.interp(probability, levels, gauge_quantiles), 0, 300)))


def calculate_metrics(prediction, observation):
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    error = prediction - observation
    denominator = np.sum((observation - observation.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1 - np.sum(error**2) / denominator),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "COR": float(np.corrcoef(observation, prediction)[0, 1]),
    }


def read_aggregate_model(directory, forecast_source="model"):
    rows = read_csv(GROUP_DIR / directory / "aggregate_per_lead_time.csv")
    values = defaultdict(dict)
    metric_names = {"mae": "MAE", "r2_pooled": "R2", "rmse": "RMSE", "corr": "COR"}
    for row in rows:
        if row["forecast_source"] == forecast_source and row["metric"] in metric_names:
            values[int(float(row["lead_time"]))][metric_names[row["metric"]]] = float(row["mean"])
    return values


def build_lead_time_data(full_rows, grid, gauge):
    by_lead = defaultdict(lambda: {"observation": [], "quantile": []})
    for row in full_rows:
        lead = int(float(row["lead_time"]))
        by_lead[lead]["observation"].append(float(row["observation"]))
        by_lead[lead]["quantile"].append(quantile_prediction(row, grid, gauge))

    quantile_metrics = {
        lead: calculate_metrics(values["quantile"], values["observation"])
        for lead, values in by_lead.items()
    }
    sources = {
        "Our model": read_aggregate_model("full_vifos"),
        "S2S": read_aggregate_model("full_vifos", "ecmwf_s2s"),
        "CNN-LSTM": read_aggregate_model("cnn_lstm"),
        "U-Net": read_aggregate_model("unet"),
        "S2S-only Transformer": read_aggregate_model("without_gsmap"),
        "Quantile Mapping": quantile_metrics,
    }
    fieldnames = ["lead_time"] + [
        f"{model}|{metric}"
        for model in sources
        for metric in ("MAE", "R2", "RMSE", "COR")
    ]
    output_rows = []
    for lead in range(7, 47):
        row = {"lead_time": lead}
        for model, values in sources.items():
            for metric in ("MAE", "R2", "RMSE", "COR"):
                row[f"{model}|{metric}"] = f"{values[lead][metric]:.8f}"
        output_rows.append(row)

    output = FIG_DIR / "all_metrics_comparison_with_quantile.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def plot_lead_time_metrics(rows):
    lead = np.asarray([float(row["lead_time"]) for row in rows])
    styles = {
        "Our model": dict(label="Ours", color="#e62e2e", marker="o", linestyle="-"),
        "S2S": dict(label="S2S", color="#4d4d4d", marker="^", linestyle="--"),
        "CNN-LSTM": dict(label="CNN-LSTM", color="#2ca02c", marker="s", linestyle=":"),
        "U-Net": dict(label="U-Net", color="#ff7f00", marker="v", linestyle=(0, (3, 1, 1, 1))),
        "S2S-only Transformer": dict(label="S2S-only Transformer", color="#8c564b", marker="P", linestyle=(0, (5, 2))),
        "Quantile Mapping": dict(label="Quantile Mapping", color="#7b3294", marker="D", linestyle="-."),
    }
    panels = [
        ("R2", r"R$^2$ Score", r"(a) $R^2$ across different lead times."),
        ("COR", "Correlation Coefficient", "(b) Correlation across different lead times."),
        ("MAE", "MAE Error (mm/1w)", "(c) MAE across different lead times."),
        ("RMSE", "RMSE Error (mm/1w)", "(d) RMSE across different lead times."),
    ]
    handles = []
    for metric, ylabel, _caption in panels:
        fig, axis = plt.subplots(figsize=(6.4, 4.8))
        for model, style in styles.items():
            values = np.asarray([float(row[f"{model}|{metric}"]) for row in rows])
            line, = axis.plot(lead, values, linewidth=1.5, markersize=4, **style)
            if len(handles) < len(styles):
                handles.append(line)
        axis.set_xlabel("Lead Time (days)", fontweight="bold")
        axis.set_ylabel(ylabel, fontweight="bold")
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.minorticks_on()
        if metric == "R2":
            axis.axhline(0, color="#ff6b6b", linewidth=1, linestyle=":")
        fig.tight_layout()
        filename = {"R2": "r2", "COR": "corr", "MAE": "mae", "RMSE": "rmse"}[metric]
        for suffix in ("png", "pdf", "eps"):
            fig.savefig(FIG_DIR / f"{filename}_comparison.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    legend_figure = plt.figure(figsize=(12.5, 0.55))
    legend_figure.legend(
        handles=handles,
        labels=[line.get_label() for line in handles],
        loc="center",
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=11,
    )
    for suffix in ("png", "pdf", "eps"):
        legend_figure.savefig(FIG_DIR / f"legend.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(legend_figure)


def model_predictions_by_sample(rows):
    return {
        row["sample_id"]: float(row["prediction_model"])
        for row in rows
        if row["station_id"] == STATION_ID and int(float(row["lead_time"])) == LEAD_TIME
    }


def build_time_series(full_rows, cnn_rows, unet_rows, s2s_only_rows, grid, gauge):
    selected_full = [
        row for row in full_rows
        if row["station_id"] == STATION_ID and int(float(row["lead_time"])) == LEAD_TIME
    ]
    selected_full.sort(key=lambda row: int(row["sample_index"]))
    cnn_by_sample = model_predictions_by_sample(cnn_rows)
    unet_by_sample = model_predictions_by_sample(unet_rows)
    s2s_only_by_sample = model_predictions_by_sample(s2s_only_rows)
    series = []
    for index, row in enumerate(selected_full):
        series.append({
            "index": index,
            "sample_id": row["sample_id"],
            "groundtruth": float(row["observation"]),
            "S2S": float(row["prediction_ecmwf_s2s"]),
            "CNN-LSTM": cnn_by_sample[row["sample_id"]],
            "U-Net": unet_by_sample[row["sample_id"]],
            "S2S-only Transformer": s2s_only_by_sample[row["sample_id"]],
            "Ours": float(row["prediction_model"]),
            "Quantile Mapping": quantile_prediction(row, grid, gauge),
        })
    output = FIG_DIR / f"time_series_with_quantile_seed52_{STATION_ID}_lead{LEAD_TIME:02d}.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    return series


def plot_time_series(series):
    x = np.asarray([row["index"] for row in series])
    truth = np.asarray([row["groundtruth"] for row in series])
    panels = [
        ("S2S", "#4d4d4d", "(a) S2S"),
        ("CNN-LSTM", "#2ca02c", "(b) CNN-LSTM"),
        ("U-Net", "#ff7f00", "(c) U-Net"),
        ("S2S-only Transformer", "#8c564b", "(d) S2S-only Transformer"),
        ("Ours", "#e62e2e", "(e) Our model"),
        ("Quantile Mapping", "#7b3294", "(f) Quantile Mapping"),
    ]
    output_dir = FIG_DIR / "4-results" / "Station" / "S2S" / f"leadtime{LEAD_TIME}"
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "S2S": "s2s_comparison",
        "CNN-LSTM": "cnn-lstm_comparison",
        "U-Net": "unet_comparison",
        "S2S-only Transformer": "s2s-only_transformer_comparison",
        "Ours": "ours_comparison",
        "Quantile Mapping": "quantile_mapping_comparison",
    }
    for name, color, _caption in panels:
        fig, axis = plt.subplots(figsize=(5.2, 4.1))
        prediction = np.asarray([row[name] for row in series])
        axis.plot(x, truth, color="#3db7d6", linewidth=1.0, marker=".", markersize=2.5, label="Groundtruth")
        axis.plot(x, prediction, color=color, linewidth=1.0, marker=".", markersize=2.5, label=name)
        axis.set_xlim(0, len(series) - 1)
        axis.set_ylim(0, 310)
        axis.set_xticks(np.arange(0, len(series), 30))
        axis.set_xlabel("Index", fontweight="bold")
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=12,
                    frameon=True, fancybox=False, edgecolor="black", handlelength=1.2, handletextpad=0.35)
        axis.set_ylabel("Precipitation (mm/1w)", fontweight="bold")
        fig.tight_layout()
        for suffix in ("png", "pdf", "eps"):
            fig.savefig(output_dir / f"{filenames[name]}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    grid, gauge = load_distributions()
    full_rows = read_csv(FULL_PREDICTIONS)
    cnn_rows = read_csv(CNN_PREDICTIONS)
    unet_rows = read_csv(UNET_PREDICTIONS)
    s2s_only_rows = read_csv(S2S_ONLY_PREDICTIONS)
    lead_rows = build_lead_time_data(full_rows, grid, gauge)
    plot_lead_time_metrics(lead_rows)
    time_series = build_time_series(
        full_rows, cnn_rows, unet_rows, s2s_only_rows, grid, gauge
    )
    plot_time_series(time_series)


if __name__ == "__main__":
    main()
