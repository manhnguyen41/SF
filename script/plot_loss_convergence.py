#!/usr/bin/env python3
"""Plot Full and No-pretrain train/validation loss curves on one axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_wandb_loss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    value_columns = [
        column
        for column in frame.columns
        if column != "Step" and not column.endswith(("__MIN", "__MAX"))
    ]
    if len(value_columns) != 1:
        raise ValueError(
            f"Expected exactly one loss column in {path}, found {value_columns}"
        )

    result = frame[["Step", value_columns[0]]].copy()
    result.columns = ["epoch", "loss"]
    result["epoch"] = pd.to_numeric(result["epoch"], errors="raise").astype(int)
    result["loss"] = pd.to_numeric(result["loss"], errors="raise")
    return result.sort_values("epoch").reset_index(drop=True)


def first_within_one_percent(valid: pd.DataFrame) -> int:
    best = float(valid["loss"].min())
    candidates = valid.loc[valid["loss"] <= best * 1.01, "epoch"]
    return int(candidates.iloc[0])


def summarize(name: str, train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, float | int | str]:
    best_index = valid["loss"].idxmin()
    best_loss = float(valid.loc[best_index, "loss"])
    best_epoch = int(valid.loc[best_index, "epoch"])
    initial_valid = float(valid.iloc[0]["loss"])
    return {
        "model": name,
        "epochs_observed": int(max(train["epoch"].max(), valid["epoch"].max()) + 1),
        "initial_train_loss": float(train.iloc[0]["loss"]),
        "final_train_loss": float(train.iloc[-1]["loss"]),
        "initial_valid_loss": initial_valid,
        "final_valid_loss": float(valid.iloc[-1]["loss"]),
        "best_valid_loss": best_loss,
        "best_valid_epoch": best_epoch,
        "first_epoch_within_1pct_of_best": first_within_one_percent(valid),
        "valid_improvement_abs": initial_valid - best_loss,
        "valid_improvement_pct": (initial_valid - best_loss) / initial_valid * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-dir", type=Path, default=Path("loss"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("loss/loss_convergence_full_vs_no_pretrain"),
    )
    args = parser.parse_args()

    curves = {
        "Full": {
            "train": read_wandb_loss(args.loss_dir / "train.csv"),
            "valid": read_wandb_loss(args.loss_dir / "valid.csv"),
            "color": "#0072B2",
        },
        "No-pretrain": {
            "train": read_wandb_loss(args.loss_dir / "train_no-pretrain.csv"),
            "valid": read_wandb_loss(args.loss_dir / "valid_no-pretrain.csv"),
            "color": "#D55E00",
        },
    }

    summaries = [
        summarize(name, values["train"], values["valid"])
        for name, values in curves.items()
    ]
    summary = pd.DataFrame(summaries)

    common_last_epoch = min(
        int(values[kind]["epoch"].max())
        for values in curves.values()
        for kind in ("train", "valid")
    )
    for values in curves.values():
        for kind in ("train", "valid"):
            row = values[kind].loc[values[kind]["epoch"] == common_last_epoch]
            summary.loc[
                summary["model"] == next(
                    name for name, candidate in curves.items() if candidate is values
                ),
                f"{kind}_loss_at_common_epoch_{common_last_epoch}",
            ] = float(row.iloc[0]["loss"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7.2), constrained_layout=True)

    for name, values in curves.items():
        color = values["color"]
        train = values["train"]
        valid = values["valid"]

        ax.plot(
            train["epoch"],
            train["loss"],
            color=color,
            linewidth=2.2,
            label=f"{name} — train",
        )
        ax.plot(
            valid["epoch"],
            valid["loss"],
            color=color,
            linewidth=2.2,
            linestyle="--",
            marker="o",
            markersize=3.5,
            markevery=2,
            label=f"{name} — validation",
        )

        best_row = valid.loc[valid["loss"].idxmin()]
        best_epoch = int(best_row["epoch"])
        best_loss = float(best_row["loss"])
        ax.scatter(
            [best_epoch],
            [best_loss],
            s=75,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"best {best_loss:.6f}\nepoch {best_epoch}",
            xy=(best_epoch, best_loss),
            xytext=(8, 13 if name == "Full" else -34),
            textcoords="offset points",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
        )

    full_best = float(summary.loc[summary["model"] == "Full", "best_valid_loss"].iloc[0])
    no_pretrain_best = float(
        summary.loc[summary["model"] == "No-pretrain", "best_valid_loss"].iloc[0]
    )
    delta = no_pretrain_best - full_best
    ax.text(
        0.985,
        0.975,
        "Best validation loss\n"
        f"Full: {full_best:.6f}\n"
        f"No-pretrain: {no_pretrain_best:.6f}\n"
        f"Δ (No-pretrain − Full): {delta:+.6f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#F8FAFC", "ec": "#94A3B8"},
    )

    all_losses = np.concatenate(
        [
            values[kind]["loss"].to_numpy()
            for values in curves.values()
            for kind in ("train", "valid")
        ]
    )
    margin = (all_losses.max() - all_losses.min()) * 0.08
    ax.set_ylim(all_losses.min() - margin, all_losses.max() + margin)
    ax.set_xlim(left=0)
    ax.set_title("Training convergence: Full vs No-pretrain", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Weighted MSE loss (normalized target)", fontsize=11)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, ncol=2)
    ax.grid(True, color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.005,
        "Raw epoch losses; no smoothing. Labels follow filenames. All four CSV headers currently identify Strans-V5.",
        fontsize=8.5,
        color="#475569",
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), index=False)

    print(summary.to_string(index=False))
    print(f"common_last_epoch={common_last_epoch}")
    print(f"png={args.output_prefix.with_suffix('.png')}")
    print(f"pdf={args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
