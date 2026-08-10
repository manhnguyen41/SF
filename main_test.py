"""Training entry point with reproducible experiment result saving.

Environment variables added by this revision:

VIFOS_EXPERIMENT_NAME
    Stable name shared by every seed of one configuration, for example
    ``full_vifos`` or ``without_gsmap``.
VIFOS_RESULTS_DIR
    Root output directory. Defaults to ``experiment_results``.
VIFOS_COMPUTE_THRESHOLDS_ONLY=1
    Read the training set, save P90/P95/P99 rainfall thresholds, and exit.
VIFOS_THRESHOLDS_FILE
    Optional path to the train-derived threshold JSON file.
VIFOS_EXTREME_THRESHOLDS
    Optional absolute thresholds, e.g. ``50,100,150`` or ``heavy=100``.
VIFOS_TEST_ONLY=1
    Skip training and evaluate an existing checkpoint for the current session.
VIFOS_COMPARE_WITH
    Optional experiment name for paired seed tests after the run completes.
"""

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from src.model import model_2head, models
from src.model.baseline import cnn_lstm, conv_lstm
from src.model.baseline.unet import UNet
from src.utils import get_scaler, test_func, train_func, utils
from src.utils.dataloader import CustomDataset3
from src.utils.get_option import get_option
from src.utils.get_session_name import get_session_name
from src.utils.loss import (
    CombinedWeightedLoss,
    ExpMagnitudeWeightedMAELoss,
    LogMagnitudeWeightedHuberLoss,
    MSLELoss,
    MagnitudeWeightedHuberLoss,
    QuantileLoss,
    WeightedMSELoss,
    WeightedThresholdMSE,
)


def get_device():
    if torch.cuda.is_available():
        print("Device: GPU")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def get_loss_function(config):
    loss_name = config.LOSS.NAME.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss(reduction="mean")
    if loss_name == "huberloss":
        return nn.HuberLoss(delta=1.0, reduction="mean")
    if loss_name == "expweightedloss":
        return ExpMagnitudeWeightedMAELoss(config.LOSS.k)
    if loss_name == "weightedmse":
        return WeightedMSELoss(weight_func=config.LOSS.WEIGHT_FUNC)
    if loss_name == "magnitudeweight":
        return MagnitudeWeightedHuberLoss(delta=config.LOSS.DELTA)
    if loss_name == "msle":
        return MSLELoss()
    if loss_name == "logmagnitudeweight":
        return LogMagnitudeWeightedHuberLoss(
            delta=config.LOSS.DELTA,
            alpha=config.LOSS.ALPHA,
        )
    if loss_name == "quantile":
        return QuantileLoss(quantile=0.7, alpha=0.4)
    if loss_name == "weightedthresholdmse":
        return WeightedThresholdMSE(
            high_weight=config.LOSS.HIGH_WEIGHT,
            low_weight=config.LOSS.LOW_WEIGHT,
            threshold=config.LOSS.GROUNDTRUTH_THRESHOLD,
        )
    if loss_name == "combineweightloss":
        return CombinedWeightedLoss(gamma=2.0, beta=0.01)
    raise ValueError(f"Invalid loss function name: {config.LOSS.NAME}")


def get_model(config, device):
    name = config.MODEL.NAME.lower()
    model_map = {
        "model_v1": models.Model_Ver1,
        "model_v2": models.Model_Ver2,
        "strans": models.SwinTransformer,
        "strans-v2": models.SwinTransformer_Ver2,
        "strans-v3": models.SwinTransformer_Ver3,
        "strans-v4": models.SwinTransformer_Ver4,
        "strans-v4b": models.SwinTransformer_Ver4b,
        "strans-v5": models.SwinTransformer_Ver5,
        "strans-v6": models.SwinTransformer_Ver6,
        "cnn-lstm": cnn_lstm.CNN,
        "cnn-lstm-se": cnn_lstm.CNN_LSTM_SE,
        "conv-lstm": conv_lstm.ConvLSTMModel,
        "unet": UNet,
        "vit-2head": model_2head.VIT_2Head,
    }
    if name not in model_map:
        raise ValueError(f"Wrong model name: {config.MODEL.NAME}")
    return model_map[name](config).to(device)


def create_checkpoint_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def init_wandb(config):
    if not config.WANDB.STATUS:
        return
    # Authentication is read from WANDB_API_KEY or the user's existing W&B
    # login.  Do not commit an API key in source code.
    wandb.init(
        entity="aiotlab",
        project="SubSeasonalForecasting",
        group=config.WANDB.GROUP_NAME,
        name=config.WANDB.SESSION_NAME,
        config=config,
    )


def _slug(value):
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return value.strip("._-") or "unnamed"


def _is_enabled(environment_name):
    return os.getenv(environment_name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _experiment_layout(config):
    result_root = Path(os.getenv("VIFOS_RESULTS_DIR", "experiment_results"))
    group_name = _slug(config.WANDB.GROUP_NAME)
    learning_rate = str(config.OPTIMIZER.LR).replace(".", "p")
    fallback_name = f"{config.MODEL.NAME}_lr_{learning_rate}"
    experiment_name = _slug(os.getenv("VIFOS_EXPERIMENT_NAME", fallback_name))
    seed = int(config.MODEL.SEED)
    group_dir = result_root / group_name
    experiment_dir = group_dir / experiment_name
    result_dir = experiment_dir / f"seed_{seed}"
    return result_root, group_dir, experiment_dir, result_dir, experiment_name


def _parse_manual_thresholds(raw_value):
    thresholds = {}
    if not raw_value:
        return thresholds
    for index, token in enumerate(raw_value.split(","), start=1):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            label, value = token.split("=", 1)
        else:
            label, value = f"absolute_{index}", token
        thresholds[_slug(label)] = float(value)
    return thresholds


def _save_config_snapshot(config, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        content = config.dump()
    except (AttributeError, TypeError):
        content = str(config)
    with (result_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        stream.write(content)
        if not content.endswith("\n"):
            stream.write("\n")


def _numeric_train_results(results):
    if not isinstance(results, dict):
        return {}
    output = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float, bool, str)) or value is None:
            output[str(key)] = value
    return output


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    _args, config = get_option()
    config.WANDB.SESSION_NAME = get_session_name(config)
    device = get_device()
    config.DEVICE = device
    seed = int(config.MODEL.SEED)
    utils.seed_everything(seed)

    (
        result_root,
        group_dir,
        experiment_dir,
        result_dir,
        experiment_name,
    ) = _experiment_layout(config)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("*************** Get scaler ***************")
    input_scaler, esp_scaler, output_scaler = get_scaler.get_scaler(config)

    print("*************** Init dataset ***************")
    train_dataset = CustomDataset3(
        mode="train",
        config=config,
        ecmwf_scaler=input_scaler,
        esp_scaler=esp_scaler,
        output_scaler=output_scaler,
        shuffle=True,
    )
    valid_dataset = CustomDataset3(
        mode="valid",
        config=config,
        ecmwf_scaler=input_scaler,
        esp_scaler=esp_scaler,
        output_scaler=output_scaler,
    )
    test_dataset = CustomDataset3(
        mode="test",
        config=config,
        ecmwf_scaler=input_scaler,
        esp_scaler=esp_scaler,
        output_scaler=output_scaler,
    )

    threshold_file = Path(
        os.getenv(
            "VIFOS_THRESHOLDS_FILE",
            str(result_root / "thresholds" / "train_rainfall_percentiles.json"),
        )
    )
    if _is_enabled("VIFOS_COMPUTE_THRESHOLDS_ONLY"):
        thresholds = test_func.compute_training_percentiles(
            train_dataset=train_dataset,
            config=config,
            output_scaler=output_scaler,
            output_path=threshold_file,
        )
        print(f"Training rainfall percentiles: {thresholds}")
        print(f"Saved thresholds to: {threshold_file}")
        return

    manual_thresholds = _parse_manual_thresholds(
        os.getenv("VIFOS_EXTREME_THRESHOLDS", "")
    )
    extreme_thresholds = test_func.load_extreme_thresholds(
        path=threshold_file,
        manual_thresholds=manual_thresholds,
    )
    if not extreme_thresholds:
        print(
            "WARNING: Heavy-rainfall metrics will be skipped because no thresholds "
            f"were found at {threshold_file}. Run once with "
            "VIFOS_COMPUTE_THRESHOLDS_ONLY=1."
        )

    loss_func = get_loss_function(config)
    base_metadata = {
        "experiment_name": experiment_name,
        "group_name": str(config.WANDB.GROUP_NAME),
        "session_name": str(config.WANDB.SESSION_NAME),
        "model_name": str(config.MODEL.NAME),
        "seed": seed,
        "learning_rate": float(config.OPTIMIZER.LR),
        "optimizer": str(config.OPTIMIZER.NAME),
        "loss": str(config.LOSS.NAME),
        "device": str(device),
        "threshold_file": str(threshold_file),
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_config_snapshot(config, result_dir)

    model_name = config.MODEL.NAME.lower()
    if model_name in {"quantitle", "quantile"}:
        init_wandb(config)
        saved = test_func.test_func_quantile(
            test_dataset,
            loss_func,
            config,
            esp_scaler,
            output_scaler,
            device,
            result_dir=result_dir,
            run_metadata=base_metadata,
            extreme_thresholds=extreme_thresholds,
        )
        test_func.aggregate_seed_results(experiment_dir)
        if config.WANDB.STATUS and wandb.run:
            wandb.finish()
        print(f"Completed quantile-mapping run: {saved['result_dir']}")
        return

    checkpoint_dir = Path("saved_checkpoints") / config.WANDB.GROUP_NAME / "checkpoint"
    create_checkpoint_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{config.WANDB.SESSION_NAME}.pt"
    early_stopping = utils.EarlyStopping(
        patience=config.EARLY_STOPPING.PATIANCE,
        verbose=True,
        delta=config.EARLY_STOPPING.DELTA,
        path=str(checkpoint_path),
    )

    print("*************** Init model ***************")
    model = get_model(config, device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params_count}")

    trainable_params = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if config.OPTIMIZER.NAME.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.OPTIMIZER.LR,
            weight_decay=config.OPTIMIZER.L2_COEF,
        )
    elif config.OPTIMIZER.NAME.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.OPTIMIZER.LR,
            weight_decay=config.OPTIMIZER.L2_COEF,
        )
    else:
        raise ValueError(f"Wrong optimizer name: {config.OPTIMIZER.NAME}")

    init_wandb(config)
    test_only = _is_enabled("VIFOS_TEST_ONLY")
    if test_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"VIFOS_TEST_ONLY=1 but checkpoint was not found: {checkpoint_path}"
            )
        results = {"test_only": True}
        training_seconds = 0.0
        training_peak_gpu_memory_mb = 0.0
        print(f"Skipping training; using checkpoint: {checkpoint_path}")
    else:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        training_start = time.perf_counter()
        results = train_func.train_func(
            model,
            train_dataset,
            valid_dataset,
            early_stopping,
            loss_func,
            optimizer,
            config,
            device,
        )
        _synchronize(device)
        training_seconds = time.perf_counter() - training_start
        training_peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
    if isinstance(results, dict) and "final_train_loss" in results:
        print(f"Final Train Loss: {results['final_train_loss']:.4f}")

    utils.load_model(model, str(checkpoint_path))
    numeric_results = _numeric_train_results(results)
    run_metadata = {
        **base_metadata,
        "checkpoint_path": str(checkpoint_path),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params_count,
        "training_seconds": training_seconds,
        "training_peak_gpu_memory_mb": training_peak_gpu_memory_mb,
        "test_only": test_only,
        "train_results": numeric_results,
    }
    for epoch_key in ("epochs_trained", "num_epochs", "n_epochs"):
        epochs = numeric_results.get(epoch_key)
        if isinstance(epochs, (int, float)) and epochs > 0:
            run_metadata["training_seconds_per_epoch"] = training_seconds / epochs
            run_metadata["training_epoch_count_source"] = epoch_key
            break
    saved = test_func.test_func(
        model,
        test_dataset,
        loss_func,
        config,
        esp_scaler,
        output_scaler,
        device,
        result_dir=result_dir,
        run_metadata=run_metadata,
        extreme_thresholds=extreme_thresholds,
        prediction_name="model",
    )

    aggregate_path = test_func.aggregate_seed_results(experiment_dir)
    if aggregate_path:
        print(f"Updated seed aggregate: {aggregate_path}")

    compare_with = os.getenv("VIFOS_COMPARE_WITH", "").strip()
    if compare_with:
        comparison_path = test_func.compare_experiments(
            group_dir,
            experiment_name,
            _slug(compare_with),
        )
        if comparison_path:
            print(f"Updated paired comparison: {comparison_path}")
        else:
            print(
                "Paired comparison was not created yet. Both experiments need "
                "at least two matching seeds."
            )

    if config.WANDB.STATUS and wandb.run:
        wandb.finish()
    print(f"Completed experiment run: {saved['result_dir']}")


if __name__ == "__main__":
    main()
