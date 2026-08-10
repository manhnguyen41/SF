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
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None

from src.model import model_2head, models
from src.model.baseline import cnn_lstm, conv_lstm
from src.model.baseline.unet import UNet
from src.utils import get_scaler, test_func_only as test_func, train_func, utils
from src.utils.dataloader import CustomDataset3
from src.utils.evaluation_io import require_checkpoint, resolve_checkpoint_path
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
    if wandb is None:
        raise RuntimeError("W&B is enabled but the 'wandb' package is not installed.")
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


def _sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint_path(config):
    """Backward-compatible local alias used by existing callers/tests."""
    return resolve_checkpoint_path(config)


def _require_checkpoint(config, context="VIFOS_TEST_ONLY=1"):
    return require_checkpoint(config, context=context)


def _git_metadata():
    command_prefix = ["git", "-c", "safe.directory=C:/Study/Lab/SF"]
    try:
        commit = subprocess.run(
            command_prefix + ["rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            command_prefix + ["status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"git_commit": commit, "git_dirty": bool(status), "git_status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"git_commit": None, "git_dirty": None, "git_error": str(error)}


def _namespace_to_dict(value):
    if hasattr(value, "items"):
        return {str(key): _namespace_to_dict(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): _namespace_to_dict(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_namespace_to_dict(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


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
        import yaml

        content = yaml.safe_dump(
            _namespace_to_dict(config), sort_keys=False, allow_unicode=True
        )
    except (ImportError, TypeError):
        content = json.dumps(_namespace_to_dict(config), indent=2, default=str)
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
    trained_batch_size = int(config.TRAIN.BATCH_SIZE)
    config.WANDB.SESSION_NAME = get_session_name(config)
    evaluation_batch_size = int(
        os.getenv("VIFOS_EVAL_BATCH_SIZE", str(trained_batch_size))
    )
    if evaluation_batch_size <= 0:
        raise ValueError(
            f"VIFOS_EVAL_BATCH_SIZE must be positive, got {evaluation_batch_size}."
        )
    # The checkpoint/session name above must retain the training batch size.
    # Only the evaluation DataLoaders use this lower-memory override.
    config.TRAIN.BATCH_SIZE = evaluation_batch_size
    test_only = _is_enabled("VIFOS_TEST_ONLY")
    thresholds_only = _is_enabled("VIFOS_COMPUTE_THRESHOLDS_ONLY")
    if (test_only or thresholds_only) and not _is_enabled("VIFOS_ENABLE_WANDB"):
        config.WANDB.STATUS = False
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
    if _is_enabled("VIFOS_DRY_RUN"):
        checkpoint_path, candidates = _resolve_checkpoint_path(config)
        print(
            "DRY_RUN test-only: "
            f"experiment={experiment_name} model={config.MODEL.NAME} seed={seed} "
            f"data_idx_dir={config.DATA.DATA_IDX_DIR} checkpoint={checkpoint_path}"
        )
        print("Checkpoint candidates: " + " | ".join(str(path) for path in candidates))
        return
    if _is_enabled("VIFOS_PREFLIGHT_ONLY"):
        checkpoint_path, candidates = _require_checkpoint(
            config, context="Preflight requested"
        )
        print(f"PREFLIGHT_OK seed={seed} checkpoint={checkpoint_path}")
        return
    completion_marker = result_dir / "run_metadata.json"
    if test_only and completion_marker.exists() and not _is_enabled("VIFOS_FORCE"):
        if _is_enabled("VIFOS_SKIP_EXISTING_RESULTS"):
            print(f"SKIP existing completed result: {completion_marker}")
            return
        raise FileExistsError(
            f"Completed output already exists: {completion_marker}. "
            "Set VIFOS_SKIP_EXISTING_RESULTS=1 to skip it or VIFOS_FORCE=1 "
            "to overwrite it explicitly."
        )

    threshold_file = Path(
        os.getenv(
            "VIFOS_THRESHOLDS_FILE",
            str(result_root / "thresholds" / "train_rainfall_percentiles.json"),
        )
    )
    if thresholds_only:
        thresholds = test_func.compute_training_percentiles_from_files(
            config=config, output_path=threshold_file
        )
        print(f"Training rainfall percentiles: {thresholds}")
        print(f"Saved thresholds to: {threshold_file}")
        return

    print("*************** Get scaler ***************")
    input_scaler, esp_scaler, output_scaler = get_scaler.get_scaler(config)

    print("*************** Init dataset ***************")
    train_dataset = valid_dataset = test_dataset = None
    if thresholds_only or not test_only:
        train_dataset = CustomDataset3(
            mode="train", config=config, ecmwf_scaler=input_scaler,
            esp_scaler=esp_scaler, output_scaler=output_scaler, shuffle=True,
        )
    if not thresholds_only and not test_only:
        valid_dataset = CustomDataset3(
            mode="valid", config=config, ecmwf_scaler=input_scaler,
            esp_scaler=esp_scaler, output_scaler=output_scaler,
        )
    if not thresholds_only:
        test_dataset = CustomDataset3(
            mode="test", config=config, ecmwf_scaler=input_scaler,
            esp_scaler=esp_scaler, output_scaler=output_scaler,
        )

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
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_batch_size": trained_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "data_index_dir": str(config.DATA.DATA_IDX_DIR),
        "test_index_path": str(Path(config.DATA.DATA_IDX_DIR) / "test.csv"),
        "test_index_sha256": (
            _sha256(Path(config.DATA.DATA_IDX_DIR) / "test.csv")
            if (Path(config.DATA.DATA_IDX_DIR) / "test.csv").is_file()
            else None
        ),
        "data_split_policy": "identical_train_valid_test_csv_across_seeds_52_62_72_82_92",
        "threshold_file": str(threshold_file),
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        **_git_metadata(),
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
        if config.WANDB.STATUS and wandb is not None and wandb.run:
            wandb.finish()
        print(f"Completed quantile-mapping run: {saved['result_dir']}")
        return

    checkpoint_path, checkpoint_candidates = _resolve_checkpoint_path(config)
    checkpoint_dir = checkpoint_path.parent
    if not test_only:
        create_checkpoint_dir(checkpoint_dir)
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
    if test_only:
        checkpoint_path, checkpoint_candidates = _require_checkpoint(config)
        results = {"test_only": True}
        training_seconds = None
        training_peak_gpu_memory_mb = None
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
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params_count,
        "training_seconds": training_seconds,
        "training_peak_gpu_memory_mb": training_peak_gpu_memory_mb,
        "training_timing_note": (
            "not_available_from_test_only" if test_only else "measured_in_current_training_run"
        ),
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
            _slug(compare_with),
            experiment_name,
        )
        if comparison_path:
            print(f"Updated paired comparison: {comparison_path}")
        else:
            print(
                "Paired comparison was not created yet. Both experiments need "
                "at least two matching seeds."
            )

    if config.WANDB.STATUS and wandb is not None and wandb.run:
        wandb.finish()
    print(f"Completed experiment run: {saved['result_dir']}")


if __name__ == "__main__":
    main()
