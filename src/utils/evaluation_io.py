"""Lightweight checkpoint I/O helpers for test-only evaluation."""

import os
from pathlib import Path


def resolve_checkpoint_path(config):
    seed = int(config.MODEL.SEED)
    override = os.getenv("VIFOS_CHECKPOINT_PATH", "").strip()
    candidates = []
    if override:
        expanded = Path(
            override.format(seed=seed, session=config.WANDB.SESSION_NAME)
        )
        candidates.append(
            expanded / f"{config.WANDB.SESSION_NAME}.pt"
            if expanded.is_dir()
            else expanded
        )
    checkpoint_root = Path("saved_checkpoints") / config.WANDB.GROUP_NAME
    candidates.extend(
        (
            checkpoint_root / "checkpoint" / f"{config.WANDB.SESSION_NAME}.pt",
            checkpoint_root / f"{config.WANDB.SESSION_NAME}.pt",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), candidates
    return candidates[0].resolve(), candidates


def require_checkpoint(config, context="VIFOS_TEST_ONLY=1"):
    checkpoint_path, candidates = resolve_checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{context} but no checkpoint was found. Paths tried:\n- "
            + "\n- ".join(str(path.resolve()) for path in candidates)
        )
    return checkpoint_path, candidates
