"""Lightweight checkpoint I/O helpers for test-only evaluation."""

import os
from pathlib import Path


def resolve_checkpoint_path(config):
    seed = int(config.MODEL.SEED)
    primary_session = str(config.WANDB.SESSION_NAME)
    session_names = [primary_session]

    # Legacy compatibility: strans-v4b was trained before get_session_name
    # had an explicit mapping, so those checkpoints were labelled Modelv1.
    # Some later evaluation/training code used Strans-V4b. Accept both exact
    # session names without changing any other hyperparameter component.
    if str(getattr(config.MODEL, "NAME", "")).lower() == "strans-v4b":
        for source, target in (
            ("_Modelv1_", "_Strans-V4b_"),
            ("_Strans-V4b_", "_Modelv1_"),
            ("_Strans-V4B_", "_Modelv1_"),
        ):
            if source in primary_session:
                alias = primary_session.replace(source, target, 1)
                if alias not in session_names:
                    session_names.append(alias)

    override = os.getenv("VIFOS_CHECKPOINT_PATH", "").strip()
    candidates = []
    if override:
        for session_name in session_names:
            expanded = Path(override.format(seed=seed, session=session_name))
            candidates.append(
                expanded / f"{session_name}.pt"
                if expanded.is_dir()
                else expanded
            )
    checkpoint_root = Path("saved_checkpoints") / config.WANDB.GROUP_NAME
    for session_name in session_names:
        candidates.extend(
            (
                checkpoint_root / "checkpoint" / f"{session_name}.pt",
                checkpoint_root / f"{session_name}.pt",
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
