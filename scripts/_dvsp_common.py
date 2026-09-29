"""Shared helpers for the DVSP scripts. Run scripts from the project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config_loader import load_config  # noqa: E402
from src.config.config_models import AppConfig  # noqa: E402
from src.dvsp.evaluation import load_splits, resolve_split_files  # noqa: E402
from src.dvsp.instance import VSPInstance  # noqa: E402


def load_dvsp_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    config = load_config(path)
    if config.dvsp is None:
        raise ValueError(f"{path} has no 'dvsp' section")
    return config


def load_dvsp_splits(
    config: AppConfig, splits=("train", "val", "test")
) -> dict[str, list[VSPInstance]]:
    cfg = config.dvsp
    instance_dir = cfg.instance_dir
    if not instance_dir.is_absolute():
        instance_dir = PROJECT_ROOT / instance_dir
    files = resolve_split_files(
        instance_dir=instance_dir,
        split_seed=cfg.split_seed,
        split_fractions=cfg.split_fractions,
        counts={
            "train": cfg.nb_train_instances,
            "val": cfg.nb_val_instances,
            "test": cfg.nb_test_instances,
        },
        explicit={
            "train": cfg.train_instances,
            "val": cfg.val_instances,
            "test": cfg.test_instances,
        },
    )
    return load_splits({s: files[s] for s in splits})
