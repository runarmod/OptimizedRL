from pathlib import Path

import yaml

from src.config.config_models import AppConfig


def load_config(path: Path) -> AppConfig:
    """Load YAML and return a validated configuration dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise TypeError(
            f"Config file '{path}' must contain a mapping at the top level, "
            f"got {type(raw_config).__name__}."
        )

    return AppConfig.model_validate(raw_config)
