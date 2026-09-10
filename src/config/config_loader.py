from pathlib import Path

import yaml

from src.config.config_models import AppConfig


def load_config(path: Path) -> AppConfig:
    """Load YAML and return a validated configuration dictionary."""
    with path.open() as config_file:
        raw_config = yaml.safe_load(config_file)

    return AppConfig.model_validate(raw_config)
