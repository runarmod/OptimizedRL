import os
from pathlib import Path

import numpy as np
import yaml

from src.config.config_models import AppConfig
from src.models.model_interface import Model


def save_model_params(
    path: Path, model: Model, config: AppConfig, completed_iters: int
) -> None:
    """Save the trained model parameters together with the config that produced them.

    Each parameter array from ``model.get_params()`` is stored under the name of
    the model attribute holding it (aA/aB/b/c, or e.g. theta for vector models).
    The file is written to a temporary path first and then renamed, so an
    interrupt during the write never corrupts the previously saved file.
    """
    data = {
        name: np.asarray(values).tolist()
        for name, values in model.get_params().arrays().items()
    }
    data["completed_iters"] = completed_iters
    data["config"] = config.model_dump(mode="json")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as params_file:
        yaml.safe_dump(data, params_file, sort_keys=False)
    os.replace(tmp_path, path)


def load_model_params(path: Path, model: Model) -> None:
    """Load parameters saved by ``save_model_params`` into ``model``."""
    with path.open() as params_file:
        saved = yaml.safe_load(params_file)

    for name, current in model.get_params().arrays().items():
        if name not in saved:
            raise KeyError(f"{path} has no parameter '{name}'.")
        values = np.asarray(saved[name], dtype=float)
        if values.shape != np.shape(current):
            raise ValueError(
                f"Parameter '{name}' in {path} has shape {values.shape}, "
                f"but the model expects {np.shape(current)}."
            )
        setattr(model, name, values)
