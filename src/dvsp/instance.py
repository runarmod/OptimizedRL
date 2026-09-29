from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VSPInstance:
    """Static vehicle scheduling instance. Location 0 is the depot.

    Times are normalized (hours by default). A request must be started exactly
    at ``start_time``; vehicles may wait at no cost.
    """

    name: str
    coordinate: np.ndarray  # (n_locations, 2)
    service_time: np.ndarray  # (n_locations,)
    start_time: np.ndarray  # (n_locations,)
    duration: np.ndarray  # (n_locations, n_locations)

    @property
    def n_locations(self) -> int:
        return int(self.start_time.shape[0])

    @property
    def n_customers(self) -> int:
        return self.n_locations - 1


def read_vsp_instance(path: Path, normalization: float = 3600.0) -> VSPInstance:
    """Read a EURO-NeurIPS 2022 VRPTW file as a VSP instance.

    Mirrors ``read_vsp_instance`` in DynamicVehicleRouting.jl: each request's
    start time is the middle of its time window, demands and capacities are
    ignored, and all times are divided by ``normalization``.
    """
    path = Path(path)
    mode = ""
    n_locations = 0
    duration_rows = []
    coordinate = service_time = start_time = None

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("DIMENSION"):
                n_locations = int(line.split(":")[1])
                coordinate = np.zeros((n_locations, 2))
                service_time = np.zeros(n_locations)
                start_time = np.zeros(n_locations)
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                if line.split(":")[1].strip() != "EXPLICIT":
                    raise ValueError(f"{path}: only EXPLICIT edge weights supported")
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                if line.split(":")[1].strip() != "FULL_MATRIX":
                    raise ValueError(f"{path}: only FULL_MATRIX edge weights supported")
            elif line.startswith("NODE_COORD_SECTION"):
                mode = "coord"
            elif line == "DEMAND_SECTION":
                mode = "demand"
            elif line == "DEPOT_SECTION":
                mode = "depot"
            elif line == "EDGE_WEIGHT_SECTION":
                mode = "edge_weights"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "EOF":
                break
            elif mode == "coord":
                node, x, y = line.split()
                coordinate[int(node) - 1] = (float(x), float(y))
            elif mode == "edge_weights":
                duration_rows.append([float(e) for e in line.split()])
            elif mode == "service_t":
                node, t = line.split()
                service_time[int(node) - 1] = float(t)
            elif mode == "time_windows":
                node, lo, hi = line.split()
                start_time[int(node) - 1] = (float(lo) + float(hi)) / 2

    duration = np.asarray(duration_rows, dtype=float)
    if duration.shape != (n_locations, n_locations):
        raise ValueError(f"{path}: duration matrix has shape {duration.shape}")

    return VSPInstance(
        name=path.stem,
        coordinate=coordinate / normalization,
        service_time=service_time / normalization,
        start_time=start_time / normalization,
        duration=duration / normalization,
    )


def list_instance_files(instance_dir: Path) -> list[Path]:
    return sorted(p for p in Path(instance_dir).iterdir() if p.suffix == ".txt")


def split_instances(
    files: list[Path],
    seed: int,
    fractions: tuple[float, float] = (0.34, 0.34),
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle and split instance files into train/validation/test.

    Uses the same fractions as the Structured-RL setup, but numpy's RNG, so the
    concrete instances differ from paper 02's Julia split. Pass explicit
    instance names in the config to reproduce a specific split.
    """
    files = sorted(files)
    order = np.random.default_rng(seed).permutation(len(files))
    shuffled = [files[i] for i in order]
    n_train = int(round(fractions[0] * len(files)))
    n_val = int(round(fractions[1] * len(files)))
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    return train, val, test
