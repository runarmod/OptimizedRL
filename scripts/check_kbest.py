"""Check KBestSolver against enumerating every DAP assortment.

For random states and parameters, the K best assortments found by the solver
must have the same objective values as the K best of all C(n_items, K)
assortments, scored with DapModel.assortment_value, and DapModel's completed
candidates must keep the solver's assortments and values.

    uv run python scripts/check_kbest.py
"""

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gym_envs.dap_env import N_ITEM_FEATURES  # noqa: E402
from src.models.dap_model import N_FEATURES, DapModel  # noqa: E402
from src.solvers.kbest import KBestSolver  # noqa: E402

N_ITEMS = 20
ASSORTMENT_SIZE = 4
N_PIECES = 2
K = 8
N_TRIALS = 5
TOL = 1e-6


def enumerated_values(model: DapModel) -> np.ndarray:
    values = []
    for items in combinations(range(N_ITEMS), ASSORTMENT_SIZE):
        x = np.zeros(N_ITEMS)
        x[list(items)] = 1.0
        values.append(model.assortment_value(x))
    return np.sort(values)


def check(pairwise: bool, n_pieces: int, rng: np.random.Generator) -> None:
    solver = KBestSolver(k=K)
    times = []
    for _ in range(N_TRIALS):
        theta = rng.normal(size=2 * N_FEATURES + n_pieces * (2 * N_FEATURES + 1))
        model = DapModel(theta, N_ITEMS, ASSORTMENT_SIZE, n_pieces, pairwise=pairwise)
        state = rng.uniform(0, 1, size=N_ITEMS * N_ITEM_FEATURES)
        model._update_state(state)
        model.phi, model.phi_bar, model.phi_pairs = model._features(state)

        node = model.get_LP_formulation()
        start = time.perf_counter()
        pool = solver.solve(node)
        times.append(time.perf_counter() - start)

        assert len(pool) == K, f"expected {K} candidates, got {len(pool)}"
        xs = {tuple(sol["x"][:N_ITEMS].round().astype(int)) for sol in pool}
        assert len(xs) == K, "duplicate assortments in the pool"
        for sol in pool:
            assert sol["x"][:N_ITEMS].sum() == ASSORTMENT_SIZE

        solver_vals = np.array([sol["fun"] for sol in pool])
        exact_vals = enumerated_values(model)[:K]
        assert np.allclose(np.sort(solver_vals), exact_vals, atol=TOL), (
            solver_vals,
            exact_vals,
        )

        completed = model.complete_candidates(pool)
        for sol, comp in zip(pool, completed):
            assert np.array_equal(sol["x"][:N_ITEMS], comp["x"][:N_ITEMS])
            assert abs(sol["fun"] - comp["fun"]) < TOL

        best = solver.solve_best(node, model)
        assert abs(model.assortment_value(best) - exact_vals[0]) < TOL

    print(
        f"pairwise={pairwise!s:5} pieces={n_pieces}: ok, "
        f"{1000 * np.mean(times):.0f} ms per pool of {K}"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    for pairwise in (True, False):
        for n_pieces in (N_PIECES, 0):
            check(pairwise, n_pieces, rng)


if __name__ == "__main__":
    main()
