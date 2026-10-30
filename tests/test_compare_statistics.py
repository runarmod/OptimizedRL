import numpy as np
import pytest

from compare_srl_paper import (
    CONFIDENCE,
    format_p,
    paired_difference,
    seed_level_tests,
)

N_SEEDS, N_EPISODES = 10, 100


def models(offset: float, seed: int, seed_spread: float = 5.0) -> np.ndarray:
    """Per-episode revenue (N_SEEDS, N_EPISODES) of a policy whose models differ
    by seed and whose episodes share instance difficulty with other policies."""
    rng = np.random.default_rng(seed)
    episode_difficulty = np.random.default_rng(0).normal(500, 50, N_EPISODES)
    model_effect = rng.normal(0, seed_spread, (N_SEEDS, 1))
    noise = rng.normal(0, 20, (N_SEEDS, N_EPISODES))
    return episode_difficulty + offset + model_effect + noise


def test_clear_difference_is_significant():
    a, b = models(30.0, seed=1), models(0.0, seed=2)
    mean, low, high, p = paired_difference(a, b, np.random.default_rng(0))
    assert mean == pytest.approx(a.mean() - b.mean())
    assert 0 < low < mean < high
    assert p < 0.01
    tests = seed_level_tests(a, b)
    assert set(tests) == {"Welch t-test", "Mann-Whitney U"}
    assert all(p < 0.01 for p in tests.values())


def test_no_difference_is_not_significant():
    a, b = models(0.0, seed=3), models(0.0, seed=4)
    _, low, high, p = paired_difference(a, b, np.random.default_rng(0))
    assert low < 0 < high
    assert p > 0.05


@pytest.mark.parametrize("offset", [0.0, 3.0, 6.0, 10.0])
def test_bootstrap_p_value_agrees_with_the_interval(offset):
    a, b = models(offset, seed=5), models(0.0, seed=6)
    _, low, high, p = paired_difference(a, b, np.random.default_rng(0))
    excludes_zero = low > 0 or high < 0
    assert (p < 1 - CONFIDENCE) == excludes_zero


def test_seed_level_tests_against_a_single_policy():
    a = models(30.0, seed=7)
    greedy = models(0.0, seed=8)[:1]
    tests = seed_level_tests(a, greedy)
    assert set(tests) == {"one-sample t-test", "Wilcoxon signed-rank"}
    assert all(p < 0.01 for p in tests.values())
    # The direction of the pair does not change the p-values.
    assert seed_level_tests(greedy, a) == pytest.approx(tests)


def test_single_models_get_no_seed_level_test():
    assert seed_level_tests(models(0.0, 9)[:1], models(0.0, 10)[:1]) == {}


def test_format_p():
    assert format_p(0.0123) == "p = 0.0123"
    assert format_p(0.0, 0.0002) == "p < 0.0002"
    assert format_p(1e-7) == "p < 0.0001"
