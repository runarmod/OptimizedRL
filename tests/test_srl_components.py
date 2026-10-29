import numpy as np
import pytest
import torch

from src.gym_envs.dap_env import N_ITEM_FEATURES
from src.srl.buffer import ReplayBuffer, Transition, discounted_returns
from src.srl.co_layer import top_k, top_k_numpy
from src.srl.config import Schedule
from src.srl.fenchel_young import perturbed_fenchel_young_loss
from src.srl.networks import Actor, FutureCritic, RewardCritic
from src.srl.target import softmax_target

N_ITEMS, K = 20, 4
PRICE = 4


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def random_features(batch: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.as_tensor(
        rng.uniform(1, 10, (batch, N_ITEMS, N_ITEM_FEATURES)), dtype=torch.float32
    )


def random_assortments(batch: int, seed: int = 0) -> torch.Tensor:
    return top_k(torch.as_tensor(np.random.default_rng(seed).normal(size=(batch, N_ITEMS))), K).float()


# Networks


def test_actor_is_linear_and_matches_linear_scores():
    actor = Actor(generator())
    features = random_features(3)
    w, b = actor.linear_scores()
    scores = actor(features)
    assert scores.shape == (3, N_ITEMS)
    np.testing.assert_allclose(scores.detach().numpy(), features.numpy() @ w + b, rtol=1e-5)


def test_actor_by_hand():
    actor = Actor(generator())
    with torch.no_grad():
        actor.hidden.weight.zero_()
        actor.hidden.weight[0] = 1.0
        actor.hidden.bias.zero_()
        actor.out.weight.zero_()
        actor.out.weight[0, 0] = 1.0
        actor.out.bias.fill_(0.5)
    features = random_features(2)
    np.testing.assert_allclose(
        actor(features).detach().numpy(), features.sum(-1).numpy() + 0.5, rtol=1e-6
    )


def test_reward_critic_uses_shown_items_in_index_order():
    critic = RewardCritic(K, generator())
    with torch.no_grad():
        critic.item.weight.zero_()
        critic.item.weight[0, PRICE] = 1.0
        critic.item.bias.zero_()
        critic.out.weight.zero_()
        critic.out.bias.zero_()
        # Only the block of the first shown item (lowest index) counts.
        critic.out.weight[0, 0] = 1.0
    features = random_features(3)
    assortments = random_assortments(3)
    values = critic(features, assortments)
    first_shown = assortments.argmax(-1)
    expected = features[torch.arange(3), first_shown, PRICE]
    np.testing.assert_allclose(values.detach().numpy(), expected.numpy(), rtol=1e-6)


def test_reward_critic_rejects_wrong_assortment_size():
    critic = RewardCritic(K, generator())
    assortments = torch.zeros(1, N_ITEMS)
    assortments[0, :K - 1] = 1
    with pytest.raises(ValueError):
        critic(random_features(1), assortments)


def test_future_critic_by_hand():
    critic = FutureCritic(N_ITEMS, generator())
    with torch.no_grad():
        critic.item.weight.zero_()
        # First hidden unit of every item: its assortment indicator.
        critic.item.weight[0, N_ITEM_FEATURES] = 1.0
        critic.item.bias.zero_()
        critic.hidden.weight.zero_()
        # The item outputs are flattened item by item, 5 units each.
        critic.hidden.weight[0, 0::5] = 1.0
        critic.hidden.bias.zero_()
        critic.out.weight.zero_()
        critic.out.weight[0, 0] = 1.0
        critic.out.bias.zero_()
    values = critic(random_features(2), random_assortments(2))
    np.testing.assert_allclose(values.detach().numpy(), [K, K])


def test_networks_are_reproducible_from_seed():
    a, b = Actor(generator(3)), Actor(generator(3))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)
    # Flux initializes biases to zero.
    assert torch.all(a.hidden.bias == 0)


# CO-layer


def test_top_k():
    scores = torch.tensor([[0.1, 5.0, -1.0, 3.0, 2.0, 0.0]])
    assert top_k(scores, 3).tolist() == [[0, 1, 0, 1, 1, 0]]
    assert top_k_numpy(scores[0].numpy(), 3).tolist() == [0, 1, 0, 1, 1, 0]


# Fenchel-Young loss


def fy_inputs(seed: int = 0, n_samples: int = 20, batch: int = 3):
    rng = np.random.default_rng(seed)
    theta = torch.tensor(rng.normal(size=(batch, N_ITEMS)), requires_grad=True)
    vertices = top_k(torch.as_tensor(rng.normal(size=(batch, 5, N_ITEMS))), K)
    weights = torch.softmax(torch.as_tensor(rng.normal(size=(batch, 5))), -1)
    target = (weights.unsqueeze(-1) * vertices).sum(1)
    noise = torch.as_tensor(rng.normal(size=(batch, n_samples, N_ITEMS)))
    return theta, target, noise


def test_fy_gradient_is_mean_action_minus_target():
    theta, target, noise = fy_inputs()
    epsilon = 0.7
    perturbed_fenchel_young_loss(theta, target, K, epsilon, noise).sum().backward()
    mean_action = top_k(theta.detach().unsqueeze(1) + epsilon * noise, K).mean(1)
    np.testing.assert_allclose(theta.grad.numpy(), (mean_action - target).numpy())


def test_fy_value_and_lower_bound():
    theta, target, noise = fy_inputs(seed=1)
    epsilon = 1.0
    loss = perturbed_fenchel_young_loss(theta, target, K, epsilon, noise)
    eta = theta.detach().unsqueeze(1) + epsilon * noise
    f = (eta * top_k(eta, K)).sum(-1).mean(-1)
    np.testing.assert_allclose(loss.detach().numpy(), (f - (theta.detach() * target).sum(-1)).numpy())
    # Each sample's action maximizes over the hull that contains the target, so
    # the estimate is at least epsilon * mean_k <Z_k, target>.
    bound = epsilon * (noise.mean(1) * target).sum(-1)
    assert torch.all(loss >= bound - 1e-9)


def test_fy_matches_finite_differences():
    theta, target, noise = fy_inputs(seed=2)
    loss = lambda t: perturbed_fenchel_young_loss(t, target, K, 1.0, noise).sum()
    loss(theta).backward()
    h = 1e-7
    with torch.no_grad():
        for b, i in [(0, 0), (1, 7), (2, 19)]:
            shifted = theta.detach().clone()
            shifted[b, i] += h
            numeric = (loss(shifted) - loss(theta.detach())) / h
            assert numeric.item() == pytest.approx(theta.grad[b, i].item(), abs=1e-5)


def test_fy_is_nonnegative_and_zero_at_a_dominant_target():
    theta, target, noise = fy_inputs(seed=3, n_samples=20_000, batch=4)
    loss = perturbed_fenchel_young_loss(theta.detach(), target, K, 1.0, noise)
    assert torch.all(loss > -0.05)

    vertex = top_k(torch.as_tensor(np.random.default_rng(4).normal(size=(4, N_ITEMS))), K)
    loss = perturbed_fenchel_young_loss(50.0 * vertex, vertex, K, 1.0, noise)
    assert torch.all(loss.abs() < 0.05)


# Target action


def test_softmax_target_is_a_convex_combination():
    rng = np.random.default_rng(0)
    candidates = top_k(torch.as_tensor(rng.normal(size=(2, 41, N_ITEMS))), K)
    values = torch.as_tensor(rng.normal(size=(2, 41)))
    target = softmax_target(candidates, values, 1.0)
    np.testing.assert_allclose(target.sum(-1).numpy(), [K, K])
    assert torch.all((target >= 0) & (target <= 1))


def test_softmax_target_tends_to_the_argmax():
    rng = np.random.default_rng(1)
    candidates = top_k(torch.as_tensor(rng.normal(size=(2, 41, N_ITEMS))), K)
    values = torch.as_tensor(rng.normal(size=(2, 41)))
    target = softmax_target(candidates, values, 1e-6)
    best = candidates[torch.arange(2), values.argmax(-1)]
    np.testing.assert_allclose(target.numpy(), best.numpy(), atol=1e-9)


def test_softmax_target_falls_back_to_argmax():
    candidates = top_k(torch.as_tensor(np.random.default_rng(2).normal(size=(1, 3, N_ITEMS))), K)
    values = torch.tensor([[1.0, float("inf"), 0.0]], dtype=candidates.dtype)
    target = softmax_target(candidates, values, 1.0)
    assert torch.equal(target[0], candidates[0, 1])


# Experience


def test_discounted_returns_keep_the_srl_quirk():
    np.testing.assert_allclose(discounted_returns(np.array([1.0, 2, 3, 4]), 0.5), [2.75, 3.5, 3.0, 0.0])
    np.testing.assert_allclose(discounted_returns(np.array([5.0]), 0.99), [0.0])


def test_replay_buffer_wraps_around():
    buffer = ReplayBuffer(3, np.random.default_rng(0))
    transitions = [Transition(np.zeros(1), np.zeros(1), float(i)) for i in range(5)]
    buffer.add(transitions)
    assert len(buffer) == 3
    assert [t.revenue for t in buffer.items] == [3.0, 4.0, 2.0]
    assert {t.revenue for t in buffer.sample(50)} <= {2.0, 3.0, 4.0}


def test_schedule():
    schedule = Schedule(start=2.0, end=1.0)
    assert schedule.value(0, 200) == 2.0
    assert schedule.value(100, 200) == pytest.approx(1.5)
    assert schedule.value(200, 200) == pytest.approx(1.0)
    assert schedule.value(300, 200) == 1.0
