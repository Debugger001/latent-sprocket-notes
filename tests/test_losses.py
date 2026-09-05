import math

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.losses import BNPOLossOutput, tokenwise_bnpo_loss


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_policy_loss"),
    [
        (2.0, 2.0, -2.4),  # positive advantage clips at 1 + epsilon
        (0.5, -2.0, 1.6),  # negative advantage clips at 1 - epsilon
    ],
)
def test_bnpo_clips_both_advantage_signs(
    ratio: float,
    advantage: float,
    expected_policy_loss: float,
):
    current = torch.tensor([[math.log(ratio)]], dtype=torch.float64)
    zeros = torch.zeros_like(current)

    output = tokenwise_bnpo_loss(
        current,
        zeros,
        current,  # zero KL
        torch.tensor([[advantage]], dtype=current.dtype),
        torch.ones_like(current, dtype=torch.bool),
        beta=0.0,
    )

    assert output.policy_loss.item() == pytest.approx(expected_policy_loss)
    assert output.loss.item() == pytest.approx(expected_policy_loss)
    assert output.clip_fraction.item() == pytest.approx(1.0)


def test_bnpo_masks_tokens_and_normalizes_once_across_the_batch():
    ratios = torch.tensor(
        [[1.0, 1.1, 1.19], [0.9, 1.18, 1.17]], dtype=torch.float64
    )
    current = ratios.log()
    old = torch.zeros_like(current)
    advantages = torch.tensor(
        [[1.0, 2.0, 1000.0], [3.0, 1000.0, 1000.0]], dtype=current.dtype
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    current[~mask] = float("nan")

    output = tokenwise_bnpo_loss(
        current,
        old,
        current,
        advantages,
        mask,
        beta=0.0,
    )

    # Three active tokens contribute 1*1, 1.1*2, and 0.9*3.  This differs
    # from averaging the two sequences after normalizing each independently.
    expected = -(1.0 + 2.2 + 2.7) / 3.0
    assert output.policy_loss.item() == pytest.approx(expected)
    assert output.token_count.item() == 3
    assert output.clip_fraction.item() == pytest.approx(0.0)


def test_bnpo_uses_sampled_token_kl_estimator_and_default_beta():
    current = torch.zeros((1, 1), dtype=torch.float64)
    old = torch.zeros_like(current)
    ref = torch.full_like(current, math.log(2.0))

    output = tokenwise_bnpo_loss(
        current,
        old,
        ref,
        torch.zeros_like(current),
        torch.ones_like(current, dtype=torch.bool),
    )

    expected_kl = 1.0 - math.log(2.0)
    assert output.kl.item() == pytest.approx(expected_kl)
    assert output.loss.item() == pytest.approx(0.001 * expected_kl)


def test_bnpo_gradients_flow_only_through_active_current_logps():
    current = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.tensor(
        [[math.log(2.0), math.log(3.0)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    advantages = torch.zeros_like(current, requires_grad=True)

    output = tokenwise_bnpo_loss(
        current,
        old,
        ref,
        advantages,
        torch.tensor([[1, 0]], dtype=torch.bool),
    )
    output.loss.backward()

    # At ref-current = log(2), d KL / d current = 1 - 2 = -1.
    assert current.grad is not None
    assert current.grad[0, 0].item() == pytest.approx(-0.001)
    assert current.grad[0, 1].item() == pytest.approx(0.0)
    assert old.grad is None
    assert ref.grad is None
    assert advantages.grad is None


def test_bnpo_is_zero_safe_when_no_completion_tokens_are_active():
    current = torch.tensor([[0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    output = tokenwise_bnpo_loss(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        torch.ones_like(current),
        torch.zeros_like(current, dtype=torch.bool),
    )

    assert isinstance(output, BNPOLossOutput)
    assert output.token_count.item() == 0
    assert output.loss.item() == pytest.approx(0.0)
    assert output.policy_loss.item() == pytest.approx(0.0)
    assert output.kl.item() == pytest.approx(0.0)
    assert output.clip_fraction.item() == pytest.approx(0.0)
    assert set(output.as_dict()) == {
        "loss",
        "policy_loss",
        "kl",
        "clip_fraction",
        "token_count",
    }

    output.loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))
