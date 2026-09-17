import math

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.losses import (
    AnswerRLLossOutput,
    rank_grpo_loss,
    sequence_grpo_loss,
)


def test_sequence_grpo_averages_tokens_per_sequence_then_sequences():
    current = torch.zeros((2, 3), dtype=torch.float64)
    current[0, 2] = float("nan")
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    advantages = torch.tensor([1.0, 2.0], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    output = sequence_grpo_loss(
        current, old, ref, advantages, mask, beta=0.0
    )

    # Per-sequence means are -1 and -2, for -1.5 overall.  A global token
    # reduction would instead produce -1.6.
    assert isinstance(output, AnswerRLLossOutput)
    assert output.policy_loss.item() == pytest.approx(-1.5)
    assert output.token_count.item() == 5
    assert output.action_count.item() == 2


def test_sequence_grpo_uses_asymmetric_clipping_for_both_signs():
    ratios = torch.tensor([[0.5], [1.2]], dtype=torch.float64)
    current = ratios.log()
    zeros = torch.zeros_like(current)
    output = sequence_grpo_loss(
        current,
        zeros,
        current,
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.ones_like(current, dtype=torch.bool),
        clip_epsilon_low=0.06,
        clip_epsilon_high=0.08,
        beta=0.0,
    )

    assert output.policy_loss.item() == pytest.approx((0.94 - 1.08) / 2)
    assert output.clip_fraction.item() == pytest.approx(1.0)


def test_rank_grpo_uses_geometric_item_ratio_and_equal_item_average():
    # Item zero has token ratios 2 and 8 -> geometric ratio 4, clipped to 1.08.
    # Item one has ratio 1.  Both items receive equal weight despite lengths.
    ratios = torch.tensor([[2.0, 8.0, 1.0]], dtype=torch.float64)
    current = ratios.log()
    zeros = torch.zeros_like(current)
    output = rank_grpo_loss(
        current,
        zeros,
        current,
        torch.ones_like(current),
        torch.tensor([[0, 0, 1]], dtype=torch.long),
        torch.ones_like(current, dtype=torch.bool),
        expected_ranks=2,
        beta=0.0,
    )

    assert output.policy_loss.item() == pytest.approx((-1.08 - 1.0) / 2)
    assert output.clip_fraction.item() == pytest.approx(0.5)
    assert output.token_count.item() == 3
    assert output.action_count.item() == 2


def test_rank_grpo_uses_standard_tokenwise_per_completion_kl():
    current = torch.zeros((1, 4), dtype=torch.float64)
    old = torch.zeros_like(current)
    ref = torch.tensor([[math.log(2.0), 0.0, 0.0, 0.0]], dtype=torch.float64)
    output = rank_grpo_loss(
        current,
        old,
        ref,
        torch.zeros_like(current),
        torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        torch.ones_like(current, dtype=torch.bool),
        expected_ranks=2,
        beta=1.0,
    )

    first_item_kl = 1.0 - math.log(2.0)
    assert output.kl.item() == pytest.approx(first_item_kl / 4)
    assert output.loss.item() == pytest.approx(first_item_kl / 4)


def test_rank_grpo_keeps_fixed_k_denominator_for_short_outputs():
    values = torch.zeros((1, 1), dtype=torch.float64)
    output = rank_grpo_loss(
        values,
        values,
        values,
        torch.ones_like(values),
        torch.tensor([[0]], dtype=torch.long),
        torch.ones_like(values, dtype=torch.bool),
        expected_ranks=4,
        beta=0.0,
    )

    assert output.policy_loss.item() == pytest.approx(-0.25)


def test_rank_grpo_requires_one_advantage_per_item():
    values = torch.zeros((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="one advantage"):
        rank_grpo_loss(
            values,
            values,
            values,
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.ones_like(values, dtype=torch.bool),
            expected_ranks=2,
        )


def test_answer_rl_losses_detach_old_reference_and_advantages():
    current = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.full_like(current, math.log(2.0), requires_grad=True)
    advantages = torch.zeros((1,), dtype=torch.float64, requires_grad=True)
    output = sequence_grpo_loss(
        current,
        old,
        ref,
        advantages,
        torch.ones_like(current, dtype=torch.bool),
    )
    output.loss.backward()

    assert current.grad is not None
    assert current.grad.item() == pytest.approx(-0.001)
    assert old.grad is None
    assert ref.grad is None
    assert advantages.grad is None
