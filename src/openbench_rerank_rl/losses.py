"""PyTorch loss utilities for Rank-MaskPO training.

PyTorch is imported only when :func:`tokenwise_bnpo_loss` is called so the
rest of the lightweight evaluation package remains usable without the
training dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class BNPOLossOutput:
    """Scalar BNPO objective and detached-friendly training diagnostics.

    All fields are scalar tensors.  Keeping the losses as tensors preserves
    autograd; callers can convert diagnostics to Python numbers only when they
    log them.
    """

    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor
    token_count: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return the output in a logger-friendly mapping."""

        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "kl": self.kl,
            "clip_fraction": self.clip_fraction,
            "token_count": self.token_count,
        }


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without training deps
        raise ImportError(
            "tokenwise_bnpo_loss requires PyTorch; install the training dependencies"
        ) from exc
    return torch


def tokenwise_bnpo_loss(
    current_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    token_advantages: Tensor,
    completion_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.001,
) -> BNPOLossOutput:
    """Compute a tokenwise clipped BNPO objective over original completions.

    For active completion token ``t``, the policy ratio and sampled-token KL
    estimator are

    ``ratio_t = exp(current_logp_t - old_logp_t)``

    ``kl_t = exp(ref_logp_t - current_logp_t)
              - (ref_logp_t - current_logp_t) - 1``.

    The PPO surrogate uses symmetric ratio clipping.  BNPO then sums token
    losses across the batch and divides once by the total active-token count,
    rather than first averaging each completion.  Old-policy probabilities,
    reference probabilities, and advantages are detached because they are
    fixed targets.  Counterfactual probes are intentionally absent from this
    API and therefore cannot receive gradient through this loss.
    """

    torch = _torch()

    if not isinstance(current_logps, torch.Tensor):
        raise TypeError("current_logps must be a torch.Tensor")
    if not current_logps.is_floating_point():
        raise TypeError("current_logps must have a floating-point dtype")
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    def fixed_tensor(value: Tensor, name: str) -> Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=current_logps.dtype,
            device=current_logps.device,
        )
        if tensor.shape != current_logps.shape:
            raise ValueError(
                f"{name} must have shape {tuple(current_logps.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.detach()

    old = fixed_tensor(old_logps, "old_logps")
    ref = fixed_tensor(ref_logps, "ref_logps")
    advantages = fixed_tensor(token_advantages, "token_advantages")

    active = torch.as_tensor(completion_mask, device=current_logps.device)
    if active.shape != current_logps.shape:
        raise ValueError(
            f"completion_mask must have shape {tuple(current_logps.shape)}, "
            f"got {tuple(active.shape)}"
        )
    active = active.bool()
    token_count = active.sum()
    denominator = token_count.clamp_min(1).to(dtype=current_logps.dtype)

    # Select first rather than multiplying by a zero mask after computing the
    # objective.  Besides doing less work, this prevents ignored padding values
    # such as NaN or +/-inf from contaminating the reduction.
    current_active = current_logps[active]
    old_active = old[active]
    ref_active = ref[active]
    advantages_active = advantages[active]

    log_ratio = current_active - old_active
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(
        ratio * advantages_active,
        clipped_ratio * advantages_active,
    )
    per_token_policy_loss = -surrogate

    ref_current_delta = ref_active - current_active
    per_token_kl = torch.exp(ref_current_delta) - ref_current_delta - 1.0

    policy_loss = per_token_policy_loss.sum() / denominator
    kl = per_token_kl.sum() / denominator
    loss = policy_loss + beta * kl

    clipped = (ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)
    clip_fraction = clipped.to(current_logps.dtype).sum() / denominator

    return BNPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=clip_fraction,
        token_count=token_count,
    )


def bnpo_loss(
    current_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    token_advantages: Tensor,
    completion_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.001,
) -> BNPOLossOutput:
    """Concise alias for :func:`tokenwise_bnpo_loss`."""

    return tokenwise_bnpo_loss(
        current_logps,
        old_logps,
        ref_logps,
        token_advantages,
        completion_mask,
        clip_epsilon=clip_epsilon,
        beta=beta,
    )


@dataclass(frozen=True)
class AnswerRLLossOutput:
    """Loss and diagnostics for a pure answer-only GRPO objective.

    ``action_count`` counts sequence actions for vanilla GRPO and item actions
    for Rank-GRPO.  ``token_count`` separately reports the sampled tokens used
    to form those actions and the tokenwise reference-KL estimator.
    """

    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor
    token_count: Tensor
    action_count: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "kl": self.kl,
            "clip_fraction": self.clip_fraction,
            "token_count": self.token_count,
            "action_count": self.action_count,
        }


def _validate_answer_rl_loss_inputs(
    current_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    completion_mask: Tensor,
    *,
    clip_epsilon_low: float,
    clip_epsilon_high: float | None,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Validate shared tensor inputs and detach fixed policies/targets."""

    torch = _torch()
    if not isinstance(current_logps, torch.Tensor):
        raise TypeError("current_logps must be a torch.Tensor")
    if not current_logps.is_floating_point():
        raise TypeError("current_logps must have a floating-point dtype")
    if current_logps.ndim != 2:
        raise ValueError("answer-only log probabilities must have shape (batch, tokens)")
    if current_logps.shape[0] == 0:
        raise ValueError("answer-only losses require a non-empty batch")
    if not 0.0 <= clip_epsilon_low < 1.0:
        raise ValueError("clip_epsilon_low must be in [0, 1)")
    high = clip_epsilon_low if clip_epsilon_high is None else clip_epsilon_high
    if high < 0.0:
        raise ValueError("clip_epsilon_high must be non-negative")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    def fixed(value: Tensor, name: str) -> Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=current_logps.dtype,
            device=current_logps.device,
        )
        if tensor.shape != current_logps.shape:
            raise ValueError(
                f"{name} must have shape {tuple(current_logps.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.detach()

    old = fixed(old_logps, "old_logps")
    ref = fixed(ref_logps, "ref_logps")
    active = torch.as_tensor(completion_mask, device=current_logps.device)
    if active.shape != current_logps.shape:
        raise ValueError(
            f"completion_mask must have shape {tuple(current_logps.shape)}, "
            f"got {tuple(active.shape)}"
        )
    return old, ref, active.bool(), float(high)


def sequence_grpo_loss(
    current_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    sequence_advantages: Tensor,
    completion_mask: Tensor,
    *,
    clip_epsilon_low: float = 0.2,
    clip_epsilon_high: float | None = None,
    beta: float = 0.001,
) -> AnswerRLLossOutput:
    """Paper-faithful vanilla GRPO loss for answer-only completions.

    Importance ratios and reverse-KL estimates are tokenwise.  Each sequence
    is first averaged over its active completion tokens, and sequence means
    are then averaged with equal query/rollout weight.  This is intentionally
    different from the token-global BNPO reduction used by MaskPO.
    """

    torch = _torch()
    old, ref, active, high = _validate_answer_rl_loss_inputs(
        current_logps,
        old_logps,
        ref_logps,
        completion_mask,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        beta=beta,
    )
    advantages = torch.as_tensor(
        sequence_advantages,
        dtype=current_logps.dtype,
        device=current_logps.device,
    )
    if advantages.shape != (current_logps.shape[0],):
        raise ValueError(
            "sequence_advantages must have shape (batch,), got "
            f"{tuple(advantages.shape)}"
        )
    advantages = advantages.detach()

    # Sanitize padding before arithmetic so ignored NaN/inf sentinels cannot
    # contaminate either the forward reduction or backward pass.
    zero = torch.zeros((), dtype=current_logps.dtype, device=current_logps.device)
    current = torch.where(active, current_logps, zero)
    old_safe = torch.where(active, old, zero)
    ref_safe = torch.where(active, ref, zero)
    log_ratio = current - old_safe
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio, 1.0 - clip_epsilon_low, 1.0 + high
    )
    advantage_tokens = advantages.unsqueeze(1).expand_as(current_logps)
    policy_tokens = -torch.minimum(
        ratio * advantage_tokens,
        clipped_ratio * advantage_tokens,
    )
    ref_current_delta = ref_safe - current
    kl_tokens = torch.exp(ref_current_delta) - ref_current_delta - 1.0

    active_float = active.to(current_logps.dtype)
    counts = active_float.sum(dim=1)
    denominators = counts.clamp_min(1.0)
    policy_per_sequence = (policy_tokens * active_float).sum(dim=1) / denominators
    kl_per_sequence = (kl_tokens * active_float).sum(dim=1) / denominators

    low_clipped = (ratio < 1.0 - clip_epsilon_low) & (advantage_tokens < 0)
    high_clipped = (ratio > 1.0 + high) & (advantage_tokens > 0)
    clipped_per_sequence = (
        ((low_clipped | high_clipped).to(current_logps.dtype) * active_float).sum(dim=1)
        / denominators
    )

    policy_loss = policy_per_sequence.mean()
    kl = kl_per_sequence.mean()
    loss = policy_loss + beta * kl
    return AnswerRLLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=clipped_per_sequence.mean(),
        token_count=active.sum(),
        action_count=(counts > 0).sum(),
    )


def rank_grpo_loss(
    current_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    token_advantages: Tensor,
    item_ids: Tensor,
    completion_mask: Tensor,
    *,
    expected_ranks: int,
    clip_epsilon_low: float = 0.06,
    clip_epsilon_high: float | None = 0.08,
    beta: float = 0.001,
) -> AnswerRLLossOutput:
    """Rank-GRPO loss with geometric item ratios and equal item weighting.

    For item ``i``, the log importance weight is the arithmetic mean of its
    token log-ratios, so exponentiation yields the geometric mean probability
    ratio from the paper. Policy loss is computed once per item rather than
    once per token. The paper's fixed ``1 / (G K)`` denominator is preserved
    through ``expected_ranks``: ungenerated ranks contribute zero without
    amplifying a short completion's earlier actions, while overflow penalties
    remain additive. Reference KL remains a conventional tokenwise
    per-completion mean and is averaged across sibling completions.

    A short completion includes its observable termination action;
    ungenerated missing ranks have no policy action even though their zero
    reward participates in the fixed ``G x K`` advantage normalization.
    """

    torch = _torch()
    if type(expected_ranks) is not int or expected_ranks <= 0:
        raise ValueError("expected_ranks must be a positive integer")
    old, ref, base_mask, high = _validate_answer_rl_loss_inputs(
        current_logps,
        old_logps,
        ref_logps,
        completion_mask,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        beta=beta,
    )
    advantages = torch.as_tensor(
        token_advantages,
        dtype=current_logps.dtype,
        device=current_logps.device,
    ).detach()
    if advantages.shape != current_logps.shape:
        raise ValueError(
            f"token_advantages must have shape {tuple(current_logps.shape)}, "
            f"got {tuple(advantages.shape)}"
        )
    segments = torch.as_tensor(item_ids, device=current_logps.device)
    if segments.shape != current_logps.shape:
        raise ValueError(
            f"item_ids must have shape {tuple(current_logps.shape)}, "
            f"got {tuple(segments.shape)}"
        )
    if segments.dtype == torch.bool or segments.is_floating_point():
        raise TypeError("item_ids must have an integer dtype")
    segments = segments.to(dtype=torch.long)
    active = base_mask & (segments >= 0)

    query_policy: list[Tensor] = []
    query_kl: list[Tensor] = []
    query_clip: list[Tensor] = []
    action_count = torch.zeros((), dtype=torch.long, device=current_logps.device)

    for row in range(current_logps.shape[0]):
        row_active = active[row]
        row_items = torch.unique(segments[row][row_active], sorted=True)
        item_policy: list[Tensor] = []
        item_clip: list[Tensor] = []
        for item_id in row_items:
            item_mask = row_active & (segments[row] == item_id)
            current_item = current_logps[row][item_mask]
            old_item = old[row][item_mask]
            advantage_item = advantages[row][item_mask]
            first_advantage = advantage_item[0]
            if not torch.allclose(
                advantage_item,
                first_advantage.expand_as(advantage_item),
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError("all tokens in one Rank-GRPO item need one advantage")

            mean_log_ratio = (current_item - old_item).mean()
            ratio = torch.exp(mean_log_ratio)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - clip_epsilon_low, 1.0 + high
            )
            item_policy.append(
                -torch.minimum(
                    ratio * first_advantage,
                    clipped_ratio * first_advantage,
                )
            )
            item_clip.append(
                (
                    ((ratio < 1.0 - clip_epsilon_low) & (first_advantage < 0))
                    | ((ratio > 1.0 + high) & (first_advantage > 0))
                ).to(current_logps.dtype)
            )

        if item_policy:
            rank_denominator = current_logps.new_tensor(float(expected_ranks))
            query_policy.append(torch.stack(item_policy).sum() / rank_denominator)
            query_clip.append(torch.stack(item_clip).sum() / rank_denominator)
            action_count = action_count + len(item_policy)
        else:
            # Keep an autograd connection while making an empty query's
            # contribution exactly zero.
            row_zero = current_logps[row][row_active].sum() * 0.0
            query_policy.append(row_zero)
            query_clip.append(row_zero)

        if bool(row_active.any().to(device="cpu").item()):
            delta = ref[row][row_active] - current_logps[row][row_active]
            query_kl.append((torch.exp(delta) - delta - 1.0).mean())
        else:
            query_kl.append(current_logps[row].sum() * 0.0)

    policy_loss = torch.stack(query_policy).mean()
    kl = torch.stack(query_kl).mean()
    loss = policy_loss + beta * kl
    return AnswerRLLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=torch.stack(query_clip).mean(),
        token_count=active.sum(),
        action_count=action_count,
    )
