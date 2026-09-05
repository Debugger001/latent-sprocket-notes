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
