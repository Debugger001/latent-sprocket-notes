"""MaskPO utilities for public fixed-slate MIND reranking experiments."""

from .advantages import (
    MaskPOConfig,
    aggregate_mask_advantages,
    combine_answer_advantages,
    grpo_group_advantages,
    rank_grpo_position_advantages,
    rank_shift_mask_residuals,
    rms_normalize_rubric_deltas,
    rubric_delta_advantages,
)
from .evaluation import aggregate_evaluations, evaluate_prediction
from .losses import BNPOLossOutput, bnpo_loss, tokenwise_bnpo_loss
from .masking import (
    MASKED_RUBRIC_CONTENT,
    build_counterfactual_prefix,
    build_counterfactual_prefixes,
)
from .metrics import dcg_at_k, mrr_at_k, ndcg_at_k, recall_at_k, slate_auc
from .parsers import parse_answer, parse_completion_structure
from .pipeline import plan_counterfactual_probes, score_maskpo_group
from .rewards import evaluate_format, format_reward, ranking_reward
from .routing import route_token_advantages

__all__ = [
    "BNPOLossOutput",
    "MASKED_RUBRIC_CONTENT",
    "MaskPOConfig",
    "aggregate_evaluations",
    "aggregate_mask_advantages",
    "bnpo_loss",
    "build_counterfactual_prefix",
    "build_counterfactual_prefixes",
    "combine_answer_advantages",
    "dcg_at_k",
    "evaluate_format",
    "evaluate_prediction",
    "format_reward",
    "grpo_group_advantages",
    "mrr_at_k",
    "ndcg_at_k",
    "parse_answer",
    "parse_completion_structure",
    "plan_counterfactual_probes",
    "rank_grpo_position_advantages",
    "rank_shift_mask_residuals",
    "ranking_reward",
    "recall_at_k",
    "rms_normalize_rubric_deltas",
    "route_token_advantages",
    "rubric_delta_advantages",
    "score_maskpo_group",
    "slate_auc",
    "tokenwise_bnpo_loss",
]
