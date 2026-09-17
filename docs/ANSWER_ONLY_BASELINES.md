# Answer-only GRPO baselines

This repository includes two answer-only controls for the MIND MaskPO run:

- vanilla GRPO, with one sibling-normalized sequence nDCG advantage and
  token-level importance ratios; and
- Rank-GRPO(`exp_inf`), with immediate relevance normalized across siblings at
  each rank and one geometric-mean importance ratio per ranked item.

Both controls start from the audited
`p2_teacher_answer_only_sft` Qwen3-1.7B LoRA adapter. They do **not** start from
the rubric-reasoning adapter: doing so would confound the method comparison
with a prompt/checkpoint mismatch.

## Prompt and grading contract

The baseline data uses the archived MIND answer-only prompt verbatim. It asks
for one bare JSON-style list, contains the historical fixed demonstration
`[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]`, and contains no reasoning or `<answer>`
wrapper. The fixed example is retained because the starting adapter was trained
with it; replacing it with the newer literal schema would be a separate prompt
ablation. Generation explicitly uses Qwen3's non-thinking assistant prefix, so
the sampled completion itself starts with the list instead of an unrequested
`<think>` trace.

Ranking reward uses the same lenient integer-list parser as MaskPO. A parsed
list is scored exactly as emitted; invalid IDs and duplicates occupy positions,
and a positive earns credit only on its first occurrence. Strict bare-list,
valid-ID, uniqueness, and exact-permutation rates are logged separately and do
not enter either baseline reward.

## Rank-GRPO adaptation to JSON lists

The public Rank-GRPO implementation segments newline-separated recommendation
items. The archived MIND answer is a single-line JSON list, so copying that
segmenter would incorrectly treat the whole ranking as one action. Here each
parsed integer is one item action. The opening bracket belongs to the first
item, trailing comma/space belongs to the preceding item, and the closing
bracket and final stop token belong to the last item. Multi-token integers form
one action.

For `exp_inf`, rank reward is binary immediate relevance. Every sibling
participates in a fixed `G x K` matrix: missing positions, invalid IDs, and
repeated positives receive zero. Rank columns use the released implementation's
sample standard deviation plus `1e-4`. Premature termination and overflow use
the paper's `-0.1` penalties; the released code's additional `+0.1` exact-length
bonus is deliberately omitted because it is not part of the paper definition.

The item importance ratio is

```text
rho_item = exp(mean_token(log pi_theta - log pi_old))
```

and PPO clipping is applied once per item. Policy terms use the paper's fixed
`1/(G*K)` denominator: missing ranks contribute zero, a premature-stop action
occupies one missing-rank slot, and overflow penalties are additive beyond
`K`. Reference KL remains a separate tokenwise per-completion mean. This
differs from the released repository's default TRL `bnpo` reduction, which
repeats an item loss over its tokens and therefore weights longer item strings
more heavily. Vanilla GRPO likewise averages tokens within each completion
before averaging sibling completions.

## Controlled-comparison settings

Shared settings match the stable MaskPO run:

| Setting | Value |
| --- | ---: |
| Starting adapter | `p2_teacher_answer_only_sft` |
| Slate boundary | `K <= 20` |
| Siblings per prompt | `4` |
| Prompts / original rollouts per update | `8 / 32` |
| Learning rate | `5e-6` |
| Reference-KL coefficient | `0.01` |
| Temperature / top-k / top-p | `0.6 / 20 / 0.95` |
| Maximum new tokens | `2048` |
| PPO passes per fresh batch | `1` |
| Updates | `3,000` |
| Validation | fixed 200 rows, baseline and every 10 updates |
| Checkpoints | every 50 updates, newest 5 retained |

Method-specific PPO bounds follow the paper: `0.20/0.26` for GRPO and
`0.06/0.08` for Rank-GRPO.

The experiment is therefore an algorithm-faithful controlled baseline, not a
literal replication of the paper's movie dataset, full-parameter optimizer,
G=8, temperature 1.0, or effective batch 384.

## Materialize and launch

Import the answer-only adapter:

```bash
python scripts/import_archived_adapter.py \
  /path/to/openbench-rerank-rl-26b4998-audit-copy.zip \
  --adapter p2_teacher_answer_only_sft
```

Prepare the same 24,000/200 row membership and order with the answer-only
template:

```bash
python scripts/prepare_mind.py \
  --news data/raw/mind/MINDsmall_train/news.tsv \
  --behaviors data/raw/mind/MINDsmall_train/behaviors.tsv \
  --output-dir data/processed/mind-small-train-k20-24k-answer-only-seed42 \
  --max-candidates 20 \
  --sample-size 24200 \
  --validation-fraction 0.008264462809917356 \
  --shuffle-training \
  --include-prompts \
  --prompt-style answer-only \
  --seed 42
```

After a one-query smoke test, launch either checked-in configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_answer_rl.py \
  --config configs/grpo_answer_only_qwen3_1p7b_a100_stable.yaml \
  --device-map cuda:0 \
  --reference-device-map cuda:1

CUDA_VISIBLE_DEVICES=0,1 python scripts/train_answer_rl.py \
  --config configs/rank_grpo_answer_only_qwen3_1p7b_a100_stable.yaml \
  --device-map cuda:0 \
  --reference-device-map cuda:1
```

The original method sources are the
[Rank-GRPO paper](https://arxiv.org/abs/2510.20150) and the pinned
[reference implementation](https://github.com/yaochenzhu/Rank-GRPO/tree/0808d82c0807396c07b4f4815a6f9bd9c4f03ae8).
