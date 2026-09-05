# MaskPO for MIND reranking

[![tests](https://github.com/Debugger001/latent-sprocket-notes/actions/workflows/tests.yml/badge.svg)](https://github.com/Debugger001/latent-sprocket-notes/actions/workflows/tests.yml)

This repository rebuilds the latest MaskPO experiment for fixed-slate news
reranking on the public [MIND dataset](https://msnews.github.io/).  It starts
from the archived `p2_rubric_reasoning_sft` Qwen3-1.7B LoRA adapter and assigns
credit to four independently maskable reasoning rubrics, the synthesis, and
individual answer items.

The repository contains the implementation, prompt template, deterministic
MIND preparation code, tests, and an inspectable training runtime.  It does
not contain raw MIND files, the handoff archive, extracted model weights, or
generated checkpoints.

## What is implemented

- The current four-rubric prompt, with the literal schema placeholder
  `[permutation of 1 through K]` instead of a sample ordering.
- Lenient nDCG grading for originals and counterfactuals: any parsed integer
  list is scored as emitted, and a positive candidate earns credit only at its
  first appearance.
- A sequence reward `R_seq = R_rank + R_format`, where the format component is
  the mean of nine checks scaled to `[0, 0.1]`.
- Four original sibling rollouts and four one-rubric counterfactual probes per
  structurally valid original.  A probe replaces exactly one rubric body with
  `[MASKED_RUBRIC_CONTENT]`, keeps the other three bodies fixed, and regenerates
  only the text after `**Synthesis:**`.
- RMS-normalized signed rubric deltas without mean centering, position-level
  Rank-GRPO credit, rank-shift residuals, semantic token routing, and a safe
  sequence-advantage fallback.
- Tokenwise clipped BNPO with PPO clip `0.2` and reference-KL coefficient
  `0.001`. Eight distinct prompts (32 originals) are accumulated per optimizer
  update and normalized once over all active original tokens. Counterfactual
  suffixes are scoring probes and receive no gradient.

The precise equations and edge-case rules are in
[docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## Quick start

Create a lightweight environment and run the model-independent test suite:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

For MIND download helpers and model training dependencies:

```bash
python -m pip install -e '.[data,training,dev]'
```

Import the Phase-2 rubric adapter from the original audit-copy ZIP.  The script
selectively extracts only the expected adapter directory and records a SHA-256
manifest:

```bash
python scripts/import_archived_adapter.py /path/to/openbench-rerank-rl-26b4998-audit-copy.zip
```

Request access to [`yjw1029/MIND`](https://huggingface.co/datasets/yjw1029/MIND),
review the upstream terms, authenticate with `hf auth login`, and download the
small split:

```bash
python scripts/download_mind.py --size small --extract --accept-license
```

Turn a local MIND split into deterministic, prompt-ready JSONL:

```bash
python scripts/prepare_mind.py \
  --news data/raw/mind/MINDsmall_train/news.tsv \
  --behaviors data/raw/mind/MINDsmall_train/behaviors.tsv \
  --output-dir data/processed/mind-small \
  --include-prompts
```

Score generated JSONL rows containing `completion`, `positive_indices`, and
`k` with exactly the same lenient grader used during training:

```bash
python scripts/evaluate_outputs.py outputs/predictions.jsonl \
  --output outputs/predictions.scored.jsonl
```

The checked-in starting configuration is
[`configs/maskpo_qwen3_1p7b.yaml`](configs/maskpo_qwen3_1p7b.yaml).  See
[docs/REPRODUCING.md](docs/REPRODUCING.md) for the training invocation, GPU-host
setup, checkpoints, and the distinction between recovered historical settings
and newly chosen runtime defaults.

## Repository layout

```text
configs/                         Reproducible MaskPO run configuration
docs/ALGORITHMS.md               Exact reward, counterfactual, routing, loss rules
docs/REPRODUCING.md              End-to-end local and GPU-server instructions
scripts/download_mind.py         Gated official-data download helper
scripts/import_archived_adapter.py  Selective adapter import from the handoff ZIP
scripts/prepare_mind.py          TSV parsing and deterministic JSONL preparation
scripts/evaluate_outputs.py      Matching offline lenient grader and aggregates
scripts/train_maskpo.py          Transformers/PEFT training entry point
src/openbench_rerank_rl/         Model-independent algorithm and runtime code
tests/                           Synthetic, redistributable unit tests
```

## Data and reproducibility boundaries

MIND has its own license and access terms.  Raw and processed data are ignored
by Git and must not be pushed to this repository.  The adapter and future
checkpoints are also local artifacts by default.

The archived adapter provides a byte-identical Phase-2 SFT starting point when
imported from the same handoff ZIP.  Phase-1 SFT and a fresh/improved Phase-2 SFT
can be rebuilt as separate follow-up stages.  Optimizer state and a complete
historical RL runtime were not present in the public handoff, so settings marked
as “reproducibility default” are reasonable choices for a new run, not claims
about the original experiment.
