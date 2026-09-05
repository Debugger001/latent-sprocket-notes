# Reproducing the MIND MaskPO run

This guide reconstructs the latest MaskPO experiment from two inputs that are
kept outside Git:

1. an official MIND download; and
2. the public handoff ZIP containing `p2_rubric_reasoning_sft`.

The Phase-2 rubric adapter is the current RL starting point.  Rebuilding
Phase-1 SFT and training a new or improved Phase-2 SFT are intentionally left
as later, separately measurable stages.

## 1. Install

Python 3.10 or newer is supported.  The core metrics and credit-assignment code
has no runtime dependencies; GPU training uses the optional extras.

```bash
git clone https://github.com/Debugger001/latent-sprocket-notes.git
cd latent-sprocket-notes
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,training,dev]'
pytest
```

The UCLA environment inspected during this rebuild has Python 3.10,
PyTorch 2.6, Transformers 4.52, Accelerate 1.8, and PEFT 0.16.  These are not
claimed as the exact historical training versions: the archived adapter's
model configuration records Transformers 4.57.6, while its generation
configuration records 4.51.0.  The dependency ranges in `pyproject.toml`
cover that compatible family rather than asserting an unrecoverable exact
environment.

## 2. Recover the Phase-2 rubric adapter

Do not unpack the 4 GB audit archive into the repository.  Selectively import
the one adapter needed by this run:

```bash
python scripts/import_archived_adapter.py \
  /path/to/openbench-rerank-rl-26b4998-audit-copy.zip
```

The inspected handoff ZIP has SHA-256
`c196251d87fdd80549270afb11678a04cbbf2615a08ce5478ce9c71a1742004b`.
The importer does not require the rest of the archive to remain byte-identical;
it verifies the selected adapter files directly.

The default destination is
`artifacts/adapters/p2_rubric_reasoning_sft/`.  The importer:

- locates the adapter by its archive-relative suffix, so an outer ZIP directory
  is allowed;
- requires the adapter, tokenizer, model, and generation configuration files;
- streams only that directory to a temporary location;
- verifies every imported file against the canonical handoff checksums;
- refuses to overwrite an existing destination; and
- records each extracted file's byte count and SHA-256 digest in
  `IMPORT_MANIFEST.json`.

The source ZIP, extracted adapter, and manifest are ignored by Git.  Preserve
the manifest beside experiment metadata to establish exactly which SFT bytes
started the run.

## 3. Obtain MIND under its upstream terms

The official MIND page is <https://msnews.github.io/>.  The current download is
hosted as the gated Hugging Face dataset
[`yjw1029/MIND`](https://huggingface.co/datasets/yjw1029/MIND).  Request access,
review and accept the official terms, then authenticate interactively:

```bash
hf auth login
python scripts/download_mind.py --size small --extract --accept-license
```

`--accept-license` records only your local acknowledgement; it does not grant
access or replace the upstream acceptance flow.  MINDsmall publishes `train`
and `dev`; MINDlarge additionally publishes `test`.  Select splits explicitly
when needed:

```bash
python scripts/download_mind.py \
  --size large \
  --split train \
  --split dev \
  --extract \
  --accept-license
```

If automated access is unavailable, download the ZIPs manually after approval
and place/extract them under `data/raw/mind/`.  Never commit or redistribute the
raw files from this repository.  `behaviors.tsv` supplies impression, user,
timestamp, ordered history, and candidate click labels; `news.tsv` supplies
news ID, category, subcategory, title, abstract, URL, and two entity fields.
Only category, subcategory, title, and abstract are rendered into prompts.
The official `MINDsmall_train.zip` currently has 52,953,372 bytes and SHA-256
`a966e5138ad103376e9817e02395719bf1c62ec56e6e98c30d46fbb991a7fafa`;
record and verify that digest if the approved file is transferred between hosts.

## 4. Prepare deterministic examples

Prepare the official split in place.  This example preserves every row with at
least one positive and adds the current prompt to the generated JSONL:

```bash
python scripts/prepare_mind.py \
  --news data/raw/mind/MINDsmall_train/news.tsv \
  --behaviors data/raw/mind/MINDsmall_train/behaviors.tsv \
  --output-dir data/processed/mind-small-train-k20-800-seed42 \
  --max-candidates 20 \
  --sample-size 800 \
  --include-prompts \
  --seed 42
```

The RL stage deliberately uses `K <= 20`, where the policy can produce exact
permutations substantially more reliably. Evaluation may still report broader
slate slices, but those rows are not included in this RL training set.
For a different stable development subset, change `--sample-size N`. To create
a stable held-out partition within an input file, add `--validation-fraction F`.
The script emits `train.jsonl`, `validation.jsonl`, and `summary.json`. All are
gitignored because they derive from licensed source data.

Before a long run, inspect several prompts and confirm:

- candidate indices are one-based and labels are not exposed;
- the surrounding instruction contains the row's concrete `K`;
- the demonstration contains the literal
  `[permutation of 1 through K]`, not an arbitrary ordering; and
- all four rubric headers and `**Synthesis:**` match exactly.

## 5. Configure the latest algorithm

Copy the reference configuration so each experiment keeps an immutable record:

```bash
cp configs/maskpo_qwen3_1p7b.yaml configs/my_maskpo_run.yaml
```

The following values are recovered algorithm decisions and should not drift in
the canonical run:

| Setting | Canonical value |
| --- | ---: |
| original siblings | `4` |
| rubric probes per structurally valid original | `4` |
| RL slate boundary | `K <= 20` |
| rank reward | lenient nDCG |
| format reward maximum | `0.1` |
| mask residual scale | `0.05` |
| mask residual clip | `2.0` |
| answer mask weight | `0.5` |
| PPO clip | `0.2` |
| reference-KL coefficient | `0.001` |
| learning rate | `1e-5` |
| temperature / top-k / top-p | `0.6 / 20 / 0.95` |
| maximum new tokens | `2048` for originals and probes |
| distinct prompts per optimizer update | `8` initially; profile before increasing to `16` |
| effective original rollouts per update | `32` initially; `64` at prompt batch `16` |

Fields commented as reproducibility defaults were not recovered from the
historical runtime.  Changing those fields creates a well-specified new run,
but should not be described as reproducing an undocumented historical choice.

## 6. Run a smoke test, then train

Run a short, one-query smoke test before scheduling a full job.  The training
entry point reads the prepared JSONL and YAML configuration:

```bash
python scripts/train_maskpo.py \
  --config configs/maskpo_qwen3_1p7b.yaml \
  --train-file data/processed/mind-small-train-k20-800-seed42/train.jsonl \
  --max-query-steps 1 \
  --output-dir outputs/maskpo-smoke
```

Then remove the smoke-test override and launch the recorded configuration:

```bash
python scripts/train_maskpo.py \
  --config configs/my_maskpo_run.yaml \
  --train-file data/processed/mind-small-train-k20-800-seed42/train.jsonl
```

For each logged prompt group, verify diagnostics show:

- four originals sampled before the actor changes;
- up to 16 one-body-masked probes sampled by the same pre-update policy;
- invalid probes excluded, but valid zero deltas retained in the RMS scale;
- probes absent from the loss token count;
- nonzero rubric-region and answer-integer routing counts when parsing succeeds;
- explicit sequence fallback counts when semantic routing fails; and
- finite policy loss, KL, clip fraction, and active-token count.

Save the copied YAML, package versions, Git commit, adapter import manifest,
data-preparation summary, random seed, device model, and each checkpoint's hash
with the run outputs.

Offline prediction files use one JSON object per line with `completion` (or
`prediction`), `positive_indices`, and `k`.  Score them with the same lenient
parser and metrics as training:

```bash
python scripts/evaluate_outputs.py outputs/predictions.jsonl \
  --output outputs/predictions.scored.jsonl
```

The command prints macro aggregates and optionally adds a per-row `evaluation`
object to the scored JSONL.  Its `mrr_at_k` is reciprocal rank of the first
positive; it is a diagnostic helper, not the official MIND multi-click MRR
aggregation.  The canonical MaskPO ranking reward is nDCG.

## 7. Reproduce on a GPU server

Use any host with a recent NVIDIA GPU.  A single 80 GB GPU is the simplest
starting point for an actor plus frozen reference; memory-conscious runtimes may
place the two copies on separate devices.

```bash
ssh <gpu-host>
git clone https://github.com/Debugger001/latent-sprocket-notes.git
cd latent-sprocket-notes
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,training,dev]'
pytest
```

Transfer the handoff ZIP through an approved channel, run the selective adapter
importer on the server, and either download MIND there using your approved
account or transfer your locally obtained dataset through an allowed channel.
Do not place credentials in the repository or command-line arguments.

Choose an available GPU through the scheduler or environment used by that host,
run the one-step smoke test, and only then start the full job.  Low free disk
space is a practical risk: budget room for the base model cache, two model
instances, prepared data, optimizer state, and checkpoint rotation.

## 8. What “reproduce” means here

There are three distinct targets:

1. **Algorithm reproduction:** deterministic tests validate reward, masking,
   normalization, routing, and BNPO equations.
2. **Starting-point reproduction:** importing the same archive yields the exact
   `p2_rubric_reasoning_sft` adapter bytes, verified by the generated manifest.
3. **Training-run reproduction:** fixed config, package versions, data snapshot,
   seed, and hardware make a new run auditable, but stochastic GPU generation
   and missing historical optimizer/runtime state mean byte-identical historical
   checkpoints are not promised.

Use fixed held-out impressions and the same lenient grader for comparisons.
Report nDCG alongside the nine individual format checks so ranking improvements
cannot be mistaken for formatting improvements.
