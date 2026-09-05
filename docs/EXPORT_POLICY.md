# Export Policy

This repository is built by allowlist, not by copying experiment directories.

Allowed:

- source code that is standalone and does not require proprietary training
  infrastructure;
- public-benchmark task descriptions, prompt templates, metric definitions, and
  sanitized analysis;
- aggregate eval metrics and plots from public benchmark eval sets;
- small teacher-trajectory samples containing only public benchmark text;
- compressed full teacher trajectories for allowlisted open benchmarks when they
  contain only public prompt text, teacher responses, stable row IDs, dataset
  names, reasoning style, split, and chat roles;
- model-weight exports for public-benchmark experiments when they are explicitly
  audited and limited to open-base-model adapters or other redistributable
  weights, with large tensors tracked through Git LFS.
- raw model-output exports for allowlisted public benchmark evals when they
  contain only public prompt text, model responses, labels, row IDs, and
  aggregate/per-row benchmark metrics, with large files tracked through Git LFS.

Excluded:

- internal datasets, member data, proprietary catalog data, and local data dumps;
- internal model weights, proprietary checkpoints, unaudited adapters, tokenizer
  assets from non-public models, or internal manifests;
- generated training/eval runtime configs containing internal paths or
  infrastructure details;
- answer-only and format-only trajectory derivatives unless explicitly reviewed;
- object-storage URIs, internal job IDs, queue names, experiment-tracker links,
  and absolute local paths;
- auth material, environment files, keys, and certificates.

Before pushing, run a file inventory, a large-file check, and a text scan for:

- local workspace paths;
- object-storage paths or model-artifact paths;
- internal job or queue identifiers;
- experiment-tracker links;
- auth material, environment files, keys, or certificates.

Expected result: no non-LFS files over 10 MB and no sensitive internal matches.
Public benchmark text may contain ordinary public references to companies,
products, or media titles.
