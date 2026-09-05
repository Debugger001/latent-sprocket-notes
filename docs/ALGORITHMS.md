# MaskPO algorithm specification

This document is the executable specification for the latest MIND MaskPO
implementation.  It supersedes earlier archived variants that used eight
rollouts, mean-centered mask deltas, or added sequence credit to rubric bodies.

## 1. Task and parser

A query contains a clicked-news history, an impression time, and a candidate
slate of size `K`.  Candidate IDs are the one-based integers `1..K`.  The policy
must emit four rubric sections followed by synthesis and an answer:

```text
<think>
**Section And Topic Affinity:** ...
**Entity And Storyline Continuity:** ...
**Candidate Angle And Marginal Novelty:** ...
**Temporal And Session Intent:** ...
**Synthesis:** ...
</think>
<answer>
[permutation of 1 through K]
</answer>
```

The literal line `[permutation of 1 through K]` is a schema description in the
demonstration, not an actual example ordering.  The surrounding prompt states
the row's concrete `K` and requires an exact `1..K` permutation.

The ranking parser is intentionally lenient.  If an integer list can be parsed,
it is scored as emitted; it is not deduplicated, completed, or repaired.
Out-of-range IDs and duplicates consume their emitted positions.  Each positive
ID can contribute only once, at its first appearance.  The same parser and
ranking scorer are used for original completions and counterfactual suffixes.

## 2. Ranking and format rewards

For an emitted list `y`, binary-relevance DCG is

```text
DCG(y) = sum over positions p:
         first_occurrence_positive(y[p]) / log2(p + 1)
```

where positions are one-based.  With `P` distinct positive candidates and
slate cutoff `K`,

```text
IDCG = sum_{p=1}^{min(|P|, K)} 1 / log2(p + 1)
R_rank(y) = nDCG(y) = DCG(y) / IDCG
```

An unparseable original receives ranking reward zero.  A parseable but malformed
list still receives its lenient nDCG.

The dense format reward is based on nine independent binary checks:

1. exactly one valid `<think>...</think>` / `<answer>...</answer>` envelope;
2. `**Section And Topic Affinity:**` appears exactly once in the think block and
   in its prescribed relative position;
3. `**Entity And Storyline Continuity:**` does likewise;
4. `**Candidate Angle And Marginal Novelty:**` does likewise;
5. `**Temporal And Session Intent:**` does likewise;
6. `**Synthesis:**` appears exactly once inside the think block, after the four
   rubrics;
7. an integer list is parseable;
8. parsed IDs are unique and all lie in `1..K`;
9. the parsed list is exactly a permutation of `1..K`.

If `b_j` is check `j`, then

```text
R_format(y) = 0.1 * (1/9) * sum_{j=1}^{9} b_j
R_seq(y) = R_rank(y) + R_format(y)
```

For each query, sample exactly six sibling originals.  Sequence credit is the
population z-score within those six:

```text
A_seq,i = (R_seq,i - mean_j R_seq,j) / (std_j R_seq,j + epsilon)
```

When the within-group standard deviation is effectively zero, all six sequence
advantages are zero.  `R_format` is included here once; it is never separately
added to rubric or answer-item advantages.

## 3. One-rubric counterfactuals

All sampling for a training group uses a frozen pre-update policy.  After each
original rollout `y_i`, locate the four rubric bodies.  For rubric `r`, build an
assistant prefix that:

1. retains the original text through all four rubric sections;
2. replaces only body `r` with `[MASKED_RUBRIC_CONTENT]`;
3. leaves the other three original bodies unchanged; and
4. ends immediately after the literal `**Synthesis:**` header.

The same frozen policy generates only the new synthesis prose and final answer.
Thus a complete group has at most `6 x 4 = 24` valid counterfactuals.  These
suffixes are probes only and are excluded from the gradient batch.

A probe is valid whenever the shared answer parser finds an integer list.  It
does not have to pass the exact-permutation check.  For every valid probe,

```text
delta_{i,r} = R_rank(y_i) - R_rank(y_i,mask(r))
```

Only ranking reward enters this delta.  Probe format is deliberately irrelevant
to rubric causality.  Across every valid element `V_q` of one query's `6 x 4`
matrix,

```text
s_q = sqrt((1 / |V_q|) * sum_{(j,u) in V_q} delta_{j,u}^2)
A_rubric,i,r = delta_{i,r} / s_q
```

There is no subtraction of the delta mean.  Helpful rubrics therefore retain a
positive sign and harmful rubrics retain a negative sign.  Invalid probes are
excluded from both numerator and denominator.  Valid zero deltas participate in
the denominator and receive exactly zero.  If all valid deltas are zero (or
there are no valid probes), the available rubric advantages safely resolve to
zero.

## 4. Position and rank-shift credit

Rank-GRPO first computes each emitted answer position's local nDCG contribution
and z-normalizes contributions across the six siblings at the same position:

```text
A_rank,i,p = zscore_siblings(contribution(y_i[p], p))
```

If a parseable sibling list ends before position `p`, that sibling is absent
from the position-`p` normalization bucket; no synthetic zero is imputed.

For mask residuals, compare a positive candidate's original rank `a` with its
rank `b` under one valid mask.  Missing candidates have zero discount.  For
nDCG,

```text
d(rank) = 1 / (log2(rank + 1) * IDCG)
m = d(a) - d(b)
```

If `m > 0`, masking moved the positive down (or removed it), so assign residual
`m` to that positive item in the original answer.  If `m < 0`, masking improved
the positive, so assign the negative residual to the original items whose slots
the positive crossed.  Multi-positive residuals add.  A residual exists only
where this rule assigns one; absent entries are not implicit zeros.

For each original candidate `c`, average its existing residual entries over the
rollout's valid masks and scale them:

```text
A_mask,i(c) = clip(mean(existing residuals for c) / 0.05, -2, 2)
```

The integer at answer position `p`, whose value is `c_i,p`, receives

```text
A_answer,i,p = A_rank,i,p + 0.5 * A_mask,i(c_i,p)
```

## 5. Semantic token routing

Only original completion tokens enter the optimizer.  Their advantages are:

| Token region | Advantage |
| --- | --- |
| `<think>`, `<answer>`, headers, delimiters, punctuation | `A_seq,i` |
| body of rubric `r` | `A_rubric,i,r` only |
| synthesis prose | `A_seq,i` |
| integer at answer position `p` | `A_rank,i,p + 0.5 A_mask,i(c_i,p)` |

All subword tokens overlapping one answer integer receive the same answer-item
advantage.  Whitespace-only separator tokens around a rubric body remain
formatting tokens and receive sequence credit.  If a particular
counterfactual is invalid, that unavailable rubric body conservatively retains
sequence credit; this body-local rule is an explicit implementation choice for
the otherwise undefined `A_rubric,i,r`, not a recovered historical setting.
If the completion structure or integer-to-token alignment is ambiguous, the
entire completion falls back to `A_seq,i`.

## 6. Tokenwise BNPO objective

BNPO here means **batch-normalized policy optimization**: active token losses
are summed over the whole batch and divided once by the total number of active
completion tokens.  It is not an additional reward normalization.

Let `pi_theta` be the current policy, `pi_old` the frozen pre-update policy that
sampled the group, and `pi_ref` the frozen reference initialized from the same
Phase-2 SFT adapter.  For active original token `t`,

```text
rho_t = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))
rho_clip,t = clip(rho_t, 0.8, 1.2)
L_policy,t = -min(rho_t A_t, rho_clip,t A_t)

x_t = log pi_ref(a_t|s_t) - log pi_theta(a_t|s_t)
KL_t = exp(x_t) - x_t - 1
```

With `T` the set of active, non-padding original completion tokens,

```text
L_BNPO = (1 / |T|) * sum_{t in T} (L_policy,t + 0.001 * KL_t)
```

Old-policy log probabilities, reference log probabilities, routed advantages,
and counterfactual generations are detached targets.  The actor is updated only
after all six originals and their probes have been sampled and scored.

## 7. Constants for the latest run

| Quantity | Value |
| --- | ---: |
| Sibling originals per query | `6` |
| Rubrics per valid original | `4` |
| Ranking metric | lenient binary nDCG |
| Maximum format reward | `0.1` |
| Mask residual scale `tau` | `0.05` |
| Mask residual clip | `[-2, 2]` |
| Answer mask weight | `0.5` |
| PPO ratio clip | `0.2` |
| Reference-KL coefficient | `0.001` |
| Sampling temperature | `0.6` |
| Sampling top-k | `20` |
| Sampling top-p | `0.95` |

Optimizer schedule, batch packing, maximum generation length, checkpoint
cadence, and random seed were not recovered as historical facts.  Values for
them in the checked-in configuration are explicitly labeled reproducibility
defaults.
