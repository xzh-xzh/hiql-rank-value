# Rank value implementation notes

## Default objective

For each valid transition index `i`, `HGCDataset.sample` locates the terminal
index of its trajectory. It samples `k` uniformly from the remaining future
indices and then samples `j` uniformly from `[i+1, k]`. Thus `i < j <= k` and
`g = s_k`. Uniform conditional sampling does not give a uniform distribution over
all possible `(i, j, k)` triples.

`rank_value_loss` evaluates both current value heads at `(s_i, g)` and `(s_j, g)`.
It sums the two per-head hinge penalties and averages over valid batch rows:

```text
gap[h, b] = V_h(s_j[b], g[b]) - V_h(s_i[b], g[b])
L_rank = sum_{h,b} mask[b] * relu(margin - gap[h,b])
         / max(sum_b mask[b], 1)
```

There is no additional division by the number of heads. Both endpoints receive
gradients. A common additive shift in values does not change the ranking loss.
The objective sets a minimum desired gap; it does not pull a gap above the margin
back down to the margin.

Rows without a future state have `rank_masks=0`. They remain available for the
actor losses. The rank critic does not use rewards, masks for Bellman backups,
next observations, or the target value network. The target network is retained in
the network structure for TD compatibility but is not soft-updated in rank mode.

Rank mode overrides the TD value-goal mixture and geometric sampling settings.
The low/high actor goal construction and policy losses remain HIQL's. Image
augmentation uses shared crop offsets for the sampled rank observations and other
goal observations, including the `j=k` endpoint case.

## Policy extraction

HIQL's high-level actor predicts a subgoal representation. Its low-level actor
predicts actions conditioned on that representation. Both use the original AWR
weight `min(exp(alpha * advantage), 100)`.

The rank margin changes the numerical scale available to these value differences.
It should therefore be considered together with `high_alpha` and `low_alpha`.
Increasing ranking accuracy alone does not establish a better policy.

## Optional non-default regularizer

The existing source includes an optional upper-gap hinge:

```text
upper_margin = rank_upper_margin_ratio * rank_margin
L_upper = sum_{h,b} mask[b] * relu(gap[h,b] - upper_margin)
          / max(sum_b mask[b], 1)
L_value = L_rank + rank_upper_weight * L_upper
```

`rank_upper_weight=0.0` disables this branch and preserves the original single
hinge objective. A positive weight enables a different objective. The upper
margin ratio must lie strictly between zero and one. This optional branch is
preserved from the existing implementation; no associated experiments or outcome
claims are included in this repository.

## Tests and evidence boundary

The synthetic tests cover trajectory boundaries and `j=k`, gradient directions
at both endpoints, shift invariance, masked rows, absence of Bellman-target
dependence, unchanged target parameters after rank updates, shared augmentation,
configuration validation, the optional upper hinge, and TD/actor parity with the
pinned upstream implementation.

These are implementation checks, not a benchmark. No datasets, measured returns,
success rates, checkpoints, learning curves, or experimental records are bundled.
An actual CUDA training run and its environment-specific evaluation remain
separate from CPU unit-test results.
