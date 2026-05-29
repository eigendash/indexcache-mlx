# indexcache-mlx

A small independent MLX implementation of *IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse* (arXiv:2603.12201), plus a toy decoder-only transformer trained on a synthetic associative-recall task to try the mechanism out. It runs on Apple silicon CPU/GPU.

The paper starts from DeepSeek Sparse Attention (DSA): a cheap "lightning indexer" scores every query against every preceding key and the core attention runs over only the top-k of them, so core attention is O(Lk) but the indexer itself is still O(L^2) and runs at every layer. IndexCache observes that the top-k sets selected at consecutive layers overlap heavily, so it partitions the layers into a few **Full** layers that keep their indexers and a majority of **Shared** layers that simply reuse the nearest preceding Full layer's index set. Two ways to choose and optimise that partition are proposed: a training-free greedy search over which layers keep indexers, scored by language-modelling loss on a calibration set, and a training-aware multi-layer distillation loss that trains each retained indexer against the average attention distribution of all the layers it serves. On a 30B DSA model the paper reports removing 75% of indexer computations with negligible quality loss (1.82x prefill, 1.48x decode at 200K context).

This is not a reproduction. It is a few hundred thousand parameters on a synthetic task, it measures nothing about the paper's 30B model or its benchmark numbers, and its numbers are not comparable to the paper's. What it does do is implement the mechanism end to end and report what happens at toy scale, including one result that contradicts the paper's.

## What is in here

**The lightning indexer** (`indexcache/indexer.py`). The paper describes selection only as "a multi-head ReLU-gated dot product" and does not give the projections, so the reading here is: per head `h`, project the hidden state with learned low-rank `q_proj_h` and `k_proj_h`, score `I_h(t, s) = ReLU(<q_h(t), k_h(s)> / sqrt(d_h))`, and sum the heads with non-negative per-head gates, `I(t, s) = sum_h w_h I_h(t, s)`, `w_h >= 0`. `indexer_scores` does this for arbitrary leading batch/head dimensions and `DSAIndexer` wraps it for the model.

**Selection** (`indexcache/attention.py`). `resolve_indices` turns a score matrix into a top-k index set per query token. It ranks the *allowed* candidates explicitly (causal, and windowed when a window is configured) with a stable argsort over the raw scores, which makes ties resolve to the lower position. The sorting is deliberately not folded into one composite float key: with `-score * span + position / span`, a masked `-1e30` score leaves a tie-breaking offset of `1e-45`, and for ordinary scores of magnitude ~50 float32 cannot represent the offset at all, so position tie-breaking silently disappears. Rows that can see fewer than k keys repeat their last selected key as padding, which keeps every row non-empty without introducing a key that was not selected.

**Sparse core attention** (`indexcache/attention.py`). `SparseAttention` takes the gathered keys and values for the shortlist, masks slots that are causally unavailable (or outside the sliding window) and duplicate padding slots, softmaxes over the remaining ones and writes the result back. With no selection it is ordinary causal attention. The paper does not say whether its DSA model keeps a local window, so the default here is `window=-1` (none) and the experiment runs without one.

**Full/Shared partition** (`indexcache/pattern.py`). `LayerPattern` holds the paper's binary pattern string, resolves each Shared layer to `f(l) = max{j < l : c_j = F}`, and can enumerate the layers a given Full layer serves. Layer 0 is always Full; a pattern that starts with `S` is representable but raises when asked to resolve.

**Model** (`indexcache/model.py`). A decoder-only transformer with RMSNorm, a GELU MLP, learned positional embeddings (not the RoPE the paper's model uses) and one DSA attention layer per block. `forward(tokens, pattern)` walks the layers, calls the indexer exactly once per Full layer, caches the resulting index set and hands it to every Shared layer until the next Full one, and hands the same scores to the attention layer so a layer never scores twice. `training_loss` is the language-modelling cross-entropy at every position (or, for this toy task, at the answer position) plus an optional multi-layer distillation term.

**Multi-layer distillation** (`indexcache/distill.py`). The objective is the paper's Eq. 1, `sum_j 1/(m+1) * sum_t KL(p_t^{(l+j)} || q_t^{(l)})`, with `p` the head-averaged attention distribution of a served layer and `q` the indexer's softmax. The paper's Proposition 1 says the gradient equals that of distilling against the averaged target `p_bar`; `mode="averaged"` implements that form and `tests/test_distill.py` checks the two gradients agree numerically. Subtracting the target's own entropy leaves the gradient unchanged and makes the value exactly zero at `p == q`. In the sparse phase the KL is restricted to the selected top-k and both the target and the model distribution are renormalised over that support. By default the value is divided by `log(S)` so it stays in [0, 1] and can be added to a language-modelling loss without a rescaling that depends on the vocabulary size.

**Greedy layer selection** (`indexcache/greedy.py`). Algorithm 1: start all-Full, then repeatedly flip the remaining Full layer (layer 0 excluded) to Shared that costs the least LM loss on a fixed cached calibration set, and commit that flip. `n_shared=0` is a no-op that never calls the evaluator.

**Task, metrics, experiment** (`indexcache/task.py`, `indexcache/metrics.py`, `scripts/train_toy.py`). The task is multi-key associative recall: `n_pairs` key/value bindings shuffled into the context, then a query key, and the model must emit the value bound to it at the final position. Keys and values come from disjoint vocabulary ranges and each answer value appears exactly once, so the answer cannot be copied from nearby. `metrics.py` computes the mean Jaccard similarity `|A n B| / |A u B|` between consecutive layers' top-k sets (the measure the paper's premise is about) and the paper's own `|A n B| / k` ratio, plus indexer cost in units of score entries.

## Checks

`pytest` runs 85 tests. The ones that matter most:

- **The sparse path is exact when it should be.** With `k >= L` and a full shortlist, sparse attention equals full causal attention to better than `1e-6` (measured `2.4e-7`), both with and without a sliding window (`test_attention.py`). This is the invariant that catches most selection bugs; during development it caught a gather that read the wrong rows because the flat row offset was `position * L` instead of `row * L`.
- **Selection is the real top-k.** `resolve_indices` is compared against a NumPy reference over many `(L, k, window)` combinations and both diagonal modes (`tests/test_attention.py`); rows are ascending, distinct up to padding, inside the causal window, and the `k=1` case returns the argmax over the query's allowed keys.
- **Shared layers reuse exactly the source layer's set.** A spy on the block call checks that each layer attends to the index set of `source_for(layer)`, bit for bit, while the model-level test checks that the number of indexer invocations equals the number of Full layers, and that with `FSFS` the Shared layers' indexers receive zero gradient.
- **Distillation.** The loss is zero when the model distribution already equals the target; the per-layer and averaged-target forms have equal gradients (Proposition 1) to `1e-4`; restricting to the top-k renormalises the target; gradients are finite through the whole model.
- **Greedy search.** On a synthetic per-layer cost surface it removes the cheapest layer first and never touches layer 0; with `n_shared=0` it returns all-Full without evaluating anything.
- **The task.** Generated batches satisfy the properties the results depend on: one pair per key, the query key present exactly twice, an answer value that appears exactly once, and `answer_loss`/`answer_accuracy` computed at the final position only.

## Experiment

`scripts/train_toy.py` trains the same 4-layer, 64-wide model (about 255k parameters, `top_k=32` of 128, no sliding window) on 1000 generated examples with 4 keys drawn from a 128-key vocabulary, and evaluates on 200 held-out examples. Training is 800 AdamW steps at lr 3e-3, batch 16, for three seeds per configuration, and the table reports mean +- std over those seeds. The all-Full, 1-in-2 and 1-in-4 rows share one trained model per seed; `greedy_selected` and the training-free rows reuse the all-Full weights with a different pattern; the `trained_*` rows are trained from scratch under their own pattern.

| configuration | pattern | held-out loss | answer accuracy | indexers removed |
|---|---|---:|---:|---:|
| all-Full (DSA baseline) | `FFFF` | 0.6343 +- 0.1727 | 0.820 +- 0.043 | 0% |
| training-free 1-in-2 | `FSFS` | 0.6312 +- 0.1743 | 0.820 +- 0.051 | 50% |
| training-free 1-in-4 | `FSSS` | 0.6344 +- 0.1753 | 0.810 +- 0.053 | 75% |
| training-free greedy | `FSSS` | 0.6344 +- 0.1753 | 0.810 +- 0.053 | 75% |
| training-aware 1-in-2 | `FSFS` | 1.1352 +- 0.3787 | 0.637 +- 0.158 | 50% |
| training-aware 1-in-4 | `FSSS` | 1.8652 +- 0.3580 | 0.425 +- 0.085 | 75% |

Cross-layer top-32 overlap of the all-Full model's selections, averaged over adjacent layer pairs and three seeds:

| measure | all queries | queries with >= 16 candidates |
|---|---:|---:|
| mean Jaccard `|A n B| / |A u B|` | 0.485 +- 0.002 | 0.411 +- 0.002 |
| mean `|A n B| / k` (the paper's ratio) | 0.578 +- 0.004 | 0.518 +- 0.004 |

Measured prefill time for one teacher-forced forward pass over 16 x 128 tokens, mean of five calls after two warm-ups (no fused kernel here, so these are wall-clock numbers from this machine, not a speedup claim anywhere near the paper's):

| pattern | indexer invocations | prefill (ms) |
|---|---:|---:|
| `FFFF` | 4 / 4 | 5.79 |
| `FSFS` | 2 / 4 | 5.15 |
| `FSSS` | 1 / 4 | 4.80 |

What the numbers mean:

- **The paper's central claim holds at this scale.** Dropping three of four indexers changes held-out loss from 0.6343 to 0.6344 and accuracy from 0.820 to 0.810, well inside the seed-to-seed spread. The indexer is genuinely redundant here.
- **But the premise is much weaker than the paper reports.** Adjacent layers share about 58% of their selected keys by the paper's `|A n B| / k`, or 48% by Jaccard, against the paper's 0.7-1.0. The gap is not noise: the per-seed numbers are within +-0.002 of each other. See the next section for why I think this is what it is and what it is not evidence of.
- **Greedy selection does not beat uniform interleaving here.** Across all three seeds the greedy search removed layers 1, 2 and 3 in some order and converged on `FSSS` for all three, so it returned exactly the uniform 1-in-4 pattern. This is the same negative result the paper reports for similarity-based search in its Appendix C: local metrics do not identify the best partition, and with only four layers there is not much partition to search.
- **The training-aware variant is worse than the baseline, not better.** This contradicts the paper's training-aware result and I am not going to dress it up. The gradient-equivalence property of the objective is tested and holds, the loss is normalised, and the ablation (below) says no distillation weight I tried helps. At this scale the objective is simply a worse thing to optimise than the task loss.
- Only the indexer count and forward time are measured. Decode is not timed, there are no kernels, and the "speedup" between 5.79 ms and 4.80 ms is a small absolute difference on a tiny model.

The distillation weight was ablated at 0.0, 0.001, 0.003, 0.01 and 0.03 (two seeds, 600 steps, `results/ablation.json`). The no-distillation control won at every setting: for the 1-in-2 pattern the mean loss / accuracy was 1.2730 / 0.645 at weight 0.0 against 2.0005 / 0.368 at 0.001, 1.6759 / 0.445 at 0.003, 2.1021 / 0.343 at 0.01 and 2.1851 / 0.305 at 0.03; for 1-in-4, 0.8920 / 0.738 at 0.0 against 1.6068 / 0.473, 1.9402 / 0.358, 2.4954 / 0.248 and 2.1069 / 0.333. The weight used in the table (0.001) is not a tuned win; it is simply the least-bad non-zero setting. The ablation ran 600 steps rather than the table's 800, so its absolute values differ from the table's `trained_*` rows; both agree that the no-distillation control is best.

## Where this may differ from the paper

- **The indexer architecture is my reading.** The paper gives one sentence for the lightning indexer and no dimensions or projections beyond "multi-head ReLU-gated dot product". Everything about `q_proj`/`k_proj`/head gates/head count is inferred. The paper's real indexer is low-rank and FP8; this one is a two-head, 32-dimensional projection in float32.
- **No MLA.** The paper's DSA is built on multi-head latent attention; this model uses ordinary multi-head attention with `d_model=64`. The indexer therefore has no latent KV to score.
- **No local/streaming window by default.** The paper does not say whether its DSA layers keep one. `SparseAttention(window=w)` supports it and the equality test covers it, but the experiment leaves it off so that the only thing varying between configurations is the pattern.
- **Positional embeddings, not RoPE.** The paper's base model uses RoPE; this uses learned positional embeddings, which cannot extrapolate past `max_seq_len`.
- **The objective trained here is the answer-position loss**, not the full-sequence next-token loss. I started with the full-sequence LM loss and the toy model would not learn the retrieval task through it (the padding dominates the average, and the model fit the training set without generalising). The results file records both, and `training_loss(full_sequence=True)` remains available.
- **The greedy search searches over 3 candidate layers.** The paper's model has 47 layers, where the ordering of indexer importance is interesting. With four layers there are three flips to consider and the search cannot show much.
- **The distillation weight, normalisation and schedule are mine.** The paper trains in two stages (dense warm-up, then sparse training) with the KL computed on a detached graph; here the distillation term is added to the task loss from step 0 with weight 0.001, the indexer is not separately warmed up, and the target distributions come from the same forward pass as the model. I did not implement the paper's two-stage pipeline.
- **Overlap is measured on 200 held-out sequences at `k=32` of `L=128`.** The paper measures on 768 samples of 200K context with `k=2048`. The paper's high overlap may well be a long-context phenomenon (its own heatmap shows early/late layers with overlap <= 0.4 and mid-stack clusters), and a 128-token context with a 128-key vocabulary is a very different regime.
- **The paper's numbers are not reproduced anywhere here.** No 30B model, no GLM-5, no MRCR/GraphWalks/LongBench/RULER/AA-LCR, no 1.82x prefill claim.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/train_toy.py
```

The experiment takes a few minutes and rewrites `results/experiment.json` and `results/run_log.txt`. Useful flags: `--seeds`, `--steps`, `--top-k`, `--n-layers`, `--seq-len`, `--n-pairs`, `--distill-weight`, `--overlap-min-candidates`.
