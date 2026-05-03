"""The lightning indexer: a cheap multi-head ReLU-gated dot-product scorer.

This follows the one sentence the paper gives for the indexer ("a lightweight
lightning indexer first scores all preceding tokens against the current query
using a multi-head ReLU-gated dot product"), so the exact projections are my
reading rather than the paper's.  See the README section "Where this may
differ from the paper".
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Finite stand-in for "cannot be selected".  -inf would poison the composite
# sort key below (masked entries would sort first) and then reach the softmax
# as a -inf logit, so masked scores are pushed to -1e30 instead.
MASKED = -1e30


def indexer_scores(
    q: mx.array,
    k: mx.array,
    head_weights: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    """Multi-head ReLU-gated indexer score.

    Parameters
    ----------
    q : (..., H, Tq, d) indexer queries
    k : (..., H, L, d)  indexer keys
    head_weights : (H,)  non-negative per-head gates (softplus is applied by
        the caller; here they are used as given)

    Returns
    -------
    (..., Tq, L) score = sum_h w_h * ReLU(<q_h, k_h> * scale)
    """
    if q.shape[:-2] != k.shape[:-2]:
        raise ValueError(f"head mismatch: q {q.shape} vs k {k.shape}")
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d**0.5)
    logits = mx.matmul(q * scale, mx.swapaxes(k, -1, -2))  # (..., H, Tq, L)
    gated = mx.maximum(logits, 0.0)
    # Gating is non-negative: a negative head weight must not subtract score.
    w = mx.maximum(head_weights, 0.0)
    w = w.reshape((1,) * (gated.ndim - 3) + (head_weights.shape[0], 1, 1))
    return mx.sum(gated * w, axis=-3)


def topk_indices(scores: mx.array, k: int, *, sorted_indices: bool = True) -> mx.array:
    """Deterministic top-k over the last axis.

    ``mx.topk`` returns a flat index vector; the score-only form is used here and
    the result is re-sorted, so ties are broken by the token position and the
    output is canonical (ascending) rather than dependent on the sort algorithm.
    """
    L = scores.shape[-1]
    kk = min(int(k), int(L))
    if kk < L:
        # mx.topk has no stable option and this must be deterministic, so:
        # partition, then sort the shortlist by (position) and finally by
        # (-score, position) using the composite-key trick for the secondary
        # key.  Ties therefore always resolve to the lower token position.
        candidates = mx.argpartition(-scores, kth=kk - 1, axis=-1)[..., :kk]
        cand_scores = mx.take_along_axis(scores, candidates, axis=-1)
        pos = mx.argsort(candidates, axis=-1)
        flat_score = mx.take_along_axis(cand_scores, pos, axis=-1)
        flat_idx = mx.take_along_axis(candidates, pos, axis=-1)
        # composite key: shift scores so that the integer part is the negated
        # score ordering and the fractional part the position tie-break.
        span = float(kk) + 1.0
        key = -flat_score * span + flat_idx.astype(flat_score.dtype) / span
        order = mx.argsort(key, axis=-1)
        idx = mx.take_along_axis(flat_idx, order, axis=-1)
    else:
        idx = mx.broadcast_to(
            mx.arange(L, dtype=mx.int32), scores.shape[:-1] + (L,)
        )
    if sorted_indices:
        idx = mx.sort(idx, axis=-1)
    return idx.astype(mx.int32)


class DSAIndexer(nn.Module):
    """Low-rank multi-head scorer producing one relevance score per key."""

    def __init__(self, dim: int, n_heads: int = 2, head_dim: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.head_weight = mx.ones((n_heads,))

    def score(self, x: mx.array) -> mx.array:
        """x: (B, L, dim) -> (B, L, L) score of each query over all keys."""
        b, L, _ = x.shape
        q = self.q_proj(x).reshape(b, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, L, self.n_heads, self.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))  # (B, H, L, d)
        k = mx.transpose(k, (0, 2, 1, 3))
        w = mx.maximum(self.head_weight, 0.0)
        return indexer_scores(q, k, w)

    def query_logits(self, x: mx.array) -> mx.array:
        """The same score function, but with the query side left un-summed.

        Used for the distillation distribution, where we need d/dtheta of the
        score for the selected keys only.
        """
        b, L, _ = x.shape
        q = self.q_proj(x).reshape(b, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, L, self.n_heads, self.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        w = mx.maximum(self.head_weight, 0.0)
        logits = mx.matmul(q / (self.head_dim**0.5), mx.swapaxes(k, -1, -2))
        gated = mx.maximum(logits, 0.0)
        return mx.sum(gated * w.reshape((1, self.n_heads, 1, 1)), axis=1)  # (B, L, L)

    def __call__(self, x: mx.array) -> mx.array:
        return self.score(x)
