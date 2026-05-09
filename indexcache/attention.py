"""Sparse core attention over a selected set of keys.

DeepSeek Sparse Attention replaces the quadratic core attention with attention
over the top-k keys chosen by the lightning indexer.  This module implements
that, plus an optional sliding-window ("local") part.  The paper does not say
whether its DSA model keeps a local window, so the default here is ``window=-1``
(no local part); the experiment turns a small window on explicitly and the
effect is reported in the README.
"""

from __future__ import annotations

import mlx.core as mx

from .indexer import MASKED

# Finite stand-in for "this key is masked" in a softmax.  Attention logits are
# bounded by a few hundred, so -1e9 underflows exp() to exactly 0 while staying
# finite; feeding the -1e30 ranking sentinel here would make log_softmax return
# -inf (and then NaN through the LM cross-entropy).
NEG_LOGIT = -1e9


def time_column(L: int, ndim: int) -> mx.array:
    """(1, ..., L, 1) int32 query positions, broadcastable against (..., L, L)."""
    return mx.arange(L, dtype=mx.int32).reshape((1,) * (ndim - 2) + (L, 1))


def key_row(L: int, ndim: int) -> mx.array:
    """(1, ..., 1, L) int32 key positions.

    Note the difference from :func:`time_column`: broadcasting the *column* to
    (..., L, L) makes every row constant, so gathering key indices out of it
    would hand back the query position for every slot.
    """
    return mx.arange(L, dtype=mx.int32).reshape((1,) * (ndim - 2) + (1, L))


def causal_validity(t: int, L: int, *, window: int = -1) -> mx.array:
    """(L,) boolean: which keys position ``t`` may see.

    Causal (``j <= t``), and additionally ``j >= t - window + 1`` when window > 0.
    """
    j = mx.arange(L)
    valid = j <= t
    if window and window > 0:
        valid = valid & (j > t - window)
    return valid


def _safe_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Softmax that returns a uniform row instead of NaN for an all-masked row."""
    row_max = mx.max(x, axis=axis, keepdims=True)
    is_empty = row_max <= NEG_LOGIT / 2
    x = mx.where(is_empty, mx.zeros_like(x), x)
    e = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    p = e / mx.sum(e, axis=axis, keepdims=True)
    uniform = mx.full(x.shape, 1.0 / x.shape[axis], dtype=x.dtype)
    return mx.where(is_empty, uniform, p)


def shortlist_stats(idx: mx.array) -> mx.array:
    """1 for the first occurrence of each key in a shortlist row, 0 for repeats.

    Equivalent to ``first_occurrence[i] = min{j <= i : idx[j] == idx[i]}`` and
    returning ``first_occurrence == i``.  Written as a comparison against an
    index mask rather than as an ``argmin``/``min`` reduction over a broadcast
    of the index vector: MLX 0.32.3 right-aligns that broadcast, so the naive
    form compares the key axis with itself and marks every slot as a first
    occurrence.
    """
    n = idx.shape[-1]
    earlier = mx.arange(n, dtype=mx.int32).reshape((1,) * (idx.ndim - 1) + (1, n)) < (
        mx.arange(n, dtype=mx.int32).reshape((1,) * (idx.ndim - 1) + (n, 1))
    )
    same = idx[..., :, None] == idx[..., None, :]
    seen_before = mx.sum((same & earlier).astype(mx.int32), axis=-1)
    return (seen_before == 0).astype(mx.int32)


def select_shortlist(
    scores: mx.array, allowed: mx.array, kk: int, *, force_diagonal: bool = False
) -> mx.array:
    """Top-``kk`` keys per query row, strongest first; ``(..., L, kk)``.

    Ordering is a stable argsort over the raw scores: with ``force_diagonal``
    the query's own key leads, then the allowed keys by descending score with
    the lower position winning exact ties, then the masked positions.  Slots
    past the number of real candidates repeat the smallest selected key, so no
    row is empty and the padding adds no key that was not selected.

    Ranking is deliberately *not* folded into a composite float key
    (``-score * span + index / span``).  For a masked ``-1e30`` score the
    tie-breaking offset would be ``1e-45``, and for ordinary scores of magnitude
    ~50 float32 cannot represent it at all, so the position tie-break would be
    silently lost.
    """
    L = scores.shape[-1]
    scores = mx.stop_gradient(scores)
    descending = mx.argsort(-scores, axis=-1)
    rank = mx.argsort(descending, axis=-1).astype(mx.int32)  # slot of each key
    rank = mx.where(allowed, rank, 2 * L)
    if force_diagonal:
        is_diag = key_row(L, scores.ndim) == time_column(L, scores.ndim)
        rank = mx.where(is_diag & allowed, -1, rank)
    strong = mx.argsort(rank, axis=-1)  # strongest candidate first
    rank_full = mx.take_along_axis(rank, strong, axis=-1)
    keys_base = mx.broadcast_to(key_row(L, scores.ndim), scores.shape)
    keys = mx.take_along_axis(keys_base, strong, axis=-1)
    valid = rank_full < 2 * L
    # Padding: repeat a real selection, using L as the "no candidate" sentinel
    # so that `min` never picks a -1 (which would then be taken modulo the key
    # range and turn the padding into keys that were never selected).
    last = mx.min(mx.where(valid, keys, L), axis=-1, keepdims=True)
    filler = mx.where(last >= L, 0, last)
    return mx.where(valid, keys, filler)


def resolve_indices(
    scores: mx.array,
    k: int,
    *,
    window: int = -1,
    force_diagonal: bool = False,
) -> mx.array:
    """Top-k key indices per query position; ``(..., L, min(k, L))``.

    The distinct entries of row ``t`` are exactly its ``min(k, allowed)``
    highest-scoring allowed keys (the diagonal first when ``force_diagonal``),
    so the index *set* is canonical and the order is deterministic.  Repeats of
    the last distinct key fill the row's remaining slots.
    """
    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1]:
        raise ValueError(f"scores must be (..., L, L), got {scores.shape}")
    L = scores.shape[-1]
    # Selection is discrete: stop gradients at the entrance of this function so
    # that the argsort below never reaches MLX's autodiff (which raises
    # "Cannot calculate VJP with respect to indices" for the gather it lowers
    # to).  Nothing downstream wants d(indices)/d(weights) anyway.
    scores = mx.stop_gradient(scores)
    time = time_column(L, scores.ndim)
    j = mx.arange(L).reshape((1,) * (scores.ndim - 1) + (L,))
    if window is not None and window > 0:
        allowed = (j <= time) & (j > time - window)
    else:
        allowed = j <= time
    scores = mx.where(allowed, scores, mx.array(MASKED, dtype=scores.dtype))

    kk = min(int(k), L)
    if int(k) >= L:
        # Every allowed key is selected, in position order: the sparse path
        # then sees exactly the key set full causal attention would.  Only
        # ``kk`` of them fit when the window restricts the row to fewer keys.
        full = select_shortlist(scores, allowed, L, force_diagonal=False)
        slot = mx.arange(L, dtype=mx.int32).reshape((1,) * (scores.ndim - 1) + (L,))
        rank_ok = mx.sum(allowed.astype(mx.int32), axis=-1, keepdims=True) > slot
        keep = mx.argsort((1 - rank_ok.astype(mx.int32)) * L + full, axis=-1)
        kept = mx.take_along_axis(full, keep, axis=-1)[..., :kk]
        ok = mx.take_along_axis(rank_ok, keep, axis=-1)[..., :kk]
        filler = mx.max(mx.where(ok, kept, -1), axis=-1, keepdims=True)
        return mx.where(ok, kept, mx.maximum(filler, 0))[..., :kk]

    # Rank every key: the allowed ones by descending score (lower position wins
    # exact ties), the masked ones last.  Identical keys are exactly the
    # allowed keys, so no de-duplication step is needed; the shortlist below is
    # distinct by construction and any hole is padding.
    descending = mx.argsort(-scores, axis=-1)
    rank = mx.argsort(descending, axis=-1).astype(mx.int32)
    rank = mx.where(allowed, rank, 2 * L)
    if force_diagonal:
        is_diag = key_row(L, scores.ndim) == time
        rank = mx.where(is_diag & allowed, -1, rank)
    strong = mx.argsort(rank, axis=-1)  # (..., L, L) keys, best first
    rank_full = mx.take_along_axis(rank, strong, axis=-1)
    keys = mx.take_along_axis(
        mx.broadcast_to(key_row(L, scores.ndim), scores.shape), strong, axis=-1
    )
    slot = mx.arange(L, dtype=mx.int32).reshape((1,) * (scores.ndim - 1) + (L,))
    okay = (rank_full < 2 * L) & (slot < kk)
    idx = mx.where(okay, keys, 0)
    # Present the survivors ascending; repeats of the strongest one fill the
    # remaining slots, so a short row never introduces an unchosen key.
    order = mx.argsort((1 - okay.astype(mx.int32)) * L + idx, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)[..., :kk]
    okay = mx.take_along_axis(okay, order, axis=-1)[..., :kk]
    filler = mx.max(mx.where(okay, idx, -1), axis=-1, keepdims=True)
    return mx.where(okay, idx, mx.maximum(filler, 0))


def gather_selected(x: mx.array, idx: mx.array) -> mx.array:
    """Gather along the key axis: x (B, H, L, d), idx (B, L, kk) -> (B, H, L, kk, d).

    Gathers from one flat ``(B*H*L, d)`` table with an int32 index, so the
    backward pass scatters into that table instead of a broadcast ``L x L``
    buffer.  ``take_along_axis`` is avoided: under MLX 0.32.3 it attempts a VJP
    through its index argument and the backward pass raises.
    """
    b, h, L, d = x.shape
    kk = idx.shape[-1]
    # One flat (B*H*L, d) table plus an explicit int32 index, materialised with
    # mx.eval before use.  Both are needed under MLX 0.32.3: take_along_axis
    # raises during the backward pass ("Cannot calculate VJP with respect to
    # indices"), and gathering with a *lazy* index built from a broadcast has
    # been observed to read the wrong rows.
    table = x.reshape(b * h * L, d)
    # Row starts in the flat table.  The element at (b, h, q) sits at flat row
    # (b*H + h)*L + q, so its offset is that index times the row length -- not
    # the *position within the (b, h, L) grid* times the row length, which
    # over-counts by a factor of L and silently reads the wrong rows.
    rows = (mx.arange(b * h * L, dtype=mx.int32) // L) * L
    rows = rows.reshape(b, h, L, 1)
    flat_idx = (rows + idx.reshape(b, 1, L, kk)).reshape(-1)
    return table[flat_idx].reshape(b, h, L, kk, d)


class SparseAttention:
    """Causal attention over a per-query selected key set (+ optional window)."""

    def __init__(self, scale: float | None = None, window: int = -1):
        self.scale = scale
        self.window = window

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        selected: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """q/k/v: (B, H, L, d); selected: (B, L, kk) int32, or None for full.

        Returns ``(output, weights)``.  Weights cover the keys actually attended
        to: ``(B, H, L, L)`` for full attention, ``(B, H, L, kk)`` for the
        sparse path.
        """
        b, h, L, d = q.shape
        scale = self.scale if self.scale is not None else 1.0 / (d**0.5)
        t = mx.arange(L).reshape(L, 1)
        j = mx.arange(L).reshape(1, L)
        windowed = self.window is not None and self.window > 0

        if selected is None:
            scores = mx.matmul(q * scale, mx.swapaxes(k, -1, -2))  # (B,H,L,L)
            valid = j <= t
            if windowed:
                valid = valid & (j > t - self.window)
            scores = mx.where(valid, scores, mx.array(NEG_LOGIT, dtype=scores.dtype))
            weights = _safe_softmax(scores, axis=-1)
            return mx.matmul(weights, v), weights

        kk = selected.shape[-1]
        ks = gather_selected(k, selected)  # (B,H,L,kk,d)
        vs = gather_selected(v, selected)
        t_sel = t.reshape(1, L, 1)
        # Same lower bound as the full path, ``t - window + 1``, with an
        # infinite window: plain causality.  Writing ``t - window`` with
        # window = -1 would be off by one and mask the diagonal out.
        valid = selected > (t_sel - self.window) if windowed else selected <= t_sel
        # A shortlist pads its tail with repeats of a key it already selected.
        # Those extra slots must be masked out rather than given the same
        # softmax weight a second time, otherwise a key that the top-k chose
        # twice would outvote the rest.
        duplicate = shortlist_stats(selected) < 1
        valid = valid & mx.logical_not(duplicate)
        valid = mx.broadcast_to(valid[:, None], (b, h, L, kk))
        scores = mx.sum(q[:, :, :, None, :] * ks, axis=-1) * scale  # (B,H,L,kk)
        scores = mx.where(valid, scores, mx.array(NEG_LOGIT, dtype=scores.dtype))
        weights = _safe_softmax(scores, axis=-1)
        return mx.sum(weights[..., None] * vs, axis=-2), weights
