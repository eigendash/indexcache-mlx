"""Measurements the experiment reports: index overlap and indexer cost."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .pattern import LayerPattern


def filter_queries(indices: dict[int, mx.array], min_candidates: int) -> dict[int, mx.array]:
    """Keep only query positions that could choose from ``min_candidates`` keys.

    Early causal positions can only see a handful of keys, so their top-k is
    nearly forced and their overlap is high by construction.  Restricting to
    queries with enough candidates makes the reported overlap about selection
    behaviour rather than about the causal mask.
    """
    layers = sorted(indices)
    L = indices[layers[0]].shape[1]
    keep = np.arange(L) >= min_candidates
    out = {}
    for layer in layers:
        arr = np.asarray(indices[layer])
        out[layer] = mx.array(arr[:, keep, :])
    return out


def pairwise_jaccard(indices: dict[int, mx.array]) -> dict[str, float]:
    """|A n B| / |A u B| for consecutive layers, averaged over queries and batch.

    ``indices`` maps layer -> (B, L, kk) selected keys.  Returns the mean over
    adjacent pairs plus the full matrix keyed ``"i-j"``.
    """
    layers = sorted(indices)
    arrays = {}
    for layer in layers:
        mx.eval(indices[layer])
        arrays[layer] = np.asarray(indices[layer])

    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        # sets per (batch, query) via broadcasting
        eq = a[:, :, :, None] == b[:, :, None, :]
        inter = eq.any(axis=-1).sum(axis=-1).astype(np.float64)
        union = a.shape[-1] + b.shape[-1] - inter
        return float(np.mean(inter / union))

    out: dict[str, float] = {}
    for i, a in enumerate(layers):
        for b in layers[i + 1 :]:
            out[f"{a}-{b}"] = jaccard(arrays[a], arrays[b])
    adjacent = [out[f"{a}-{b}"] for a, b in zip(layers, layers[1:])]
    out["mean_adjacent"] = float(np.mean(adjacent)) if adjacent else float("nan")
    return out


def overlap_ratio(indices: dict[int, mx.array]) -> dict[str, float]:
    """|A n B| / k -- the ratio the paper reports (70-100% between neighbours)."""
    layers = sorted(indices)
    arrays = {layer: np.asarray(indices[layer]) for layer in layers}

    def ratio(a: np.ndarray, b: np.ndarray) -> float:
        eq = a[:, :, :, None] == b[:, :, None, :]
        inter = eq.any(axis=-1).sum(axis=-1).astype(np.float64)
        return float(np.mean(inter / a.shape[-1]))

    out = {}
    for i, a in enumerate(layers):
        for b in layers[i + 1 :]:
            out[f"{a}-{b}"] = ratio(arrays[a], arrays[b])
    adjacent = [out[f"{a}-{b}"] for a, b in zip(layers, layers[1:])]
    out["mean_adjacent"] = float(np.mean(adjacent)) if adjacent else float("nan")
    return out


def indexer_index_cost(pattern: LayerPattern, L: int) -> dict:
    """How many indexer score entries are computed, per token of context.

    The indexer scores every query against every preceding key, so one Full
    layer costs ``L(L+1)/2`` entries; a Shared layer costs none.  The paper's
    proxy for indexer cost is the number of indexers run, and this is the same
    measure in units of score entries rather than invocations.
    """
    per_layer = L * (L + 1) // 2
    return {
        "indexer_invocations": pattern.n_full,
        "indexer_score_entries": pattern.n_full * per_layer,
        "all_full_score_entries": len(pattern) * per_layer,
        "entries_removed_fraction": pattern.indexer_saving(),
    }


def jaccard_from_lists(a, b) -> float:
    """Plain |A n B| / |A u B| for two Python index lists (used by tests)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
