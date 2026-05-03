"""Sparse core attention: selection, masking and the full-attention limit."""

import mlx.core as mx
import numpy as np
import pytest

from indexcache.attention import SparseAttention, causal_validity, resolve_indices
from indexcache.indexer import MASKED


def _qkv(b, h, L, d, seed=0):
    rng = np.random.default_rng(seed)
    return tuple(
        mx.array(rng.normal(size=(b, h, L, d)).astype(np.float32)) for _ in range(3)
    )


@pytest.mark.parametrize("L", [4, 8, 16])
@pytest.mark.parametrize("k", [16, 20])
def test_k_at_least_L_equals_full_causal_attention(L, k):
    """The headline invariant: with k >= L the sparse path is exact."""
    q, key, v = _qkv(1, 2, L, 4, seed=L)
    att = SparseAttention()
    full, _ = att(q, key, v, None)
    idx = resolve_indices(mx.zeros((1, L, L)), k)
    sparse, _ = att(q, key, v, idx)
    mx.eval(full, sparse)
    np.testing.assert_allclose(np.array(sparse), np.array(full), atol=1e-4)


@pytest.mark.parametrize("window", [2, 3])
def test_k_at_least_L_equals_full_attention_with_a_window(window):
    L = 12
    q, key, v = _qkv(1, 2, L, 4, seed=window)
    att = SparseAttention(window=window)
    full, _ = att(q, key, v, None)
    idx = resolve_indices(mx.zeros((1, L, L)), L, window=window)
    sparse, _ = att(q, key, v, idx)
    mx.eval(full, sparse)
    np.testing.assert_allclose(np.array(sparse), np.array(full), atol=1e-4)


def test_duplicate_slots_do_not_double_count_a_key():
    """A repeated key in the shortlist is masked, not weighted twice."""
    L = 6
    q, key, v = _qkv(1, 1, L, 4, seed=7)
    att = SparseAttention()
    distinct = resolve_indices(mx.zeros((1, L, L)), L)
    padded = mx.concatenate([distinct[..., :1], distinct[..., :1], distinct[..., 1:]], axis=-1)
    a, _ = att(q, key, v, distinct)
    b, _ = att(q, key, v, padded)
    mx.eval(a, b)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5)


def test_selected_top1_matches_the_indexer_argmax():
    """k=1 must return the argmax over the query's allowed keys."""
    rng = np.random.default_rng(5)
    L = 12
    scores_np = rng.normal(size=(2, L, L)).astype(np.float32)
    idx = np.array(resolve_indices(mx.array(scores_np), 1))
    for b in range(2):
        for t in range(L):
            allowed = list(range(t + 1))
            order = sorted(allowed, key=lambda j: (-scores_np[b, t, j], j))
            assert idx[b, t, 0] == order[0], (b, t)


def test_selection_never_leaves_the_causal_window():
    L = 16
    idx = np.array(resolve_indices(mx.zeros((1, L, L)), 5, window=4))
    for t in range(L):
        row = idx[0, t]
        assert row.min() >= max(0, t - 4 + 1)
        assert row.max() <= t


def test_topk_rows_are_unique_up_to_padding():
    L = 10
    idx = np.array(resolve_indices(mx.zeros((2, L, L)), 4))
    for b in range(2):
        for t in range(L):
            row = idx[b, t].tolist()
            distinct = min(4, t + 1)
            assert len(set(row[:distinct])) == distinct
            assert set(row[:distinct]) <= set(range(t + 1))
            # the padding repeats the largest selected key
            assert all(x == row[distinct - 1] for x in row[distinct:])


def test_resolve_indices_rejects_a_non_square_score_matrix():
    with pytest.raises(ValueError):
        resolve_indices(mx.zeros((1, 3, 5)), 2)


def test_causal_validity_matches_the_mask_used_by_attention():
    L = 8
    for t in range(L):
        flags = np.array(causal_validity(t, L))
        assert flags[: t + 1].all() and not flags[t + 1 :].any()
        win = np.array(causal_validity(t, L, window=3))
        assert win[max(0, t - 2) : t + 1].all()
        assert not win[: max(0, t - 2)].any()


def test_fully_masked_row_yields_a_uniform_distribution_not_nan():
    """A row whose every selected key is unavailable must not produce NaN."""
    L = 4
    q, key, v = _qkv(1, 1, L, 4, seed=9)
    # Every query is handed one future key plus repeats of it, so no slot in
    # any row is causally available.
    idx = mx.broadcast_to(mx.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]]]), (1, L, L))
    out, weights = SparseAttention()(q, key, v, idx)
    mx.eval(out, weights)
    assert bool(mx.all(mx.isfinite(out)).item())
    np.testing.assert_allclose(np.array(weights)[0, 0, 0], 1.0 / L, atol=1e-6)


def test_uniform_scores_select_the_lowest_positions():
    idx = np.array(resolve_indices(mx.zeros((1, 6, 6)), 3))
    assert idx[0, 3].tolist() == [0, 1, 2]  # all scores tie, so lowest indices win
    assert idx[0, 0].tolist() == [0, 0, 0]
    assert idx[0, 5].tolist() == [0, 1, 2]


def test_non_uniform_scores_change_the_selection():
    scores = mx.zeros((1, 5, 5))
    scores = scores + mx.array(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 9.0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 dtype=np.float32)
    )
    idx = np.array(resolve_indices(scores, 2))
    # row 2 can only see keys 0,1,2; the 9.0 sits at key 3, which is masked out,
    # so the two lowest allowed keys win.
    assert idx[0, 2].tolist() == [0, 1]
