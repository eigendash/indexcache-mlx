"""The lightning indexer scorer and its top-k helper."""

import mlx.core as mx
import numpy as np
import pytest

from indexcache.indexer import MASKED, DSAIndexer, indexer_scores, topk_indices


def test_scores_are_non_negative_because_of_the_relu():
    rng = np.random.default_rng(0)
    q = mx.array(rng.normal(size=(2, 3, 5, 4)).astype(np.float32))
    k = mx.array(rng.normal(size=(2, 3, 7, 4)).astype(np.float32))
    w = mx.array(np.array([1.0, 2.0, 0.5], dtype=np.float32))
    s = indexer_scores(q, k, w)
    mx.eval(s)
    assert s.shape == (2, 5, 7)
    assert float(mx.min(s).item()) >= 0.0


def test_negative_head_weight_contributes_nothing():
    """Gates are clipped at zero, so a negative gate cannot subtract score."""
    q = mx.array(np.ones((1, 1, 2, 2), dtype=np.float32))
    k = mx.array(np.ones((1, 1, 2, 2), dtype=np.float32))
    s_neg = indexer_scores(q, k, mx.array([-1.0]))
    s_pos = indexer_scores(q, k, mx.array([1.0]))
    mx.eval(s_neg, s_pos)
    assert np.allclose(np.array(s_neg), 0.0)
    assert np.all(np.array(s_pos) > 0.0)


def test_hand_computed_two_head_score():
    q = mx.array(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))  # (1,1,2,2)
    k = mx.array(np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32))
    w = mx.array([1.0])
    s = indexer_scores(q, k, w, scale=1.0)
    mx.eval(s)
    # row0: <[1,0],[1,0]> = 1, <[1,0],[1,1]> = 1
    # row1: <[0,1],[1,0]> = 0 (relu), <[0,1],[1,1]> = 1
    assert np.allclose(np.array(s), [[[1.0, 1.0], [0.0, 1.0]]])


def test_score_dtype_is_float32_and_head_axis_is_summed_out():
    idx = DSAIndexer(dim=8, n_heads=2, head_dim=4)
    x = mx.random.normal((2, 6, 8))
    s = idx.score(x)
    mx.eval(s)
    assert s.shape == (2, 6, 6)
    assert s.dtype == mx.float32


def test_topk_returns_exactly_k_unique_indices_in_range():
    scores = mx.array(np.random.default_rng(1).normal(size=(4, 20)).astype(np.float32))
    for k in (1, 5, 20, 25):
        got = np.array(topk_indices(scores, k))
        kk = min(k, 20)
        assert got.shape == (4, kk)
        for row in got:
            assert len(set(row.tolist())) == kk
            assert row.min() >= 0 and row.max() < 20


def test_topk_breaks_ties_by_lower_position():
    scores = mx.zeros((1, 6))
    got = np.array(topk_indices(scores, 3))
    assert got.tolist() == [[0, 1, 2]]


def test_topk_is_deterministic_across_calls():
    scores = mx.array(np.random.default_rng(2).normal(size=(3, 50)).astype(np.float32))
    a = np.array(topk_indices(scores, 7))
    b = np.array(topk_indices(scores, 7))
    assert np.array_equal(a, b)


def test_topk_matches_a_sorted_reference():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(5, 32)).astype(np.float32)
    got = np.array(topk_indices(mx.array(scores), 6))
    want = np.sort(np.argsort(-scores, axis=-1, kind="stable")[:, :6], axis=-1)
    assert np.array_equal(got, want)


def test_masked_entries_are_never_selected():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(2, 30)).astype(np.float32)
    scores[:, 10:] = MASKED
    got = np.array(topk_indices(mx.array(scores), 4))
    assert np.all(got < 10)


def test_indexer_scores_rejects_mismatched_heads():
    q = mx.zeros((1, 2, 3, 4))
    k = mx.zeros((1, 3, 3, 4))
    with pytest.raises(ValueError):
        indexer_scores(q, k, mx.array([1.0, 1.0]))
