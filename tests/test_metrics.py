"""Overlap and cost metrics."""

import mlx.core as mx
import numpy as np
import pytest

from indexcache.metrics import (
    filter_queries,
    indexer_index_cost,
    jaccard_from_lists,
    overlap_ratio,
    pairwise_jaccard,
)
from indexcache.pattern import LayerPattern


def test_jaccard_of_identical_sets_is_one():
    a = mx.array(np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.int32))
    out = pairwise_jaccard({0: a, 1: a})
    assert out["0-1"] == pytest.approx(1.0)
    assert out["mean_adjacent"] == pytest.approx(1.0)


def test_jaccard_of_disjoint_sets_is_zero():
    a = mx.array(np.array([[[0, 1], [2, 3]]], dtype=np.int32))
    b = mx.array(np.array([[[4, 5], [6, 7]]], dtype=np.int32))
    out = pairwise_jaccard({0: a, 1: b})
    assert out["0-1"] == pytest.approx(0.0)


def test_jaccard_is_symmetric_and_matches_a_hand_value():
    a = mx.array(np.array([[[0, 1, 2, 3]]], dtype=np.int32))  # {0,1,2,3}
    b = mx.array(np.array([[[2, 3, 4, 5]]], dtype=np.int32))  # {2,3,4,5}
    out = pairwise_jaccard({0: a, 1: b})
    # intersection 2, union 6
    assert out["0-1"] == pytest.approx(2 / 6)
    assert pairwise_jaccard({0: b, 1: a})["0-1"] == pytest.approx(2 / 6)


def test_overlap_ratio_divides_by_k_not_by_the_union():
    a = mx.array(np.array([[[0, 1, 2, 3]]], dtype=np.int32))
    b = mx.array(np.array([[[2, 3, 4, 5]]], dtype=np.int32))
    out = overlap_ratio({0: a, 1: b})
    assert out["0-1"] == pytest.approx(2 / 4)


def test_mean_adjacent_uses_consecutive_pairs_only():
    a = mx.array(np.array([[[0, 1]]], dtype=np.int32))
    b = mx.array(np.array([[[0, 1]]], dtype=np.int32))
    c = mx.array(np.array([[[9, 9]]], dtype=np.int32))
    out = pairwise_jaccard({0: a, 1: b, 2: c})
    assert out["0-1"] == pytest.approx(1.0)
    assert out["1-2"] == pytest.approx(0.0)
    assert out["mean_adjacent"] == pytest.approx(0.5)


def test_filter_queries_keeps_only_late_positions():
    idx = {0: mx.zeros((1, 10, 3), dtype=mx.int32)}
    filtered = filter_queries(idx, 4)
    assert filtered[0].shape == (1, 6, 3)


def test_index_cost_counts_score_entries_and_invocations():
    pattern = LayerPattern("FSFS")
    cost = indexer_index_cost(pattern, L=8)
    assert cost["indexer_invocations"] == 2
    # one Full layer scores L(L+1)/2 = 36 entries
    assert cost["indexer_score_entries"] == 72
    assert cost["all_full_score_entries"] == 144
    assert cost["entries_removed_fraction"] == pytest.approx(0.5)


def test_index_cost_of_all_full_removes_nothing():
    cost = indexer_index_cost(LayerPattern.all_full(3), L=4)
    assert cost["entries_removed_fraction"] == 0.0
    assert cost["indexer_score_entries"] == cost["all_full_score_entries"]


def test_jaccard_from_lists_handles_empty_sets():
    assert jaccard_from_lists([], []) == 1.0
    assert jaccard_from_lists([1, 2], []) == 0.0
    assert jaccard_from_lists([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
