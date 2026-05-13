"""The training-free greedy layer search (Algorithm 1)."""

import mlx.core as mx
import numpy as np
import pytest

from indexcache.greedy import calibration_batches, greedy_layer_selection, order_of_removal
from indexcache.pattern import LayerPattern


def test_all_full_is_a_no_op_and_never_calls_the_evaluator():
    calls = []

    def evaluate(pattern):
        calls.append(pattern.pattern)
        return 0.0

    pattern, trace = greedy_layer_selection(evaluate, n_layers=4, n_shared=0)
    assert pattern.pattern == "FFFF"
    assert trace == []
    assert calls == []


def test_greedy_removes_the_layer_that_hurts_least():
    """A synthetic loss surface with known per-layer costs."""
    # removing layer 1 costs 0.1, layer 2 costs 0.9, layer 3 costs 0.5
    cost = {1: 0.1, 2: 0.9, 3: 0.5}

    def evaluate(pattern):
        return sum(cost[layer] for layer in cost if not pattern.is_full(layer))

    pattern, trace = greedy_layer_selection(evaluate, n_layers=4, n_shared=1)
    assert pattern.pattern == "FSFF"  # layer 1 gave up its indexer
    assert order_of_removal(trace) == [1]
    assert trace[0]["loss"] == pytest.approx(0.1)


def test_greedy_is_sequential_and_monotone_in_the_removal_cost():
    cost = {1: 0.3, 2: 0.1, 3: 0.7}

    def evaluate(pattern):
        return sum(cost[layer] for layer in cost if not pattern.is_full(layer))

    pattern, trace = greedy_layer_selection(evaluate, n_layers=4, n_shared=2)
    assert order_of_removal(trace) == [2, 1]
    assert pattern.pattern == "FSSF"  # layers 1 and 2 are Shared
    # the second step's loss includes the first removal, so it is the largest
    assert trace[1]["loss"] == pytest.approx(0.4)


def test_layer_zero_is_never_a_candidate():
    seen = []

    def evaluate(pattern):
        seen.append(pattern.pattern)
        return 0.0

    pattern, _ = greedy_layer_selection(evaluate, n_layers=5, n_shared=4)
    assert pattern.pattern == "FSSSS"
    assert all(p.startswith("F") for p in seen)


def test_greedy_cannot_share_more_than_n_minus_one_layers():
    with pytest.raises(ValueError):
        greedy_layer_selection(lambda p: 0.0, n_layers=3, n_shared=3)
    with pytest.raises(ValueError):
        greedy_layer_selection(lambda p: 0.0, n_layers=3, n_shared=-1)


def test_greedy_evaluates_every_remaining_candidate_at_each_step():
    n_layers = 5

    def evaluate(pattern):
        return float(len(pattern.shared_layers))

    _, trace = greedy_layer_selection(evaluate, n_layers=n_layers, n_shared=2)
    assert sorted(trace[0]["losses"]) == [1, 2, 3, 4]
    assert sorted(trace[1]["losses"]) == [2, 3, 4] or sorted(trace[1]["losses"]) == [1, 3, 4]


def test_calibration_batches_are_fixed_and_reused():
    tokens = mx.arange(0, 24 * 4, dtype=mx.int32).reshape(24, 4)
    batches = calibration_batches(tokens, batch_size=4, n_batches=3)
    assert len(batches) == 3
    for b in batches:
        mx.eval(b)
        assert b.shape[0] <= 4 and b.shape[1] == 4
    # calling twice yields identical data (same cache)
    again = calibration_batches(tokens, batch_size=4, n_batches=3)
    for a, b in zip(batches, again):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def test_calibration_batches_validate_input():
    with pytest.raises(ValueError):
        calibration_batches(mx.zeros((4, 4, 4), dtype=mx.int32), 2, 2)
    with pytest.raises(ValueError):
        calibration_batches(mx.zeros((4, 4), dtype=mx.int32), 2, 0)


def test_greedy_trace_holds_the_pattern_after_each_step():
    cost = {1: 3.0, 2: 1.0, 3: 2.0}

    def evaluate(pattern):
        return sum(cost[layer] for layer in cost if not pattern.is_full(layer))

    pattern, trace = greedy_layer_selection(evaluate, n_layers=4, n_shared=3)
    assert [step["pattern"] for step in trace] == ["FFSF", "FFSS", "FSSS"]
    assert pattern.pattern == trace[-1]["pattern"]
    assert LayerPattern(trace[0]["pattern"]).n_shared == 1
