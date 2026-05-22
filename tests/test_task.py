"""The synthetic associative-recall task."""

import mlx.core as mx
import numpy as np
import pytest

from indexcache.task import (
    ANSWER_POSITION,
    BOS,
    KEY_BASE,
    QUERY,
    VALUE_BASE,
    VOCAB_SIZE,
    TaskConfig,
    answer_accuracy,
    answer_loss,
    answer_tokens,
    build_task,
    is_answer_position_well_formed,
)


def test_generated_batches_are_well_formed():
    for seed in range(4):
        data = build_task(TaskConfig(n_pairs=6, seq_len=96, seed=seed))
        assert data.shape == (6, 96)
        assert data.dtype == np.int32
        assert is_answer_position_well_formed(data), seed
        assert all(0 <= t < VOCAB_SIZE for t in data.reshape(-1))


def test_the_body_holds_exactly_one_pair_per_key():
    data = build_task(TaskConfig(n_pairs=8, seq_len=128, seed=3))
    body = data[0, 1 : 1 + 16]
    # pairs are kept adjacent, so even offsets are keys and odd offsets values
    assert all(KEY_BASE <= body[i] < VALUE_BASE for i in range(0, 16, 2))
    assert all(VALUE_BASE <= body[i] < VOCAB_SIZE for i in range(1, 16, 2))


def test_the_query_key_is_present_exactly_twice():
    data = build_task(TaskConfig(n_pairs=5, seq_len=64, seed=1))
    qk = data[0, -2]
    assert int(np.sum(data[0] == qk)) == 2  # once in the body, once as the query


def test_answer_tokens_returns_the_value_column():
    data = build_task(TaskConfig(n_pairs=5, seq_len=64, seed=2))
    got = answer_tokens(data)
    assert got.shape == (5,)
    np.testing.assert_array_equal(got, data[:, ANSWER_POSITION])


def test_two_sequences_differ_between_seeds():
    a = build_task(TaskConfig(n_pairs=4, seq_len=48, seed=0))
    b = build_task(TaskConfig(n_pairs=4, seq_len=48, seed=1))
    assert not np.array_equal(a, b)


def test_seq_len_must_fit_the_pairs():
    with pytest.raises(ValueError):
        build_task(TaskConfig(n_pairs=8, seq_len=8, seed=0))


def test_answer_loss_is_cross_entropy_at_the_last_position():
    tokens = mx.array(np.array([[BOS, KEY_BASE, VALUE_BASE, QUERY, KEY_BASE + 1, VALUE_BASE + 2]], dtype=np.int32))
    # logits that put all mass on the correct answer token
    target = int(np.array(tokens)[0, ANSWER_POSITION])
    logits = mx.full((1, 6, VOCAB_SIZE), -1e4, dtype=mx.float32)
    logits = logits.at[0, ANSWER_POSITION, target].add(2e4)
    loss = answer_loss(logits, tokens)
    mx.eval(loss)
    assert float(loss.item()) < 1e-3

    # and a wrong answer gives a large loss
    wrong = mx.full((1, 6, VOCAB_SIZE), -1e4, dtype=mx.float32)
    wrong = wrong.at[0, ANSWER_POSITION, target + 1].add(2e4)
    loss_wrong = answer_loss(wrong, tokens)
    mx.eval(loss_wrong)
    assert float(loss_wrong.item()) > 1.0


def test_answer_accuracy_counts_exact_matches():
    tokens = mx.array(np.array([[BOS, KEY_BASE, VALUE_BASE, QUERY, KEY_BASE + 1, VALUE_BASE + 2]], dtype=np.int32))
    target = int(np.array(tokens)[0, ANSWER_POSITION])
    logits = mx.full((1, 6, VOCAB_SIZE), -1e4, dtype=mx.float32)
    logits = logits.at[0, ANSWER_POSITION, target].add(2e4)
    assert answer_accuracy(logits, tokens) == pytest.approx(1.0)
    logits = logits.at[0, ANSWER_POSITION, target].add(-3e4)
    assert answer_accuracy(logits, tokens) == pytest.approx(0.0)


def test_answer_loss_is_finite_and_differentiable():
    data = build_task(TaskConfig(n_pairs=4, seq_len=48, seed=5))
    tokens = mx.array(data[:2])
    logits = mx.zeros((2, 48, VOCAB_SIZE))

    def fn(w):
        return answer_loss(logits + 0.0 * mx.sum(w), tokens)

    a, g = mx.value_and_grad(fn)(mx.zeros((1,)))
    mx.eval(a, g)
    assert np.isfinite(float(a.item()))
    assert np.isfinite(np.array(g)).all()
