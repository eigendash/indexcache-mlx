"""The multi-layer distillation objective of Section 3.2."""

import mlx.core as mx
import numpy as np
import mlx.utils as mu
import pytest

from indexcache.distill import (
    aggregated_attention_distribution,
    distillation_kl,
    multi_layer_distillation_loss,
)


def _dist(B=2, L=6):
    rng = np.random.default_rng(0)
    p = rng.random((B, L, L)).astype(np.float32)
    p = p / p.sum(-1, keepdims=True)
    return mx.array(p)


def test_aggregated_distribution_averages_heads_and_renormalises():
    w = mx.array(np.array([[[[0.6, 0.4], [0.2, 0.8]]], [[[0.0, 1.0], [1.0, 0.0]]]], dtype=np.float32))
    # shape (B=2, H=1, L=2, S=2)
    p = aggregated_attention_distribution(w)
    mx.eval(p)
    assert p.shape == (2, 2, 2)
    np.testing.assert_allclose(np.array(p).sum(-1), 1.0, atol=1e-6)


def test_distillation_loss_is_zero_when_the_distribution_already_matches():
    """The invariant the paper's Proposition is about."""
    target = _dist()
    logits = mx.log(mx.maximum(target, 1e-12))  # softmax(log p) == p
    loss, n = distillation_kl(target, logits)
    mx.eval(loss)
    assert n == target.shape[0] * target.shape[1]
    assert abs(float(loss.item())) < 1e-4


def test_distillation_loss_is_positive_for_a_mismatched_distribution():
    target = _dist()
    logits = mx.zeros(target.shape)
    loss, _ = distillation_kl(target, logits)
    mx.eval(loss)
    assert float(loss.item()) > 0.0


def test_kl_without_the_entropy_correction_is_never_negative():
    target = _dist()
    logits = mx.random.normal(target.shape) * 3.0
    short, _ = distillation_kl(target, logits)
    full, _ = distillation_kl(target, logits, subtract_target_entropy=False)
    mx.eval(short, full)
    assert float(full.item()) >= -1e-6
    assert float(short.item()) >= -1e-6


def test_restricting_the_kl_to_the_top_k_renormalises_the_target():
    """The sparse-phase variant: KL only over the selected top-k tokens."""
    target = _dist(B=1, L=6)
    selected = mx.array(np.array([[[0, 1, 2]]], dtype=np.int32))
    # Full-vocabulary scores: the conditional distribution over keys 0..2 in
    # those slots and effectively -inf elsewhere.  After the loss gathers the
    # selected keys and renormalises, target and model agree exactly.
    pn = np.array(target)[:, :, :3]
    pn = pn / pn.sum(-1, keepdims=True)
    full = np.full((1, 6, 6), -1e9, dtype=np.float32)
    full[:, :, :3] = np.log(pn)
    loss, _ = distillation_kl(target, mx.array(full), selected=selected)
    mx.eval(loss)
    assert abs(float(loss.item())) < 1e-4
    # Mismatched scores move the loss away from zero.
    other = np.zeros((1, 6, 6), dtype=np.float32)
    worse, _ = distillation_kl(target, mx.array(other), selected=selected)
    mx.eval(worse)
    assert float(worse.item()) > 1e-3


def test_per_layer_and_averaged_target_modes_have_equal_gradients():
    """Proposition 1, checked numerically on a tiny parameterised indexer."""
    rng = np.random.default_rng(3)
    targets = mx.array(rng.random((3, 2, 6, 6)).astype(np.float32))
    targets = targets / targets.sum(-1, keepdims=True)
    w = mx.array(rng.normal(size=(6, 6)).astype(np.float32) * 0.1)
    x = mx.array(rng.normal(size=(2, 6, 6)).astype(np.float32))

    def loss_per_layer(w_):
        return multi_layer_distillation_loss(mx.matmul(x, w_), targets, mode="per_layer")

    def loss_averaged(w_):
        return multi_layer_distillation_loss(mx.matmul(x, w_), targets, mode="averaged")

    g1 = mx.grad(loss_per_layer)(w)
    g2 = mx.grad(loss_averaged)(w)
    mx.eval(g1, g2)
    np.testing.assert_allclose(np.array(g1), np.array(g2), rtol=1e-4, atol=1e-6)


def test_multi_layer_loss_averages_over_the_served_layers():
    targets = mx.stack([_dist(B=1, L=4), _dist(B=1, L=4)], axis=0)
    logits = mx.zeros((1, 4, 4))
    loss = multi_layer_distillation_loss(logits, targets, mode="per_layer")
    a, _ = distillation_kl(targets[0], logits)
    b, _ = distillation_kl(targets[1], logits)
    mx.eval(loss, a, b)
    assert float(loss.item()) == pytest.approx((float(a.item()) + float(b.item())) / 2, rel=1e-5)


def test_multi_layer_loss_rejects_bad_shapes_and_modes():
    with pytest.raises(ValueError):
        multi_layer_distillation_loss(mx.zeros((1, 4, 4)), mx.zeros((1, 4)), mode="per_layer")
    with pytest.raises(ValueError):
        multi_layer_distillation_loss(mx.zeros((1, 4, 4)), mx.zeros((1, 1, 4, 4)), mode="nope")


def test_negative_attention_weights_are_rejected():
    with pytest.raises(ValueError):
        aggregated_attention_distribution(mx.zeros((1, 2, 3)))


def test_gradients_are_finite_through_the_distillation_loss():
    target = _dist(B=2, L=5)
    w = mx.array(np.random.default_rng(1).normal(size=(5, 5)).astype(np.float32))
    x = mx.array(np.random.default_rng(2).normal(size=(2, 5, 5)).astype(np.float32))
    loss, grads = mx.value_and_grad(lambda w_: distillation_kl(target, mx.matmul(x, x * 1.0 @ w_))[0])(w)
    mx.eval(loss, grads)
    assert np.isfinite(np.array(grads)).all()
