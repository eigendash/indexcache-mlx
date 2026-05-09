"""The decoder model: pattern plumbing, indexer counting and gradient flow."""

import mlx.core as mx
import numpy as np
import mlx.utils as mu
import pytest

from indexcache.model import IndexCacheModel, ModelConfig, count_params
from indexcache.pattern import LayerPattern


def make_model(n_layers=4, n_heads=4, d_model=32, top_k=4, vocab=48):
    return IndexCacheModel(
        ModelConfig(
            vocab_size=vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            top_k=top_k,
            max_seq_len=32,
        )
    )


def test_parameter_count_is_positive_and_matches_the_state_tree():
    m = make_model()
    n = m.n_params()
    assert n == count_params(m)
    assert n > 0
    # calling it must not freeze the model: a gradient still flows afterwards
    tokens = mx.random.randint(0, 48, (2, 8))
    _, grads = mx.value_and_grad(lambda mdl: mdl.loss(tokens))(m)
    mx.eval(grads)
    flat = [np.array(v) for _, v in mu.tree_flatten(grads)]
    assert all(np.isfinite(v).all() for v in flat)
    assert any(np.abs(v).sum() > 0 for v in flat)


def test_the_number_of_indexer_invocations_equals_the_full_layer_count():
    """A Shared layer must not run its own indexer."""
    m = make_model()
    tokens = mx.random.randint(0, 48, (2, 8))
    calls = []
    for layer, block in enumerate(m.blocks):
        original = block.attn.indexer

        def counting(x, _orig=original, _layer=layer):
            calls.append(_layer)
            return _orig(x)

        block.attn.indexer = counting
    for pattern in ("FFFF", "FSFS", "FSSS", "FFSF"):
        calls.clear()
        p = LayerPattern(pattern)
        out = m.forward(tokens, p, collect_indices=True)
        mx.eval(out["logits"])
        # the layer computes its indexer scores once and hands them to the
        # attention layer, so each Full layer appears exactly once
        assert sorted(calls) == list(p.full_layers), pattern
        assert sorted(out["indices"]) == list(p.full_layers)


def test_shared_layers_reuse_the_nearest_full_index_set():
    """The indices a Shared layer attends to are bit-identical to its source's."""
    m = make_model(n_layers=6)
    tokens = mx.random.randint(0, 48, (2, 8))
    pattern = LayerPattern("FSFFSS")  # needs a 6-layer model
    seen = {}
    original_forward = m.blocks[0].__class__.__call__

    def spy(self, x, selected, need_dist=False, logits=None):
        seen.setdefault(id(self), []).append(selected)
        return original_forward(self, x, selected, need_dist, logits)

    m.blocks[0].__class__.__call__ = spy
    try:
        out = m.forward(tokens, pattern, collect_indices=True)
        mx.eval(out["logits"])
    finally:
        m.blocks[0].__class__.__call__ = original_forward

    for layer in range(6):
        source = pattern.source_for(layer)
        got = seen[id(m.blocks[layer])][0]
        want = out["indices"][source]
        np.testing.assert_array_equal(np.array(got), np.array(want), err_msg=f"layer {layer}")


def test_layer_zero_must_be_full():
    m = make_model()
    tokens = mx.random.randint(0, 48, (2, 4))
    with pytest.raises(ValueError):
        m.forward(tokens, LayerPattern("SFFF"))


def test_pattern_length_must_match_the_model():
    m = make_model(n_layers=4)
    tokens = mx.random.randint(0, 48, (2, 4))
    with pytest.raises(ValueError):
        m.forward(tokens, LayerPattern("FF"))


def test_sequence_longer_than_max_seq_len_is_rejected():
    m = make_model()
    with pytest.raises(ValueError):
        m.forward(mx.zeros((1, 64), dtype=mx.int32))


def test_gradients_reach_only_the_retained_indexers():
    """With the multi-layer distillation loss and indexer_only, an S layer's
    indexer receives no gradient."""
    m = make_model(n_layers=4, top_k=4)
    tokens = mx.random.randint(0, 48, (2, 8))
    pattern = LayerPattern("FSFS")
    _, grads = mx.value_and_grad(lambda mdl: mdl.multi_layer_distillation(tokens, pattern))(m)
    mx.eval(grads)
    flat = dict(mu.tree_flatten(grads))
    for layer in pattern.full_layers:
        key = f"blocks.{layer}.attn.indexer.q_proj.weight"
        assert key in flat
        assert np.abs(np.array(flat[key])).sum() > 0, f"layer {layer} has no gradient"
    for layer in pattern.shared_layers:
        key = f"blocks.{layer}.attn.indexer.q_proj.weight"
        if key in flat:
            np.testing.assert_allclose(np.array(flat[key]), 0.0, atol=0.0)


def test_gradients_are_finite_for_the_language_modelling_loss():
    m = make_model()
    tokens = mx.random.randint(0, 48, (2, 16))
    for pattern in (None, LayerPattern("FSFS"), LayerPattern("FFSS")):
        loss, grads = mx.value_and_grad(lambda mdl: mdl.loss(tokens, pattern))(m)
        mx.eval(loss, grads)
        assert np.isfinite(float(loss.item()))
        for name, value in mu.tree_flatten(grads):
            assert np.isfinite(np.array(value)).all(), name


def test_forward_shapes_and_distribution_contract():
    m = make_model()
    tokens = mx.random.randint(0, 48, (2, 12))
    out = m.forward(tokens, collect_weights=True, collect_indices=True)
    mx.eval(out["logits"])
    assert out["logits"].shape == (2, 12, 48)
    assert out["logits"].dtype == mx.float32
    for layer, w in out["weights"].items():
        mx.eval(w)
        assert w.shape == (2, 4, 12, 12)
        np.testing.assert_allclose(
            np.array(mx.sum(w, axis=-1)), 1.0, atol=1e-5, err_msg=f"layer {layer}"
        )
    for layer, i in out["indices"].items():
        mx.eval(i)
        assert i.shape == (2, 12, 4)
        assert int(mx.min(i).item()) == 0


def test_distillation_targets_are_detached_from_the_graph():
    m = make_model(n_layers=2)
    tokens = mx.random.randint(0, 48, (2, 8))
    pattern = LayerPattern("FS")
    targets, indices = m.distillation_targets(tokens, pattern)
    assert set(targets) == {0, 1}  # every layer, Shared ones included
    assert set(indices) == {0}  # only the Full layer selects
    for p in targets.values():
        mx.eval(p)
        assert bool(mx.all(mx.isfinite(p)).item())
