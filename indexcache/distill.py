"""Multi-layer distillation: train each retained indexer against the layers it serves.

The paper's training-aware objective is Eq. 1,

    L_multi = sum_{j=0..m} 1/(m+1) * sum_t KL( p_t^{(l+j)} || q_t^{(l)} )

where ``l`` is a retained Full layer, ``l+1..l+m`` are the Shared layers that
reuse its index set, ``p`` is that layer's head-averaged attention distribution
and ``q`` is the indexer's softmax output.  Proposition 1 says its gradient
equals the gradient of distilling against the averaged target
``p_bar = mean_j p^{(l+j)}``, so the two formulations are interchangeable at
training time.  Both are implemented here; ``tests/test_distill.py`` checks the
gradient equality numerically.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_EPS = 1e-12


def aggregated_attention_distribution(weights: mx.array) -> mx.array:
    """(B, H, L, S) softmax weights -> head-averaged (B, L, S) distribution."""
    if weights.ndim != 4:
        raise ValueError(f"expected (B, H, L, S) weights, got {weights.shape}")
    p = mx.mean(weights, axis=1)
    return p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), _EPS)


def _kl_terms(p: mx.array, log_q: mx.array, eps: float) -> mx.array:
    """sum_s p * (log p - log q) per position, summed over the last axis."""
    log_p = mx.log(mx.maximum(p, eps))
    return mx.sum(p * (log_p - log_q), axis=-1)


def distillation_kl(
    target: mx.array,
    logits: mx.array,
    *,
    selected: mx.array | None = None,
    mask: mx.array | None = None,
    subtract_target_entropy: bool = True,
    normalise: bool = False,
    eps: float = 1e-9,
) -> tuple[mx.array, mx.array]:
    """KL(target || softmax(logits)) averaged over positions.

    ``selected`` gathers the last axis of ``logits`` down to the top-k set and
    renormalises, which is the paper's sparse-phase variant ("the KL divergence
    computed only over the selected top-k tokens").  ``target`` is always
    renormalised over the same set.

    With ``subtract_target_entropy`` the returned value is the cross-entropy
    term alone (constant offset removed), so it is exactly zero when the model
    distribution already equals the target *and* its gradient is unchanged.
    ``normalise`` divides the result by ``log(S)``, which keeps the term in
    [0, 1] so that it can be added to a language-modelling loss whose scale is
    set by the vocabulary rather than by the attention distribution.

    Returns ``(loss, n_positions)`` so a caller can weight runs by token count.
    """
    if selected is not None:
        # ``logits`` must span the whole vocabulary: this function gathers the
        # top-k out of it.  Passing scores already restricted to k keys is a
        # silent mis-index, so check the width here.
        if target.shape != logits.shape:
            raise ValueError(
                f"with `selected`, target and logits must both be (B, L, S); "
                f"got {target.shape} and {logits.shape}"
            )
        axis = logits.ndim - 1
        b = logits.shape[0]
        L = logits.shape[1]
        kk = selected.shape[-1]
        offsets = mx.arange(b * L, dtype=mx.int32).reshape(b, L, 1) * logits.shape[-1]
        flat_idx = mx.stop_gradient((offsets + selected.astype(mx.int32)).reshape(-1))
        logits = logits.reshape(b * L * logits.shape[-1])[flat_idx].reshape(b, L, kk)
        target = target.reshape(b * L * target.shape[-1])[flat_idx].reshape(b, L, kk)

    if target.shape != logits.shape:
        raise ValueError(f"target {target.shape} != logits {logits.shape}")

    # -1e30 masking logits make log_softmax return -inf; clamp after the normaliser
    # so p * log q stays finite.
    log_q = mx.maximum(nn.log_softmax(logits.astype(mx.float32), axis=-1), -100.0)
    p = target.astype(mx.float32)
    p = p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), eps)
    if selected is not None:
        # The target is a distribution over all L keys but q only covers the
        # selected top-k, so p is renormalised over the same support.  Dropping
        # this step silently inflates the loss: q's mass sums to 1 over the
        # k keys while p's would not, unless the tail happens to be empty.
        p = p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), eps)
        # Floor the renormalised target: with a tiny top-k set much of the mass
        # can fall outside the shortlist, and log(0) would make the loss NaN.
        # The floor is applied *before* the final renormalisation so the result
        # is still a distribution.
        p = mx.maximum(p, 1e-9)
        p = p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), eps)

    if subtract_target_entropy:
        # Cross-entropy minus the target's own (constant) negative entropy: the
        # same gradient with respect to the indexer, but the value is exactly
        # zero when the indexer's distribution already equals the target.
        # Subtracting H(p) = -sum p log p cannot change the gradient because
        # nothing in that term depends on the indexer parameters.
        log_p = mx.log(mx.maximum(p, eps))
        terms = mx.sum(p * (log_p - log_q), axis=-1)
    else:
        terms = _kl_terms(p, log_q, eps)

    if mask is not None:
        terms = terms * mask.astype(terms.dtype)
        n = mx.sum(mask.astype(mx.float32))
    else:
        n = float(terms.size)
    loss = mx.sum(terms) / mx.maximum(n, 1.0)
    if normalise:
        loss = loss / float(np.log(max(logits.shape[-1], 2)))
    return loss, n


def multi_layer_distillation_loss(
    indexer_logits: mx.array,
    targets: mx.array,
    *,
    selected: mx.array | None = None,
    mask: mx.array | None = None,
    mode: str = "per_layer",
    subtract_target_entropy: bool = True,
    normalise: bool = True,
) -> mx.array:
    """Eq. 1 of the paper.

    Parameters
    ----------
    indexer_logits : (B, L, S) scores from the retained Full layer's indexer
    targets : (m+1, B, L, S) attention distributions of the layers it serves
        (including itself), already detached by the caller
    mode : ``"per_layer"`` for ``sum_j 1/(m+1) KL(p_j || q)`` or ``"averaged"``
        for ``KL(mean_j p_j || q)``.  Proposition 1: identical gradients.
    """
    if targets.ndim != 4:
        raise ValueError(f"targets must be (m+1, B, L, S), got {targets.shape}")
    if mode not in ("per_layer", "averaged"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "averaged":
        p_bar = mx.mean(targets, axis=0)
        loss, _ = distillation_kl(
            p_bar,
            indexer_logits,
            selected=selected,
            mask=mask,
            subtract_target_entropy=subtract_target_entropy,
            normalise=normalise,
        )
        return loss

    total = None
    for j in range(targets.shape[0]):
        loss, _ = distillation_kl(
            targets[j],
            indexer_logits,
            selected=selected,
            mask=mask,
            subtract_target_entropy=subtract_target_entropy,
            normalise=normalise,
        )
        total = loss if total is None else total + loss
    return total / targets.shape[0]
