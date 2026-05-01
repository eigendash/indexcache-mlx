"""A small decoder-only transformer whose attention layers are DSA-style.

The model is deliberately toy-scale.  It exists so that the Full/Shared layer
partition, the top-k index reuse and the multi-layer distillation loss can be
exercised end to end; it is not a scaled-down GLM/DeepSeek architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .attention import SparseAttention, resolve_indices
from .distill import aggregated_attention_distribution, multi_layer_distillation_loss
from .indexer import DSAIndexer
from .pattern import LayerPattern


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int | None = None
    max_seq_len: int = 512
    top_k: int = 8
    window: int = -1
    indexer_heads: int = 2
    indexer_head_dim: int = 16

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class AttentionLayer(nn.Module):
    """One DSA attention layer: lightning indexer + sparse core attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.indexer = DSAIndexer(cfg.d_model, cfg.indexer_heads, cfg.indexer_head_dim)
        self.attn = SparseAttention(window=cfg.window)

    def _qkv(self, x: mx.array):
        b, L, _ = x.shape
        h, d = self.cfg.n_heads, self.head_dim
        q = mx.transpose(self.q_proj(x).reshape(b, L, h, d), (0, 2, 1, 3))
        k = mx.transpose(self.k_proj(x).reshape(b, L, h, d), (0, 2, 1, 3))
        v = mx.transpose(self.v_proj(x).reshape(b, L, h, d), (0, 2, 1, 3))
        return q, k, v

    def __call__(
        self,
        x: mx.array,
        selected: mx.array | None,
        need_dist: bool = False,
        logits: mx.array | None = None,
    ):
        """Returns ``(output, attention_distribution, indexer_logits)``.

        The output uses the sparse path when ``selected`` is given.  When
        ``need_dist`` is set the returned distribution is the full head-averaged
        attention distribution ``p`` over all keys (computed with the same
        q/k/v), because that is the distillation target of Section 3.2 whatever
        role the layer plays; that extra forward costs O(L^2).
        """
        b, L, _ = x.shape
        q, k, v = self._qkv(x)
        if logits is None:
            logits = self.indexer(x)
        if selected is None:
            out, weights = self.attn(q, k, v, None)
        else:
            out, _ = self.attn(q, k, v, selected)
            weights = None
            if need_dist:
                _, weights = self.attn(q, k, v, None)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, L, self.cfg.d_model)
        return self.o_proj(out), weights, logits


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = AttentionLayer(cfg)
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )

    def __call__(
        self,
        x: mx.array,
        selected: mx.array | None,
        need_dist: bool = False,
        logits: mx.array | None = None,
    ):
        h, weights, lg = self.attn(self.norm1(x), selected, need_dist, logits)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, weights, lg


class IndexCacheModel(nn.Module):
    """Decoder-only LM with a configurable Full/Shared indexer pattern."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    # -- helpers ----------------------------------------------------------
    def n_params(self) -> int:
        """Parameter count (see :func:`count_params`; ``state``, never frozen)."""
        return count_params(self)

    def _embed(self, tokens: mx.array) -> mx.array:
        b, L = tokens.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(f"sequence {L} exceeds max_seq_len {self.cfg.max_seq_len}")
        return self.embed(tokens) + self.pos(mx.arange(L))[None]

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        tokens: mx.array,
        pattern: LayerPattern | None = None,
        *,
        collect_weights: bool = False,
        collect_indices: bool = False,
    ) -> dict:
        """Run the model under ``pattern`` (default all-Full).

        Returns a dict with ``logits``, ``indices`` ({layer: (B, L, k)} for the
        Full layers, when requested), ``weights`` ({layer: (B, H, L, L)} for the
        Full layers, when requested), ``selected`` and ``sources``.
        """
        if pattern is None:
            pattern = LayerPattern.all_full(self.cfg.n_layers)
        if pattern.n_layers != self.cfg.n_layers:
            raise ValueError(
                f"pattern has {pattern.n_layers} layers, model has {self.cfg.n_layers}"
            )
        if not pattern.is_full(0):
            raise ValueError("layer 0 must be Full: it seeds the shared indices")

        x = self._embed(tokens)
        cached: mx.array | None = None
        cache_source: int = 0
        indices: dict[int, mx.array] = {}
        weights: dict[int, mx.array] = {}

        for layer, block in enumerate(self.blocks):
            if pattern.is_full(layer):
                cached = resolve_indices(
                    block.attn.indexer(x),
                    self.cfg.top_k,
                    window=self.cfg.window,
                )
                cache_source = layer
                if collect_indices:
                    indices[layer] = cached
            x, w, _logits = block(x, cached, need_dist=collect_weights)
            if w is not None:
                weights[layer] = w
        del cache_source

        logits = self.lm_head(self.norm_f(x))
        return {
            "logits": logits,
            "indices": indices,
            "weights": weights,
            "selected": cached,
            "sources": pattern.resolve(),
        }

    # -- losses -----------------------------------------------------------
    def loss(self, tokens: mx.array, pattern: LayerPattern | None = None) -> mx.array:
        """Next-token cross-entropy, averaged over all positions."""
        logits = self.forward(tokens, pattern)["logits"]
        targets = tokens[:, 1:]
        logp = nn.log_softmax(logits[:, :-1].astype(mx.float32), axis=-1)
        picked = mx.take_along_axis(logp, targets[..., None], axis=-1)
        return -mx.mean(picked)

    def distillation_targets(
        self, tokens: mx.array, pattern: LayerPattern
    ) -> tuple[dict[int, mx.array], dict[int, mx.array]]:
        """Detached attention distributions + selected indices for all layers.

        A Shared layer is the target of the Full layer it copies from, so its
        full attention distribution is needed too.  Targets are detached: DSA
        trains the indexer "on a detached computational graph".
        """
        out = self.forward(tokens, pattern, collect_weights=True, collect_indices=True)
        targets = {
            layer: mx.stop_gradient(aggregated_attention_distribution(w))
            for layer, w in out["weights"].items()
        }
        return targets, out["indices"]

    def multi_layer_distillation(
        self,
        tokens: mx.array,
        pattern: LayerPattern,
        *,
        mode: str = "per_layer",
        mask: mx.array | None = None,
        indexer_only: bool = True,
    ) -> mx.array:
        """The multi-layer disillation loss of Section 3.2, as a forward pass.

        ``indexer_only=True`` refreshes the hidden states with ``mx.stop_gradient``
        at every layer, which is the DSA dense warm-up setting: only indexer
        parameters are trained and the representation the targets came from is
        held fixed.  With ``indexer_only=False`` the states stay attached to the
        graph so gradients reach every parameter that produced them.
        """
        targets, indices = self.distillation_targets(tokens, pattern)
        x = self._embed(tokens)
        total = mx.array(0.0)
        n_full = 0
        for layer, block in enumerate(self.blocks):
            if pattern.is_full(layer):
                served = pattern.serves(layer)
                stack = mx.stack([targets[j] for j in served], axis=0)
                logits = block.attn.indexer(x)
                total = total + multi_layer_distillation_loss(
                    logits,
                    stack,
                    selected=indices[layer] if pattern.n_shared else None,
                    mask=mask,
                    mode=mode,
                )
                n_full += 1
            parent = pattern.source_for(layer)
            with_state = block(x, indices[parent], False)[0]
            x = mx.stop_gradient(with_state) if indexer_only else with_state
        return total / max(n_full, 1)


def tree_flatten_state(tree, prefix=""):
    """Flatten a nested state tree into ``(name, array)`` pairs without freezing."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from tree_flatten_state(v, f"{prefix}{k}.")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from tree_flatten_state(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), tree


def count_params(model) -> int:
    """Number of scalar parameters in ``model`` (read-only; never freezes)."""
    total = 0
    for _name, value in tree_flatten_state(model.state):
        mx.eval(value)
        total += int(value.size)
    return total
