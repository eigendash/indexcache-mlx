"""IndexCache-style cross-layer top-k index reuse for sparse attention, in MLX.

Public API
----------
``DSAIndexer`` / ``indexer_scores``  -- the "lightning indexer" scorer.
``SparseAttention``                  -- top-k (plus optional sliding window) attention.
``LayerPattern``                      -- Full/Shared layer partition and index reuse.
``IndexCacheModel``                   -- small decoder with per-layer or reused indices.
``multi_layer_distillation_loss``     -- the training-aware objective.
``greedy_layer_selection``            -- the training-free pattern search.
"""

from .attention import SparseAttention, resolve_indices
from .distill import (
    aggregated_attention_distribution,
    distillation_kl,
    multi_layer_distillation_loss,
)
from .greedy import greedy_layer_selection
from .indexer import DSAIndexer, indexer_scores, topk_indices
from .model import ModelConfig, IndexCacheModel, count_params
from .pattern import LayerPattern

__all__ = [
    "DSAIndexer",
    "indexer_scores",
    "topk_indices",
    "SparseAttention",
    "resolve_indices",
    "LayerPattern",
    "ModelConfig",
    "IndexCacheModel",
    "count_params",
    "aggregated_attention_distribution",
    "distillation_kl",
    "multi_layer_distillation_loss",
    "greedy_layer_selection",
]
