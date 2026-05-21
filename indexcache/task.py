"""A synthetic multi-key associative-recall task for long-ish contexts.

The model sees ``n_pairs`` key/value bindings shuffled into the context, then a
query naming one of the keys, and must emit the value bound to it.  The answer
value occurs exactly once, typically far from the query, so a query can only
answer by retrieving a distant token: this is the setting where sparse top-k
selection matters and where a shared index set has a chance of being good
enough (the "find the matching key" step is the same at every layer).

Task specifics that the rest of the repo depends on:

* the answer is always read at the final sequence position;
* the query key is placed immediately before it, so a model that cannot
  retrieve can still rule out keys by position;
* keys and values are drawn from disjoint vocabulary ranges, so a model cannot
  answer by copying a token that is nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

PAD = 0
BOS = 1
QUERY = 2
ANSWER = 3
KEY_BASE = 4
N_KEYS = 128
VALUE_BASE = KEY_BASE + N_KEYS
N_VALUES = 128
VOCAB_SIZE = VALUE_BASE + N_VALUES
ANSWER_POSITION = -1


@dataclass(frozen=True)
class TaskConfig:
    n_pairs: int = 6
    seq_len: int = 256
    seed: int = 0

    @property
    def answer_index(self) -> int:
        return self.seq_len - 1


def build_task(cfg: TaskConfig) -> np.ndarray:
    """Generate ``(n_examples, seq_len)`` token ids for the whole task."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.seq_len < 2 * cfg.n_pairs + 3:
        raise ValueError("seq_len is too small for that many key/value pairs")
    out = np.empty((cfg.n_pairs, cfg.seq_len), dtype=np.int32)
    for i in range(cfg.n_pairs):
        keys = rng.choice(N_KEYS, size=cfg.n_pairs, replace=False)
        values = rng.choice(N_VALUES, size=cfg.n_pairs, replace=False)
        tokens = np.full(cfg.seq_len, PAD, dtype=np.int32)
        tokens[0] = BOS
        body = tokens[1 : 1 + 2 * cfg.n_pairs]
        body[:] = np.concatenate(
            [np.array([KEY_BASE + k, VALUE_BASE + v], dtype=np.int32) for k, v in zip(keys, values)]
        )
        rng.shuffle(body.reshape(-1, 2))  # shuffle the pairs, not the tokens in a pair
        # place the query and answer at the end
        target = int(rng.integers(cfg.n_pairs))
        tokens[cfg.seq_len - 3] = QUERY
        tokens[cfg.seq_len - 2] = KEY_BASE + keys[target]
        tokens[cfg.seq_len - 1] = VALUE_BASE + values[target]
        out[i] = tokens
    return out


def answer_tokens(tokens: np.ndarray) -> np.ndarray:
    """(N, L) -> (N,) the value the model has to produce."""
    return tokens[:, ANSWER_POSITION]


def answer_loss(logits: mx.array, tokens: mx.array) -> mx.array:
    """Cross-entropy at the final position only."""
    from mlx.nn import log_softmax

    logp = log_softmax(logits[:, ANSWER_POSITION].astype(mx.float32), axis=-1)
    picked = mx.take_along_axis(
        logp, mx.stop_gradient(tokens[:, ANSWER_POSITION])[:, None], axis=-1
    )
    return -mx.mean(picked)


def answer_accuracy(logits: mx.array, tokens: mx.array) -> float:
    """Fraction of examples whose greedy prediction is the bound value."""
    pred = mx.argmax(logits[:, ANSWER_POSITION], axis=-1)
    mx.eval(pred)
    return float(mx.mean((pred == tokens[:, ANSWER_POSITION]).astype(mx.float32)).item())


def is_answer_position_well_formed(tokens: np.ndarray) -> bool:
    """Sanity checks on a generated batch, used by the tests."""
    n = tokens.shape[0]
    if not np.all(tokens[:, 0] == BOS):
        return False
    if not np.all(tokens[:, -3] == QUERY):
        return False
    if not np.all(tokens[:, ANSWER_POSITION] >= VALUE_BASE):
        return False
    for i in range(n):
        qk = tokens[i, -2]
        if not (KEY_BASE <= qk < VALUE_BASE):
            return False
        # The body holds exactly one (key, value) pair for the query, and the
        # query/answer slots are the second occurrence of each.
        body = tokens[i, 1:-3]
        if int(np.sum(body == qk)) != 1:
            return False
        if int(np.sum(tokens[i] == qk)) != 2:
            return False
        av = tokens[i, ANSWER_POSITION]
        if int(np.sum(body == av)) != 1:
            return False
        # the value must be the one paired with the query key in the body
        pos = int(np.argmax(body == qk))
        if body[pos + 1] != av:
            return False
    return True
