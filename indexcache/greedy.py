"""Training-free pattern search: greedily convert Full layers to Shared.

Algorithm 1 of the paper.  Start from all-Full, then repeat K times: tentatively
flip each remaining Full layer (excluding layer 0) to Shared, evaluate LM loss on
the same fixed calibration batches, and commit the flip with the lowest loss.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .pattern import LayerPattern

EvalFn = Callable[[LayerPattern], float]


def calibration_batches(tokens, batch_size: int, n_batches: int) -> list:
    """Split a token array into a cache of fixed evaluation batches.

    All candidate patterns are scored on exactly these batches, so loss
    differences reflect the pattern and not the data.
    """
    import mlx.core as mx

    if tokens.ndim != 2:
        raise ValueError(f"tokens must be (N, L), got {tokens.shape}")
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")
    every = max(1, tokens.shape[0] // n_batches)
    out = []
    for i in range(min(n_batches, max(1, tokens.shape[0] // every))):
        start = i * every
        chunk = tokens[start : start + every]
        if chunk.shape[0] == 0:
            continue
        out.append(mx.array(chunk[:batch_size] if chunk.shape[0] > batch_size else chunk))
    return out


def greedy_layer_selection(
    evaluate: EvalFn,
    n_layers: int,
    n_shared: int,
    *,
    verbose: bool = False,
    log: Callable[[str], None] | None = None,
) -> tuple[LayerPattern, list[dict]]:
    """Return the searched pattern and a per-step trace.

    ``n_shared=0`` is a no-op: the all-Full pattern is returned without calling
    ``evaluate`` at all.  ``n_shared >= n_layers - 1`` leaves only layer 0 Full
    (the transition layer is always retained); asking for more is an error.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_shared < 0:
        raise ValueError("n_shared must be >= 0")
    max_shared = n_layers - 1
    if n_shared > max_shared:
        raise ValueError(
            f"cannot share {n_shared} of {n_layers} layers: layer 0 is always Full"
        )

    pattern = LayerPattern.all_full(n_layers)
    if n_shared == 0:
        return pattern, []

    candidates = list(range(1, n_layers))
    trace: list[dict] = []
    say = log or (print if verbose else (lambda _m: None))

    for step in range(n_shared):
        best_layer = None
        best_loss = None
        losses = {}
        for layer in candidates:
            trial = LayerPattern.from_full_layers(
                n_layers, [i for i in pattern.full_layers if i != layer]
            )
            loss = float(evaluate(trial))
            losses[layer] = loss
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_layer = layer
        assert best_layer is not None
        pattern = LayerPattern.from_full_layers(
            n_layers, [i for i in pattern.full_layers if i != best_layer]
        )
        candidates.remove(best_layer)
        trace.append(
            {
                "step": step,
                "chosen": best_layer,
                "loss": best_loss,
                "losses": dict(losses),
                "pattern": pattern.pattern,
            }
        )
        say(
            f"  greedy step {step + 1}/{n_shared}: flip layer {best_layer} -> S "
            f"(loss {best_loss:.4f})  pattern={pattern.pattern}"
        )
    return pattern, trace


def order_of_removal(trace: Iterable[dict]) -> list[int]:
    """Layers in the order the greedy search gave them up (least important first)."""
    return [int(step["chosen"]) for step in trace]
