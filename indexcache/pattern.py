"""Full/Shared layer partition and the index-reuse rule.

The paper encodes the configuration as a pattern string over ``{F, S}``: an
``F`` (Full) layer keeps its lightning indexer and computes fresh top-k indices,
an ``S`` (Shared) layer has no indexer and inherits the index set of the nearest
*preceding* Full layer, ``f(l) = max{j < l : c_j = F}``.  The first layer is
always Full, which makes ``f`` total for the paper's patterns.
"""

from __future__ import annotations

from dataclasses import dataclass


def _normalise(pattern) -> str:
    if isinstance(pattern, str):
        s = pattern.strip().upper()
    else:
        chars = []
        for c in pattern:
            if isinstance(c, bool):
                chars.append("F" if c else "S")
            else:
                chars.append(str(c).strip().upper())
        s = "".join(chars)
    if not s:
        raise ValueError("empty pattern")
    if any(c not in "FS" for c in s):
        raise ValueError(f"pattern may only contain F and S, got {s!r}")
    return s


@dataclass(frozen=True)
class LayerPattern:
    """A pattern string with the reuse rule resolved.

    A pattern whose first character is ``S`` is allowed by the class but then
    every layer before the first ``F`` has no predecessor to reuse from; calling
    :meth:`source_for` on such a layer raises ``ValueError``.  The paper sidesteps
    this by fixing layer 1 to Full.
    """

    pattern: str

    def __post_init__(self):
        object.__setattr__(self, "pattern", _normalise(self.pattern))

    # -- basics -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pattern)

    @property
    def n_layers(self) -> int:
        return len(self.pattern)

    def is_full(self, layer: int) -> bool:
        return self.pattern[layer] == "F"

    @property
    def full_layers(self) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.pattern) if c == "F")

    @property
    def shared_layers(self) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.pattern) if c == "S")

    @property
    def n_full(self) -> int:
        return len(self.full_layers)

    @property
    def n_shared(self) -> int:
        return len(self.shared_layers)

    @property
    def indexer_count(self) -> int:
        """Number of indexer invocations per full forward pass."""
        return self.n_full

    def indexer_saving(self) -> float:
        """Fraction of indexer computations removed relative to all-Full."""
        return self.n_shared / self.n_layers

    # -- reuse rule -------------------------------------------------------
    def source_for(self, layer: int) -> int:
        """The Full layer whose indices ``layer`` reuses (itself, if Full)."""
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.pattern!r}")
        if self.is_full(layer):
            return layer
        for j in range(layer - 1, -1, -1):
            if self.pattern[j] == "F":
                return j
        raise ValueError(
            f"layer {layer} is Shared but has no preceding Full layer in {self.pattern!r}"
        )

    def serves(self, layer: int) -> tuple[int, ...]:
        """All layers (including ``layer``) that share ``layer``'s index set."""
        if not self.is_full(layer):
            raise ValueError(f"layer {layer} is not a Full layer")
        return tuple(i for i in range(layer, self.n_layers) if self.source_for(i) == layer)

    def resolve(self) -> tuple[int, ...]:
        """Per-layer source index; ``resolve()[l]`` is the layer providing indices."""
        return tuple(self.source_for(i) for i in range(self.n_layers))

    # -- constructors -----------------------------------------------------
    @classmethod
    def all_full(cls, n_layers: int) -> "LayerPattern":
        return cls("F" * n_layers)

    @classmethod
    def interleaved(cls, n_layers: int, every: int) -> "LayerPattern":
        """Keep every ``every``-th layer Full, starting at layer 0 (e.g. ``FSFS``)."""
        if every < 1:
            raise ValueError("every must be >= 1")
        return cls("".join("F" if i % every == 0 else "S" for i in range(n_layers)))

    @classmethod
    def from_full_layers(cls, n_layers: int, full_layers) -> "LayerPattern":
        keep = set(int(i) for i in full_layers)
        return cls("".join("F" if i in keep else "S" for i in range(n_layers)))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.pattern

    def describe(self) -> dict:
        return {
            "pattern": self.pattern,
            "n_layers": self.n_layers,
            "n_full": self.n_full,
            "n_shared": self.n_shared,
            "indexer_saving": self.indexer_saving(),
            "full_layers": list(self.full_layers),
            "resolve": list(self.resolve()),
        }
