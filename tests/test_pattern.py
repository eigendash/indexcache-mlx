"""The Full/Shared layer partition and its index-reuse rule."""

import pytest

from indexcache.pattern import LayerPattern


def test_all_full_and_helpers():
    p = LayerPattern.all_full(4)
    assert p.pattern == "FFFF"
    assert p.n_full == 4
    assert p.n_shared == 0
    assert p.indexer_count == 4
    assert p.indexer_saving() == 0.0
    assert p.full_layers == (0, 1, 2, 3)
    assert p.shared_layers == ()


def test_interleaved_one_in_four():
    p = LayerPattern.interleaved(8, 4)
    assert p.pattern == "FSSSFSSS"
    assert p.n_full == 2
    assert p.n_shared == 6
    assert p.indexer_saving() == pytest.approx(0.75)


def test_shared_layer_reuses_the_nearest_preceding_full_layer():
    p = LayerPattern("FSFFSSS")
    assert p.resolve() == (0, 0, 2, 3, 3, 3, 3)


def test_serves_is_the_converse_of_source_for():
    p = LayerPattern("FSFFSSS")
    assert p.serves(0) == (0, 1)
    assert p.serves(2) == (2,)
    assert p.serves(3) == (3, 4, 5, 6)
    for layer in range(len(p)):
        assert layer in p.serves(p.source_for(layer))


def test_consecutive_full_layers_each_serve_themselves():
    p = LayerPattern("FFSS")
    assert p.resolve() == (0, 1, 1, 1)
    assert p.serves(0) == (0,)
    assert p.serves(1) == (1, 2, 3)


def test_first_layer_source_is_itself():
    for pattern in ("FFFF", "FSSS", "FSFS"):
        p = LayerPattern(pattern)
        assert p.source_for(0) == 0


def test_all_shared_pattern_raises_for_the_layers_with_no_predecessor():
    p = LayerPattern("SS")
    with pytest.raises(ValueError):
        p.source_for(0)
    with pytest.raises(ValueError):
        p.resolve()


def test_single_layer_model():
    p = LayerPattern("F")
    assert p.resolve() == (0,)
    assert p.serves(0) == (0,)


def test_from_full_layers_and_round_trip():
    p = LayerPattern.from_full_layers(6, [0, 3, 5])
    assert p.pattern == "FSSFSF"
    assert p.full_layers == (0, 3, 5)
    assert LayerPattern(p.pattern) == p


def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        LayerPattern("")
    with pytest.raises(ValueError):
        LayerPattern("FXF")
    with pytest.raises(ValueError):
        LayerPattern.interleaved(4, 0)
    p = LayerPattern("FS")
    with pytest.raises(IndexError):
        p.source_for(5)
    with pytest.raises(ValueError):
        p.serves(1)


def test_describe_reports_the_saving():
    d = LayerPattern("FSFS").describe()
    assert d["pattern"] == "FSFS"
    assert d["n_shared"] == 2
    assert d["indexer_saving"] == pytest.approx(0.5)
    assert d["resolve"] == [0, 0, 2, 2]
