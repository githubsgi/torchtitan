# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from torchtitan.observability import tensor_logging
from torchtitan.observability.tensor_logging import vector_metrics


class _Source(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


@pytest.fixture
def model() -> nn.Module:
    root = nn.Module()
    root.moe = _Source()
    return root


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if vector_metrics._active_state is not None:
        vector_metrics._active_state.close()
    from torchtitan.observability.tensor_logging import runtime

    if runtime._active_state is not None:
        runtime._active_state.close()


def _init(model: nn.Module, **kwargs) -> vector_metrics.VectorLoggingState:
    device = torch.device("cpu")
    tensor_logging.init(model, device=device)
    return vector_metrics.init_vectors(model, device=device, **kwargs)


def test_vector_is_reduced_element_wise(model: nn.Module) -> None:
    """Element-wise reduction keeps the imbalance ratio correct.

    Two shards observing [10, 10] and [0, 20] must combine to [10, 30], giving
    30 / 20 = 1.5. Pooling the four values as one population would give 2.0.
    """
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(
            model.moe, tokens_per_expert=torch.tensor([10.0, 10.0])
        )
        vector_metrics.log_vector(
            model.moe, tokens_per_expert=torch.tensor([0.0, 20.0])
        )

    metrics = state.collect()
    assert metrics["moe.tokens_per_expert.total"] == 40.0
    assert metrics["moe.tokens_per_expert.max"] == 30.0
    assert metrics["moe.tokens_per_expert.min"] == 10.0
    assert metrics["moe.tokens_per_expert.mean"] == 20.0
    assert metrics["moe.tokens_per_expert.imbalance"] == 1.5


def test_collect_vectors_returns_full_width(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=3)
    state = _init(model)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(
            model.moe, tokens_per_expert=torch.tensor([1.0, 2.0, 3.0])
        )

    assert state.collect_vectors() == {"moe.tokens_per_expert": [1.0, 2.0, 3.0]}


def test_disabled_step_does_not_accumulate(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model)

    with tensor_logging.set_enabled(False):
        assert not tensor_logging.should_run_logging_calls()

    # The device gate must also neutralize a write that reaches the buffer,
    # which is what happens inside a captured graph on an off-cadence step.
    state._enabled.fill_(0)
    vector_metrics.log_vector(model.moe, tokens_per_expert=torch.tensor([5.0, 5.0]))
    assert state.collect()["moe.tokens_per_expert.total"] == 0.0


def test_collect_clears_between_steps(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(model.moe, tokens_per_expert=torch.tensor([1.0, 1.0]))
    assert state.collect()["moe.tokens_per_expert.total"] == 2.0
    assert state.collect()["moe.tokens_per_expert.total"] == 0.0


def test_ragged_widths_share_one_slab(model: nn.Module) -> None:
    """A narrow metric must not read a wider metric's padding columns."""
    model.other = _Source()
    vector_metrics.register_vector(model.moe, ["wide"], width=4)
    vector_metrics.register_vector(model.other, ["narrow"], width=2)
    state = _init(model)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(model.moe, wide=torch.tensor([1.0, 2.0, 3.0, 4.0]))
        vector_metrics.log_vector(model.other, narrow=torch.tensor([9.0, 9.0]))

    vectors = state.collect_vectors()
    assert vectors["moe.wide"] == [1.0, 2.0, 3.0, 4.0]
    assert vectors["other.narrow"] == [9.0, 9.0]


def test_publish_elements_emits_one_scalar_per_element(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model, publish_elements=True)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(model.moe, tokens_per_expert=torch.tensor([7.0, 8.0]))

    metrics = state.collect()
    assert metrics["moe.tokens_per_expert.e0"] == 7.0
    assert metrics["moe.tokens_per_expert.e1"] == 8.0


def test_unregistered_name_is_rejected(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    _init(model)

    with tensor_logging.set_enabled(True):
        with pytest.raises(KeyError, match="unregistered vector metric"):
            vector_metrics.log_vector(model.moe, missing=torch.tensor([1.0, 1.0]))


def test_register_after_init_is_rejected(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    _init(model)

    with pytest.raises(RuntimeError, match="before tensor_logging.init"):
        vector_metrics.register_vector(model.moe, ["late"], width=2)


def test_close_removes_lookup_state(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model)
    assert vector_metrics._VECTOR_BUFFER_SLOTS_ATTR in model.moe.__dict__

    state.close()
    assert vector_metrics._VECTOR_BUFFER_SLOTS_ATTR not in model.moe.__dict__
    assert "_tensor_logging_vectors" not in model._modules


def test_collect_with_vectors_returns_both_and_resets(model: nn.Module) -> None:
    vector_metrics.register_vector(model.moe, ["tokens_per_expert"], width=2)
    state = _init(model)

    with tensor_logging.set_enabled(True):
        vector_metrics.log_vector(model.moe, tokens_per_expert=torch.tensor([1.0, 3.0]))

    metrics, vectors = state.collect_with_vectors()
    assert vectors["moe.tokens_per_expert"] == [1.0, 3.0]
    assert metrics["moe.tokens_per_expert.imbalance"] == 1.5

    # The buffers must be cleared exactly once, as `collect()` would do.
    _, vectors_after = state.collect_with_vectors()
    assert vectors_after["moe.tokens_per_expert"] == [0.0, 0.0]


class _RecordingSink:
    def __init__(self) -> None:
        self.histograms: dict[str, list[float]] = {}
        self.binned: dict[str, tuple[list[float], list[float]]] = {}
        self.images: dict[str, torch.Tensor] = {}
        self.texts: dict[str, str] = {}

    def log_histogram(self, name: str, values, step: int) -> None:
        self.histograms[name] = list(values)

    def log_histogram_bins(self, name: str, bin_edges, counts, step: int) -> None:
        self.binned[name] = (list(bin_edges), list(counts))

    def log_image(self, name: str, image_chw: torch.Tensor, step: int) -> None:
        self.images[name] = image_chw

    def log_text(self, name: str, text: str, step: int) -> None:
        self.texts[name] = text


def _cell_colors(image_chw: torch.Tensor, rows: int, cols: int) -> list[list[tuple]]:
    """Undo the publish-time zoom by sampling the centre of each cell."""

    zoom_h = image_chw.shape[1] // rows
    zoom_w = image_chw.shape[2] // cols
    return [
        [
            tuple(
                image_chw[:, r * zoom_h + zoom_h // 2, c * zoom_w + zoom_w // 2].tolist()
            )
            for c in range(cols)
        ]
        for r in range(rows)
    ]


def test_publish_vector_views_emits_histogram_per_vector() -> None:
    sink = _RecordingSink()
    vector_metrics.publish_vector_views(
        sink, {"layers.0.moe.tokens_per_expert": [1.0, 3.0]}, step=5
    )
    assert sink.histograms["layers.0.moe.tokens_per_expert.hist"] == [1.0, 3.0]
    # A lone member is already fully described by its histogram.
    assert sink.images == {}


def test_element_histogram_bins_by_index_not_magnitude() -> None:
    """One bin per element, so the x axis matches the heatmap's columns."""

    sink = _RecordingSink()
    vector_metrics.publish_vector_views(
        sink, {"layers.0.moe.tokens_per_expert": [7.0, 0.0, 3.0]}, step=5
    )

    edges, counts = sink.binned["layers.0.moe.tokens_per_expert.hist_by_element"]
    assert edges == [0.0, 1.0, 2.0, 3.0]
    # Bar height is the element's own value, in element order.
    assert counts == [7.0, 0.0, 3.0]


def test_publish_vector_views_stacks_siblings_into_one_heatmap() -> None:
    sink = _RecordingSink()
    vector_metrics.publish_vector_views(
        sink,
        {
            # Deliberately out of order: rows must follow the layer index.
            "layers.10.moe.tokens_per_expert": [0.0, 4.0],
            "layers.2.moe.tokens_per_expert": [2.0, 2.0],
        },
        step=5,
    )

    image = sink.images["layers.all.moe.tokens_per_expert.heatmap"]
    assert image.shape[0] == 3
    cells = _cell_colors(image, rows=2, cols=2)
    # Rows are scaled by their own max, so a balanced layer reads uniform
    # regardless of how many tokens it saw.
    assert cells[0][0] == cells[0][1]
    # The unbalanced layer's cold and hot experts must land on different colors.
    assert cells[1][0] != cells[1][1]


def test_publish_vector_views_labels_heatmap_rows() -> None:
    sink = _RecordingSink()
    vector_metrics.publish_vector_views(
        sink,
        {
            "layers.10.moe.tokens_per_expert": [0.0, 4.0],
            "layers.2.moe.tokens_per_expert": [2.0, 2.0],
        },
        step=5,
    )

    legend = sink.texts["layers.all.moe.tokens_per_expert.heatmap.legend"]
    assert "| 0 | layers.2.moe.tokens_per_expert | 2 | 1.000 |" in legend
    assert "| 1 | layers.10.moe.tokens_per_expert | 4 | 2.000 |" in legend


def test_publish_vector_views_prefixes_every_tag_with_the_namespace() -> None:
    """MoE expert load is grouped-GEMM shape data, so all its views share a section."""

    sink = _RecordingSink()
    history: dict[str, list[list[float]]] = {}
    vector_metrics.publish_vector_views(
        sink,
        {
            "layers.0.moe.tokens_per_expert": [1.0, 3.0],
            "layers.1.moe.tokens_per_expert": [2.0, 2.0],
        },
        step=5,
        history=history,
        namespace="metrics_tensor_shapes",
    )

    emitted = [*sink.histograms, *sink.binned, *sink.images, *sink.texts]
    assert emitted and all(
        name.startswith("metrics_tensor_shapes/") for name in emitted
    )
    # History stays keyed by the bare metric name, so a namespace change cannot
    # orphan an accumulated timeline.
    assert set(history) == {
        "layers.0.moe.tokens_per_expert",
        "layers.1.moe.tokens_per_expert",
    }


def test_publish_vector_views_accumulates_a_timeline_per_metric() -> None:
    sink = _RecordingSink()
    history: dict[str, list[list[float]]] = {}
    for step in range(3):
        vector_metrics.publish_vector_views(
            sink,
            {"layers.0.moe.tokens_per_expert": [1.0, float(step + 1)]},
            step=step,
            history=history,
        )

    assert history["layers.0.moe.tokens_per_expert"] == [
        [1.0, 1.0],
        [0.5, 1.0],
        [1.0 / 3.0, 1.0],
    ]
    image = sink.images["layers.0.moe.tokens_per_expert.timeline"]
    assert image.shape[0] == 3
    assert _cell_colors(image, rows=3, cols=2)[0][0] == _cell_colors(
        image, rows=3, cols=2
    )[0][1]


def test_publish_vector_views_bounds_timeline_history() -> None:
    sink = _RecordingSink()
    history: dict[str, list[list[float]]] = {}
    for step in range(vector_metrics._MAX_TIMELINE_ROWS + 5):
        vector_metrics.publish_vector_views(
            sink,
            {"layers.0.moe.tokens_per_expert": [1.0, 2.0]},
            step=step,
            history=history,
        )

    rows = history["layers.0.moe.tokens_per_expert"]
    assert len(rows) == vector_metrics._MAX_TIMELINE_ROWS


def test_publish_vector_views_skips_ragged_groups() -> None:
    sink = _RecordingSink()
    vector_metrics.publish_vector_views(
        sink,
        {
            "layers.0.moe.tokens_per_expert": [1.0, 2.0],
            "layers.1.moe.tokens_per_expert": [1.0, 2.0, 3.0],
        },
        step=5,
    )
    assert len(sink.histograms) == 2
    assert sink.images == {}


def test_heatmap_draws_gridlines_as_axis_ticks() -> None:
    """Images carry no axes, so rules every eighth cell keep them countable."""

    sink = _RecordingSink()
    width = vector_metrics._GRIDLINE_EVERY * 2
    vector_metrics.publish_vector_views(
        sink,
        {
            f"layers.{layer}.moe.tokens_per_expert": [1.0] * width
            for layer in range(2)
        },
        step=5,
    )

    image = sink.images["layers.all.moe.tokens_per_expert.heatmap"]
    zoom = image.shape[2] // width
    rule = vector_metrics._GRIDLINE_EVERY * zoom
    assert torch.all(image[:, :, rule] == 1.0)
    # Only the boundary is overwritten; the neighbouring cell keeps its colour.
    assert not torch.all(image[:, :, rule + 1] == 1.0)
