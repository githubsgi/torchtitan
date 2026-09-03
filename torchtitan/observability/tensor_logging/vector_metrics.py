# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-width vector statistics logged alongside scalar tensor statistics.

The scalar path in `statistics.py` collapses every observed tensor into seven
SUM fields plus one MAX field, so per-element structure is gone by the time
anything is reduced. Some quantities need that structure kept. MoE expert load
is a length-`num_experts` vector, and its imbalance ratio
`max(counts) / mean(counts)` cannot be recovered once the counts have been
folded into a generic population.

This module adds a parallel slab that does not collapse: one FP32 row of fixed
width per registered vector metric, combined across ranks with one extra SUM
all-reduce. It reuses the scalar path's registration walk and cadence gate but
keeps its own registry attribute and its own buffers, so enabling one does not
allocate rows for the other.

Owner-writes semantics
----------------------
A rank writes only the columns it owns and leaves the rest zero, so SUM
reconstructs the exact global vector. `runtime.py` already applies this rule
across pipeline stages ("the stage that owns a metric writes its row, other
stages contribute zero to additive fields"); here it extends to the expert
axis, where EP gives each rank a disjoint set of experts and CP gives each rank
a disjoint shard of tokens over the same experts.

Reducing by element rather than as one population is the whole point. For two
CP ranks observing expert counts `[10, 10]` and `[0, 20]`, treating the four
numbers as a single population gives `20 / 10 = 2.0`, while summing by expert
first gives `[10, 30]` and the correct ratio `30 / 20 = 1.5`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .runtime import (
    _discover_registered_metrics,
    _gather_pipeline_metric_names,
    _get_state,
    _is_activation_checkpoint_recompute,
    _local_tensor,
    _register_names,
    MetricSource,
    should_run_logging_calls,
)


# Kept separate from the scalar registry so a source can carry both without
# either path allocating rows for the other's names.
_REGISTERED_VECTOR_NAMES_ATTR = "_tensor_logging_registered_vector_names"

# Maps a registered short name to its row in the shared vector slab.
_VECTOR_BUFFER_SLOTS_ATTR = "_tensor_logging_vector_buffer_slots"

_active_state: VectorLoggingState | None = None


class VectorBuffers(nn.Module):
    """One fixed-width FP32 row per registered vector metric.

    Rows are padded to the widest registered metric so the slab is a single
    dense tensor and the cross-rank combine is one collective. Padding columns
    stay zero on every rank, so they survive SUM unchanged and are sliced off
    during collection.
    """

    values: torch.Tensor

    def __init__(
        self,
        metric_count: int,
        vector_width: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "values",
            torch.zeros(
                (metric_count, vector_width),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )

    def clear(self) -> None:
        self.values.zero_()


def register_vector(
    metric_source: MetricSource,
    registered_names: Sequence[str],
    *,
    width: int,
) -> None:
    """Register fixed-width vector names before logging is initialized.

    Args:
        metric_source: Module or parameter passed again at the logging callsite.
        registered_names: Local names accepted by `log_vector` for this source.
        width: Element count every logged tensor for these names must have.

    Example:

        register_vector(moe, ["tokens_per_expert"], width=num_experts)
        log_vector(moe, tokens_per_expert=num_local_tokens_per_expert_E)
    """

    if _active_state is not None:
        raise RuntimeError("register vector names before tensor_logging.init()")
    if width <= 0:
        raise ValueError(f"vector metric width must be positive, got {width}")
    _register_names(
        metric_source,
        [f"{name}@{width}" for name in registered_names],
        names_attr=_REGISTERED_VECTOR_NAMES_ATTR,
    )


def _split_width(registered_name: str) -> tuple[str, int]:
    """Split the `name@width` form produced by `register_vector`."""

    name, _, width = registered_name.rpartition("@")
    return name, int(width)


def _derive_metrics_from_vector(
    metric_name: str,
    values: list[float],
    *,
    publish_elements: bool,
) -> dict[str, int | float]:
    """Reduce one global vector to publishable scalars.

    `imbalance` is `max / mean`: 1.0 when perfectly balanced, and `width` in the
    degenerate case where a single element holds everything. It is meaningful
    only because the vector was summed element-wise first.
    """

    metrics: dict[str, int | float] = {}
    prefix = f"{metric_name}."
    total = sum(values)
    width = len(values)
    maximum = max(values)
    metrics[prefix + "total"] = total
    metrics[prefix + "max"] = maximum
    metrics[prefix + "min"] = min(values)
    mean = total / width
    metrics[prefix + "mean"] = mean
    if mean > 0.0:
        metrics[prefix + "imbalance"] = maximum / mean
    if publish_elements:
        for index, value in enumerate(values):
            metrics[f"{prefix}e{index}"] = value
    return metrics


def _replication_owner_factor(value: torch.Tensor) -> float:
    """Return 1.0 if this rank should contribute `value`, else 0.0.

    A WORLD SUM is only correct when each rank holds a distinct piece of the
    quantity. Shard and Partial placements satisfy that. Replicate does not: on
    a mesh axis where the tensor is replicated, every rank holds the same
    numbers, and summing them would scale the result by that axis's size.

    Applying owner-writes here makes the single WORLD collective valid for both
    cases: one rank per replicated axis contributes, the rest contribute zero.
    For example, MoE expert counts are Partial under sequence parallelism but
    Replicate under plain TP, and the same callsite must be correct in both.

    Plain tensors carry no placement information, so they are assumed
    rank-distinct. That holds for DP and CP, where every rank sees different
    tokens.
    """

    if not isinstance(value, DTensor):
        return 1.0

    mesh = value.device_mesh
    for axis, placement in enumerate(value.placements):
        if placement.is_replicate() and mesh.get_local_rank(axis) != 0:
            return 0.0
    return 1.0


class VectorLoggingState:
    """Hold the names and buffers for one active vector-logging run.

    Example:

        register_vector(moe, ["tokens_per_expert"], width=8)
        state = VectorLoggingState(model, device=device)

        log_vector(moe, tokens_per_expert=counts_E)
        metrics = state.collect()
        state.close()
    """

    def __init__(
        self,
        model_parts: Sequence[nn.Module],
        *,
        device: torch.device,
        publish_filter_regex: str = "",
        publish_elements: bool = False,
        pp_enabled: bool = False,
    ) -> None:
        self._publish_filter = (
            re.compile(publish_filter_regex) if publish_filter_regex else None
        )
        self._publish_elements = publish_elements

        registered_metrics = _discover_registered_metrics(
            model_parts,
            pp_enabled=pp_enabled,
            names_attr=_REGISTERED_VECTOR_NAMES_ATTR,
        )

        # Give every PP rank the same row order for the packed reduction. The
        # width travels inside the name so ranks that own none of a metric
        # still agree on its column count.
        decorated_names = _gather_pipeline_metric_names(
            {full_name for _, _, full_name in registered_metrics},
            pp_enabled=pp_enabled,
        )
        self.full_metric_names: list[str] = []
        self.widths: list[int] = []
        for decorated_name in decorated_names:
            name, width = _split_width(decorated_name)
            self.full_metric_names.append(name)
            self.widths.append(width)

        row_by_decorated_name = {
            decorated_name: row for row, decorated_name in enumerate(decorated_names)
        }

        # A single dense slab keeps the combine to one collective regardless of
        # how many metrics or how ragged their widths are.
        slab_width = max(self.widths, default=1)
        self.vector_buffers = VectorBuffers(
            len(decorated_names),
            slab_width,
            device=device,
        )
        self._row_indices = torch.arange(len(decorated_names), dtype=torch.int64)

        slots_by_source: dict[MetricSource, dict[str, torch.Tensor]] = {}
        for metric_source, registered_name, full_name in registered_metrics:
            source_slots = slots_by_source.setdefault(metric_source, {})
            name, _ = _split_width(registered_name)
            source_slots[name] = self._row_indices[row_by_decorated_name[full_name]]

        for metric_source, source_slots in slots_by_source.items():
            setattr(metric_source, _VECTOR_BUFFER_SLOTS_ATTR, source_slots)
        self._metric_sources = list(slots_by_source)

        self._buffer_owner = model_parts[0]
        self._buffer_owner.add_module("_tensor_logging_vectors", self.vector_buffers)

        # Share the scalar path's device-side cadence gate rather than adding a
        # second one. Under compile and CUDA graphs the accumulation stays in
        # the graph on every step, so an off-cadence step must be neutralized
        # arithmetically; a Python bool would be baked in at capture time.
        self._enabled = _get_state().statistic_buffers.enabled
        self._closed = False

    def _accumulate(
        self,
        row: torch.Tensor,
        value: torch.Tensor,
        owner_factor: float,
    ) -> None:
        value = value.detach().to(
            dtype=torch.float32,
            device=self.vector_buffers.values.device,
        )
        self.vector_buffers.values[row, : value.numel()] += (
            value * self._enabled * owner_factor
        )

    def _reduce_buffers(self) -> torch.Tensor:
        """Clone and SUM the whole slab in one collective."""

        values = self.vector_buffers.values.clone()
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

    def _buffers_to_metrics(self, reduced: torch.Tensor) -> dict[str, int | float]:
        # One device-to-host copy avoids synchronizing per metric.
        rows = cast(list[list[float]], reduced.detach().cpu().tolist())
        metrics: dict[str, int | float] = {}
        for row, metric_name in enumerate(self.full_metric_names):
            metrics.update(
                _derive_metrics_from_vector(
                    metric_name,
                    rows[row][: self.widths[row]],
                    publish_elements=self._publish_elements,
                )
            )
        # Filtering controls what gets logged, not what gets computed.
        if self._publish_filter is not None:
            metrics = {
                name: value
                for name, value in metrics.items()
                if self._publish_filter.search(name)
            }
        return metrics

    def collect(self) -> dict[str, int | float]:
        """Reduce, derive, and reset vectors from this training step."""

        metrics = self._buffers_to_metrics(self._reduce_buffers())
        self.vector_buffers.clear()
        return metrics

    def collect_vectors(self) -> dict[str, list[float]]:
        """Return the reduced global vectors themselves, without resetting.

        This is the non-collapsing surface: consumers that need the full
        distribution -- histogram sinks, per-expert GEMM shape records -- read
        it here instead of reconstructing it from published scalars.
        """

        rows = cast(list[list[float]], self._reduce_buffers().detach().cpu().tolist())
        return {
            metric_name: rows[row][: self.widths[row]]
            for row, metric_name in enumerate(self.full_metric_names)
        }

    def close(self) -> None:
        global _active_state
        if self._closed:
            return
        self._closed = True
        for metric_source in self._metric_sources:
            metric_source.__dict__.pop(_VECTOR_BUFFER_SLOTS_ATTR, None)
        self._metric_sources.clear()
        self._buffer_owner._modules.pop("_tensor_logging_vectors")
        if _active_state is self:
            _active_state = None


def init_vectors(
    model_parts: nn.Module | Iterable[nn.Module],
    *,
    device: torch.device,
    publish_filter_regex: str = "",
    publish_elements: bool = False,
    pp_enabled: bool = False,
) -> VectorLoggingState:
    """Assign registered vector names fixed slab rows and activate logging.

    Call after `tensor_logging.init()`. The two paths share one cadence gate, so
    a single `set_enabled()` scope governs both.
    """

    global _active_state
    if _active_state is not None:
        raise RuntimeError("vector logging already has active state")
    model_part_list = (
        [model_parts] if isinstance(model_parts, nn.Module) else list(model_parts)
    )
    state = VectorLoggingState(
        model_part_list,
        device=device,
        publish_filter_regex=publish_filter_regex,
        publish_elements=publish_elements,
        pp_enabled=pp_enabled,
    )
    _active_state = state
    return state


def log_vector(
    metric_source: MetricSource,
    **named_tensors: torch.Tensor,
) -> None:
    """Accumulate current-pass vectors for registered named tensors.

    Args:
        metric_source: Module or parameter used during registration.
        **named_tensors: Registered names mapped to their current 1-D tensors.

    Example:

        log_vector(moe, tokens_per_expert=num_local_tokens_per_expert_E)
    """

    if _active_state is None:
        return
    if _is_activation_checkpoint_recompute():
        return
    if not should_run_logging_calls():
        return

    slots = metric_source.__dict__.get(_VECTOR_BUFFER_SLOTS_ATTR)
    if slots is None:
        raise KeyError(
            f"no initialized vector metrics on {type(metric_source).__name__}"
        )
    slots = cast(dict[str, torch.Tensor], slots)
    for registered_name, value in named_tensors.items():
        try:
            row = slots[registered_name]
        except KeyError:
            raise KeyError(f"unregistered vector metric: {registered_name}") from None
        owner_factor = _replication_owner_factor(value)
        _active_state._accumulate(
            row,
            _local_tensor(value).reshape(-1),
            owner_factor,
        )
