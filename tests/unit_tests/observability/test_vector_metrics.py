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
