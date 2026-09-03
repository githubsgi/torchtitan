# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
import torch
from torch import nn

from torchtitan.observability import moe_shapes


class _StockExperts(nn.Module):
    """Parameter layout of the stock GroupedExperts: separate gate and up."""

    def __init__(self, num_experts: int, hidden_dim: int, dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w1_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))


class _GptOssExperts(nn.Module):
    """Parameter layout of GptOssGroupedExperts: gate and up fused into mlp1."""

    def __init__(self, num_experts: int, hidden_dim: int, dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.mlp1_weight_EGD = nn.Parameter(
            torch.empty(num_experts, hidden_dim * 2, dim)
        )


def _moe(experts: nn.Module, top_k: int = 2) -> nn.Module:
    from torchtitan.models.common.moe import MoE

    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.routed_experts = nn.Module()
    moe.routed_experts.inner_experts = experts
    moe.router = nn.Module()
    moe.router.top_k = top_k
    return moe


def _model(moe: nn.Module) -> nn.Module:
    root = nn.Module()
    root.layers = nn.Module()
    root.layers.moe = moe
    return root


def test_stock_experts_report_three_grouped_gemms() -> None:
    model = _model(_moe(_StockExperts(num_experts=4, hidden_dim=16, dim=8)))

    records = moe_shapes.collect_moe_shapes([model])

    assert len(records) == 1
    record = records[0]
    assert record.module_path == "layers.moe"
    assert record.num_experts == 4
    assert record.dim == 8
    assert record.hidden_dim == 16
    assert record.top_k == 2
    assert [(g.name, g.k, g.n) for g in record.grouped_gemms] == [
        ("w1", 8, 16),
        ("w3", 8, 16),
        ("w2", 16, 8),
    ]


def test_gpt_oss_experts_report_fused_mlp1() -> None:
    """The fused gate+up GEMM is one GEMM of width 2F, not two of width F."""
    model = _model(_moe(_GptOssExperts(num_experts=4, hidden_dim=16, dim=8)))

    record = moe_shapes.collect_moe_shapes([model])[0]

    assert record.dim == 8
    assert record.hidden_dim == 16
    assert [(g.name, g.k, g.n) for g in record.grouped_gemms] == [
        ("mlp1", 8, 32),
        ("mlp2", 16, 8),
    ]


def test_unknown_expert_layout_is_rejected() -> None:
    class _Unknown(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 4

    model = _model(_moe(_Unknown()))

    with pytest.raises(TypeError, match="unrecognized expert parameter layout"):
        moe_shapes.collect_moe_shapes([model])


def test_multiple_model_parts_are_prefixed() -> None:
    parts = [
        _model(_moe(_StockExperts(num_experts=2, hidden_dim=4, dim=2))),
        _model(_moe(_StockExperts(num_experts=2, hidden_dim=4, dim=2))),
    ]

    paths = [record.module_path for record in moe_shapes.collect_moe_shapes(parts)]

    assert paths == ["model_parts.0.layers.moe", "model_parts.1.layers.moe"]


def test_manifest_round_trips_to_json(tmp_path) -> None:
    model = _model(_moe(_StockExperts(num_experts=4, hidden_dim=16, dim=8)))
    path = tmp_path / "nested" / "moe_shapes.json"

    moe_shapes.write_moe_shape_manifest(moe_shapes.collect_moe_shapes([model]), path)

    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    layer = payload["moe_layers"][0]
    assert layer["num_experts"] == 4
    assert layer["grouped_gemms"][0] == {"name": "w1", "k": 8, "n": 16}
