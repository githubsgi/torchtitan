# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static grouped-GEMM shape manifest for MoE layers.

Of a grouped GEMM's `(M, K, N)`, only `M` varies per step: it is the number of
tokens routed to each expert. `K` and `N` are fixed by the model config, so
they need no runtime instrumentation and no collective -- one walk of the built
model at setup time records them exactly.

Pairing this manifest with the per-expert token-count vector from
`tensor_logging.vector_metrics` gives complete `(M, K, N)` coverage for every
grouped GEMM: the manifest supplies `(K, N)` once, and the vector supplies the
`M` distribution each logging step.

The three grouped GEMMs inside `GroupedExperts.forward` are:

    w1: [R, D] @ [D, F] -> [R, F]   (K = dim,        N = hidden_dim)
    w3: [R, D] @ [D, F] -> [R, F]   (K = dim,        N = hidden_dim)
    w2: [R, F] @ [F, D] -> [R, D]   (K = hidden_dim, N = dim)

where R is the routed-token count for one expert, i.e. that expert's `M`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GroupedGemmShape:
    """One grouped GEMM's static operand shape."""

    name: str
    k: int
    n: int


@dataclass(frozen=True, slots=True)
class MoEShapeRecord:
    """Static grouped-GEMM shapes for one MoE layer."""

    module_path: str
    num_experts: int
    dim: int
    hidden_dim: int
    top_k: int
    grouped_gemms: list[GroupedGemmShape]


def _weight(experts: nn.Module, name: str) -> torch.Tensor | None:
    """Return the named expert weight, or None if the module does not have it.

    `nn.Module.__getattr__` returns a submodule-or-tensor union, so narrow to a
    tensor here and let callers dispatch on presence.
    """

    weight = getattr(experts, name, None)
    return weight if isinstance(weight, torch.Tensor) else None


def _expert_shapes(experts: nn.Module) -> tuple[int, int, list[GroupedGemmShape]]:
    """Return `(dim, hidden_dim, shapes)` for one built expert module.

    Dispatches on the parameters actually present rather than on the class, so
    a converter that swaps an expert module is described by what it will run.

    The stock `GroupedExperts` issues three grouped GEMMs, keeping the gate and
    up projections separate. `GptOssGroupedExperts` subclasses it but fuses
    those two into a single GEMM of width `2 * hidden_dim`, so it issues two.
    """

    w1_EFD = _weight(experts, "w1_EFD")
    if w1_EFD is not None:
        _, hidden_dim, dim = w1_EFD.shape
        return (
            dim,
            hidden_dim,
            [
                GroupedGemmShape(name="w1", k=dim, n=hidden_dim),
                GroupedGemmShape(name="w3", k=dim, n=hidden_dim),
                GroupedGemmShape(name="w2", k=hidden_dim, n=dim),
            ],
        )

    mlp1_weight_EGD = _weight(experts, "mlp1_weight_EGD")
    if mlp1_weight_EGD is not None:
        # mlp1 is [num_experts, 2 * hidden_dim, dim]: gate and up are fused.
        _, gate_up, dim = mlp1_weight_EGD.shape
        hidden_dim = gate_up // 2
        return (
            dim,
            hidden_dim,
            [
                GroupedGemmShape(name="mlp1", k=dim, n=gate_up),
                GroupedGemmShape(name="mlp2", k=hidden_dim, n=dim),
            ],
        )

    raise TypeError(
        f"unrecognized expert parameter layout on {type(experts).__name__}; "
        "add its grouped-GEMM shapes to _expert_shapes"
    )


def collect_moe_shapes(model_parts: Sequence[nn.Module]) -> list[MoEShapeRecord]:
    """Walk built model parts and record each MoE layer's static GEMM shapes.

    Reads the built modules rather than the config tree so the result reflects
    what will actually run, including any converter that replaced an expert
    module after construction.
    """

    from torchtitan.models.common.moe import MoE

    records: list[MoEShapeRecord] = []
    for model_part_index, model_part in enumerate(model_parts):
        prefix = "" if len(model_parts) == 1 else f"model_parts.{model_part_index}"
        for module_name, module in model_part.named_modules():
            if not isinstance(module, MoE):
                continue
            module_name = ".".join(
                part
                for part in module_name.split(".")
                if part != "_checkpoint_wrapped_module"
            )
            module_path = ".".join(part for part in (prefix, module_name) if part)

            experts = module.routed_experts.inner_experts
            # Read built parameters rather than the config tree so a TP-sharded
            # or converted module reports the shape it will actually run.
            dim, hidden_dim, grouped_gemms = _expert_shapes(experts)
            records.append(
                MoEShapeRecord(
                    module_path=module_path,
                    num_experts=experts.num_experts,
                    dim=dim,
                    hidden_dim=hidden_dim,
                    top_k=module.router.top_k,
                    grouped_gemms=grouped_gemms,
                )
            )
    return records


def write_moe_shape_manifest(
    records: Sequence[MoEShapeRecord],
    path: str | Path,
) -> None:
    """Write the manifest as JSON. Written once per run, not per step."""

    payload: dict[str, Any] = {
        "version": 1,
        "moe_layers": [asdict(record) for record in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
