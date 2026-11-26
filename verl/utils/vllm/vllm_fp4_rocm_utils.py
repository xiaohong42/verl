# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compressed-tensors MXFP4 routed experts on ROCm, served by AITER.

DeepGEMM's FP4 MoE kernel is NVIDIA-only, and vLLM's ROCm MXFP4 backends cover gpt-oss checkpoints (interleaved
gate/up, SwiGLU-OAI, static FP8 activation scales). verl's compressed-tensors experts (concatenated gate/up, SiLU)
with MXFP8 inputs (``moe=MXFP4``, W4A8) run AITER's a8w4 fused MoE on gfx950. It quantizes both expert GEMM inputs
to FP8 with one E8M0 scale per 32 values, ``2^ceil(log2(amax / 448))``: the grid of the QAT input fake quant.

The checkpoint-layout parameters stay registered, so the initial load and every weight update write into them.
``process_weights_after_loading``, which vLLM's reload runs again after each update, rewrites the AITER tensors in
place, keeping the addresses the captured graphs read.
"""

from functools import cache

import torch

_MXFP4_SF_BLOCK = 32


@cache
def _method_class():
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported
    from vllm.model_executor.utils import set_weight_attrs

    class RocmAiterMxfp4MoEMethod(FusedMoEMethodBase):
        """MXFP4 experts in compressed-tensors names with MXFP8 inputs, run by AITER's a8w4 fused MoE."""

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            if hidden_size % _MXFP4_SF_BLOCK or intermediate_size_per_partition % _MXFP4_SF_BLOCK:
                raise ValueError(
                    f"MXFP4 experts need hidden ({hidden_size}) and intermediate per partition "
                    f"({intermediate_size_per_partition}) sizes divisible by {_MXFP4_SF_BLOCK}"
                )
            w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition
            scale_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.GROUP.value}
            for name, shape, attrs in (
                ("w13_weight_packed", (num_experts, w13_rows, hidden_size // 2), extra_weight_attrs),
                (
                    "w2_weight_packed",
                    (num_experts, hidden_size, intermediate_size_per_partition // 2),
                    extra_weight_attrs,
                ),
                ("w13_weight_scale", (num_experts, w13_rows, hidden_size // _MXFP4_SF_BLOCK), scale_attrs),
                (
                    "w2_weight_scale",
                    (num_experts, hidden_size, intermediate_size_per_partition // _MXFP4_SF_BLOCK),
                    scale_attrs,
                ),
            ):
                param = torch.nn.Parameter(torch.empty(shape, dtype=torch.uint8), requires_grad=False)
                layer.register_parameter(name, param)
                set_weight_attrs(param, attrs)

        def get_fused_moe_quant_config(self, layer):
            return None

        @property
        def is_monolithic(self) -> bool:
            return False

        @property
        def topk_indices_dtype(self):
            return torch.int32

        def process_weights_after_loading(self, layer) -> None:
            self._shuffle_for_aiter(layer)

        @torch.no_grad()
        def _shuffle_for_aiter(self, layer) -> None:
            from aiter import dtypes
            from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

            num_experts = layer.w13_weight_packed.shape[0]
            # a8w4 runs gate/up interleaved (GUGU) for w13, as AITER's a8w4 tests and tuned configs do
            shuffled = {
                "_verl_aiter_w13": shuffle_weight_a16w4(layer.w13_weight_packed.data.view(dtypes.fp4x2), 16, True),
                "_verl_aiter_w2": shuffle_weight_a16w4(layer.w2_weight_packed.data.view(dtypes.fp4x2), 16, False),
                "_verl_aiter_w13_scale": shuffle_scale_a16w4(
                    layer.w13_weight_scale.data.view(dtypes.fp8_e8m0).view(-1, layer.w13_weight_scale.shape[-1]),
                    num_experts,
                    True,
                ),
                "_verl_aiter_w2_scale": shuffle_scale_a16w4(
                    layer.w2_weight_scale.data.view(dtypes.fp8_e8m0).view(-1, layer.w2_weight_scale.shape[-1]),
                    num_experts,
                    False,
                ),
            }
            for name, tensor in shuffled.items():
                live = getattr(layer, name, None)
                if live is None:
                    # first load: keep these tensors (and the layout attributes AITER reads off them)
                    setattr(layer, name, tensor)
                    continue
                if live.shape != tensor.shape or live.dtype != tensor.dtype:
                    raise RuntimeError(
                        f"{name}: reload produced {tuple(tensor.shape)}/{tensor.dtype}, live is "
                        f"{tuple(live.shape)}/{live.dtype}"
                    )
                live.copy_(tensor)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
            from aiter import ActivationType, QuantType, dtypes
            from aiter.fused_moe import fused_moe

            if getattr(layer, "expert_map", None) is not None:
                raise NotImplementedError("AITER a8w4 MXFP4 experts do not support expert parallelism here")
            return fused_moe(
                x,
                layer._verl_aiter_w13,
                layer._verl_aiter_w2,
                topk_weights.to(torch.float32),
                topk_ids.to(torch.int32),
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                w1_scale=layer._verl_aiter_w13_scale,
                w2_scale=layer._verl_aiter_w2_scale,
                quant_dtype_a=dtypes.fp8,
                gate_mode="interleave",
            )

    return RocmAiterMxfp4MoEMethod


def make_rocm_aiter_mxfp4_moe_method(moe):
    """The ROCm MoE method for compressed-tensors MXFP4 experts with MXFP8 inputs (W4A8)."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        raise NotImplementedError(
            "compressed-tensors MXFP4 experts with MXFP8 inputs run AITER's a8w4 fused MoE, which needs gfx950"
        )
    return _method_class()(moe)
