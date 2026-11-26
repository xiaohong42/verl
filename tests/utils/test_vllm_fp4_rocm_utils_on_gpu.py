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

import pytest
import torch
import torch.nn.functional as F

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("the AITER MXFP4 MoE runs on ROCm GPUs only", allow_module_level=True)
if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
    pytest.skip("AITER's a8w4 MoE runs on gfx950 only", allow_module_level=True)
pytest.importorskip("aiter")
pytest.importorskip("vllm")

from verl.utils.qat.fp4 import fp4_quant  # noqa: E402
from verl.utils.vllm.vllm_fp4_rocm_utils import _method_class  # noqa: E402

# Qwen3-30B-A3B's expert shapes under rollout TP2, with fewer experts
NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 2048, 384, 4


def _mxfp8_qdq(x: torch.Tensor) -> torch.Tensor:
    """The QAT input fake quant: E4M3 with one E8M0 scale 2^ceil(log2(amax / 448)) per 32 values."""
    blocks = x.float().reshape(-1, x.shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scales = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    quantized = (blocks / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (quantized * scales).reshape(x.shape)


def _experts(generator):
    w13 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=generator) * 0.02
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=generator) * 0.02
    return w13.bfloat16(), w2.bfloat16()


def _write_checkpoint(layer: torch.nn.Module, w13: torch.Tensor, w2: torch.Tensor) -> None:
    """Load the compressed-tensors MXFP4 parameters the way vLLM's loader does: in place, in checkpoint layout."""
    for name, weight in (("w13", w13), ("w2", w2)):
        packed, scales = fp4_quant(weight)
        getattr(layer, f"{name}_weight_packed").data.copy_(packed.view(torch.uint8))
        getattr(layer, f"{name}_weight_scale").data.copy_(scales.view(torch.uint8))


def _layer(w13: torch.Tensor, w2: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    for name, rows, cols in (("w13", 2 * INTERMEDIATE, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)):
        packed = torch.empty(NUM_EXPERTS, rows, cols // 2, dtype=torch.uint8, device="cuda")
        scales = torch.empty(NUM_EXPERTS, rows, cols // 32, dtype=torch.uint8, device="cuda")
        layer.register_parameter(f"{name}_weight_packed", torch.nn.Parameter(packed, requires_grad=False))
        layer.register_parameter(f"{name}_weight_scale", torch.nn.Parameter(scales, requires_grad=False))
    _write_checkpoint(layer, w13, w2)
    return layer


def _routing(num_tokens: int, generator):
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", generator=generator)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids.to(torch.int32)


def _reference(x, w13, w2, topk_weights, topk_ids):
    """FP32 MoE on the exactly dequantized experts, with both expert GEMM inputs fake-quantized to MXFP8."""
    w13, w2 = fp4_quant(w13, dequant=True).float(), fp4_quant(w2, dequant=True).float()
    inputs = _mxfp8_qdq(x)
    output = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    for expert in topk_ids.unique().tolist():
        tokens, slots = torch.where(topk_ids == expert)
        gate_up = inputs[tokens] @ w13[expert].T
        hidden = (F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]).bfloat16()
        output.index_add_(0, tokens, (_mxfp8_qdq(hidden) @ w2[expert].T) * topk_weights[tokens, slots, None])
    return output


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


def test_a8w4_moe_matches_reference_and_reloads_in_place():
    generator = torch.Generator(device="cuda").manual_seed(0)
    w13, w2 = _experts(generator)
    layer = _layer(w13, w2)
    cls = _method_class()
    method = cls.__new__(cls)  # the kernel path needs no FusedMoEConfig
    method.process_weights_after_loading(layer)

    x = torch.randn(256, HIDDEN, device="cuda", generator=generator).bfloat16()
    topk_weights, topk_ids = _routing(x.shape[0], generator)
    output = method.apply(layer, x, topk_weights, topk_ids)
    assert output.shape == (x.shape[0], HIDDEN)
    assert _relative_error(output, _reference(x, w13, w2, topk_weights, topk_ids)) < 1e-2

    # a weight update: the loader writes the checkpoint layout in place, then vLLM's reload reruns the processing
    live = {name: getattr(layer, name).data_ptr() for name in ("_verl_aiter_w13", "_verl_aiter_w2")}
    new_w13, new_w2 = _experts(generator)
    _write_checkpoint(layer, new_w13, new_w2)
    method.process_weights_after_loading(layer)
    assert {name: getattr(layer, name).data_ptr() for name in live} == live
    output = method.apply(layer, x, topk_weights, topk_ids)
    assert _relative_error(output, _reference(x, new_w13, new_w2, topk_weights, topk_ids)) < 1e-2
