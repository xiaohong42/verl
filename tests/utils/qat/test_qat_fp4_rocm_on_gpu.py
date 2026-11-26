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

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("the AITER MXFP4 path runs on ROCm GPUs only", allow_module_level=True)
pytest.importorskip("aiter")

from verl.utils.qat import fp4, fp4_rocm  # noqa: E402

_FP4_LEVELS = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)


def _encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    magnitudes = values.abs().unsqueeze(-1)
    levels = _FP4_LEVELS.to(values.device)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    even_codes = (torch.arange(8, device=values.device) % 2 == 0).view(*((1,) * values.ndim), 8)
    codes = torch.where(candidates & even_codes, 1, 0).argmax(dim=-1)
    no_even_tie = ~(candidates & even_codes).any(dim=-1)
    codes = torch.where(no_even_tie, distances.argmin(dim=-1), codes).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def _reference(x: torch.Tensor, dequant: bool):
    cols = x.shape[-1]
    blocks = x.float().reshape(-1, cols // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = amax.clamp_min(6.0 * (2.0**-126)) / 6.0
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    codes = _encode_e2m1((blocks / scales).clamp(-6.0, 6.0))
    logical = _FP4_LEVELS.to(x.device)[(codes & 0x7).long()]
    logical = torch.where((codes & 0x8) != 0, -logical, logical)
    if dequant:
        return (logical * scales).reshape_as(x).to(x.dtype)

    flat_codes = codes.reshape(*x.shape[:-1], cols)
    packed = flat_codes[..., 0::2] | (flat_codes[..., 1::2] << 4)
    quantized = packed.contiguous().view(torch.float4_e2m1fn_x2)
    scale_shape = (*x.shape[:-1], cols // 32)
    return quantized, scales.squeeze(-1).reshape(scale_shape).to(torch.float8_e8m0fnu)


def _input(shape, generator):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 3
    rows = x.view(-1, shape[-1])
    rows[0].zero_()
    rows[1, 7] = 500.0
    ties = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0), device="cuda")
    rows[2, :64] = ties.repeat(8).to(torch.bfloat16)
    return x


# (2, 3, 64): one block per row; (100, 96): partial row and column tiles; (37, 512): two column tiles
@pytest.mark.parametrize("shape", [(2, 3, 64), (100, 96), (37, 512)])
def test_fp4_quant_matches_reference(shape):
    x = _input(shape, torch.Generator(device="cuda").manual_seed(1234))

    quantized, scales = fp4_rocm.fp4_quant(x)
    dequantized = fp4_rocm.fp4_quant(x, dequant=True)
    expected_quantized, expected_scales = _reference(x, dequant=False)

    assert quantized.shape == (*shape[:-1], shape[-1] // 2)
    assert quantized.dtype == torch.float4_e2m1fn_x2
    assert scales.shape == (*shape[:-1], shape[-1] // 32)
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(dequantized, _reference(x, dequant=True))


def test_fp4_grouped_fake_quant_matches_reference():
    generator = torch.Generator(device="cuda").manual_seed(4321)
    experts = tuple(_input((64, 96), generator) for _ in range(4))

    output = fp4_rocm.fp4_grouped_fake_quant(experts)

    assert output.shape == (4, 64, 96)
    assert output.dtype == torch.bfloat16
    for index, expert in enumerate(experts):
        assert torch.equal(output[index], _reference(expert, dequant=True))


def test_fp4_dequant_decodes_exported_weights_exactly():
    x = _input((3, 64, 320), torch.Generator(device="cuda").manual_seed(99))
    quantized, scales = fp4_rocm.fp4_quant(x)
    expected = fp4_rocm.fp4_quant(x, dequant=True)

    assert torch.equal(fp4_rocm.fp4_dequant(quantized, scales), expected)
    out = torch.empty_like(x)
    result = fp4_rocm.fp4_dequant(quantized.view(torch.uint8), scales.view(torch.uint8), out=out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out, expected)
    with pytest.raises(ValueError, match="out must be contiguous"):
        fp4_rocm.fp4_dequant(quantized, scales, out=torch.empty(3, 64, 160, device="cuda", dtype=torch.bfloat16))


def test_fp4_dispatches_to_aiter_on_rocm():
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = _input((8, 128), generator)
    experts = tuple(_input((32, 64), generator) for _ in range(3))

    quantized, scales = fp4.fp4_quant(x)
    expected_quantized, expected_scales = fp4_rocm.fp4_quant(x)

    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(fp4.fp4_quant(x, dequant=True), fp4_rocm.fp4_quant(x, dequant=True))
    assert torch.equal(fp4.fp4_grouped_fake_quant(experts), fp4_rocm.fp4_grouped_fake_quant(experts))
