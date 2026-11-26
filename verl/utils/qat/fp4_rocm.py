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

"""MXFP4 quantization on ROCm through AITER's MXFP4 quantizer.

TileLang (0.1.10) cannot run ``fp4.py``'s kernels on HIP: ``float8_e8m0fnu`` has no HIP type, the
``float4_e2m1fn`` cast returns wrong values, and the packed FP4 store fails to lower. AITER's quantizer in
its "ceil" scaling mode is the same recipe as ``fp4.py``: one E8M0 scale per 32 values,
``2^ceil(log2(max(amax, 6 * 2^-126) / 6))``, and E2M1 values rounded to nearest even. Quantization and
the QAT fake quant share AITER's quantizer, so the exported weights and the QAT forward agree bit for bit.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

MXFP4_GROUP_SIZE = 32
_BLOCK_M = 32
_BLOCK_N = 256


@triton.jit
def _e2m1_to_fp32(code):
    sign = (code >> 3) & 1
    exponent = ((code >> 1) & 3).to(tl.int32)
    mantissa = (code & 1).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        mantissa * 0.5,
        (1.0 + mantissa * 0.5) * tl.exp2((exponent - 1).to(tl.float32)),
    )
    return tl.where(sign == 1, -magnitude, magnitude)


@triton.jit
def _mxfp4_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    stride_x,
    stride_q,
    stride_s,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT: tl.constexpr,
    DEQUANT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows[:, None] < M
    rows64 = rows[:, None].to(tl.int64)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = row_mask & (cols[None, :] < N)
    x = tl.load(x_ptr + rows64 * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    packed, scale = _mxfp4_quant_op(x, BLOCK_N, BLOCK_M, 32, SCALING_MODE=1)

    if QUANT:
        packed_cols = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        tl.store(
            q_ptr + rows64 * stride_q + packed_cols[None, :],
            packed,
            mask=row_mask & (packed_cols[None, :] < N // 2),
        )
        scale_cols = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
        tl.store(
            s_ptr + rows64 * stride_s + scale_cols[None, :],
            scale,
            mask=row_mask & (scale_cols[None, :] < N // 32),
        )
    if DEQUANT:
        values = tl.interleave(_e2m1_to_fp32(packed & 0xF), _e2m1_to_fp32(packed >> 4))
        # E8M0 -> 2^(e - 127); the ceil scaling never produces e = 0
        scale_fp32 = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        y = tl.reshape(values, (BLOCK_M, BLOCK_N // 32, 32)) * scale_fp32[:, :, None]
        y = tl.reshape(y, (BLOCK_M, BLOCK_N))
        tl.store(y_ptr + rows64 * stride_y + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


def _launch(x2d: torch.Tensor, quant: bool, dequant: bool):
    rows, cols = x2d.shape
    packed = scales = output = None
    if quant:
        packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x2d.device)
        scales = torch.empty((rows, cols // MXFP4_GROUP_SIZE), dtype=torch.uint8, device=x2d.device)
    if dequant:
        output = torch.empty_like(x2d)
    if rows:
        grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(cols, _BLOCK_N))
        _mxfp4_kernel[grid](
            x2d,
            packed,
            scales,
            output,
            rows,
            cols,
            x2d.stride(0),
            packed.stride(0) if quant else 0,
            scales.stride(0) if quant else 0,
            output.stride(0) if dequant else 0,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            QUANT=quant,
            DEQUANT=dequant,
            num_warps=4,
        )
    return packed, scales, output


def fp4_quant(x: torch.Tensor, *, dequant: bool = False):
    """``verl.utils.qat.fp4.fp4_quant`` on ROCm: same arguments, outputs and numerics."""
    cols = x.shape[-1]
    x2d = x.contiguous().view(-1, cols)
    packed, scales, output = _launch(x2d, quant=not dequant, dequant=dequant)
    if dequant:
        return output.view(x.shape)
    return (
        packed.view(torch.float4_e2m1fn_x2).view(*x.shape[:-1], cols // 2),
        scales.view(torch.float8_e8m0fnu).view(*x.shape[:-1], cols // MXFP4_GROUP_SIZE),
    )


def fp4_grouped_fake_quant(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """``verl.utils.qat.fp4.fp4_grouped_fake_quant`` on ROCm: QDQ of equally shaped expert matrices."""
    stacked = torch.stack(tensors)
    num_groups, rows, cols = stacked.shape
    _, _, output = _launch(stacked.view(num_groups * rows, cols), quant=False, dequant=True)
    return output.view(num_groups, rows, cols)
