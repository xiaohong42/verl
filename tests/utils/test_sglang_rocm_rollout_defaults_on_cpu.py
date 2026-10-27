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
"""SGLang rollout defaults that differ on ROCm."""

import pytest
import torch

from verl.utils.sglang.sglang_quant_utils import prepare_sglang_quantization, set_default_sglang_quantization_backends

FP8_BLOCK = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


@pytest.mark.parametrize("hip", [None, "7.2"], ids=["cuda", "rocm"])
def test_deep_gemm_is_only_the_default_runner_off_rocm(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    engine_kwargs = {}

    set_default_sglang_quantization_backends(FP8_BLOCK, engine_kwargs)

    if hip is None:
        assert engine_kwargs == {"fp8_gemm_runner_backend": "deep_gemm", "moe_runner_backend": "deep_gemm"}
    else:
        # DeepGEMM has no ROCm kernels; SGLang's automatic selection picks AITER / Triton.
        assert engine_kwargs == {}


def test_rocm_launch_keeps_explicit_runners_and_sets_no_deep_gemm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    engine_kwargs = {"moe_runner_backend": "triton"}

    assert prepare_sglang_quantization(dict(FP8_BLOCK), [], engine_kwargs) == FP8_BLOCK
    assert engine_kwargs == {"moe_runner_backend": "triton"}
