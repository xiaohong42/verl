# MXFP4 MoE QAT on Blackwell and AMD Instinct

Last updated: 11/26/2026

MXFP4 is the OCP microscaling FP4 format: E2M1 elements with one E8M0 scale per 32-element block. In verl it covers
the routed experts of MoE models. The rollout serves MXFP4 expert weights, quantized on the trainer side, and
DeepGEMM quantizes the expert activations to FP8 at run time. Transformer Engine has no MXFP4 training, so training
stays in BF16; with QAT, the routed experts are fake-quantized during training: their weights exactly as the
exported rollout weights, their inputs to MXFP8, close to the rollout's FP8 activations.

The rollout quantization interface shared by all formats, the kernel defaults and the format details are described
in [Low-Precision Rollout and Quantization-Aware Training](qat.md). For MXFP8 end to end, see
[MXFP8 RL on Blackwell](mxfp8.md).

## Requirements

- Blackwell GPUs for the rollout: the MXFP4-weight × FP8-activation experts are served by vLLM's DeepGEMM FP4 MoE
  kernel (SM100/SM120). Hopper has no such kernel and is outside this recipe; see [qat.md](qat.md). On AMD, gfx950
  (MI350X, MI355X) serves them with AITER instead; see [AMD Instinct (ROCm)](#amd-instinct-rocm).
- Training: the Megatron-Bridge model path; QAT uses NVIDIA ModelOpt.
- Rollout: vLLM (the version verl locks).

## Configuration

```yaml
# MXFP4 rollout experts; dense layers stay in BF16
actor_rollout_ref.rollout.quantization:
  linear: null
  moe: MXFP4
  # moe_activation: bf16   # optional: BF16 expert inputs (W4A16), with qat.mode: mxfp4_w4a16

# optional: MXFP4 QAT on the routed experts
actor_rollout_ref.actor.megatron.qat:
  enable: True
  mode: mxfp4        # mxfp4_mxfp8 also trains the dense layers with MXFP8 QAT (vLLM: linear: MXFP8)
```

Notes:

- `lm_head`, the token embedding, MoE routers and every dense layer stay in BF16 on the rollout and in training.
- ModelOpt installs the fake quantizers; they run verl's TileLang MXFP4 kernel, the one the trainer-side exporter
  uses without dequantization, so the QAT forward and the exported rollout weights agree bit for bit (see
  [qat.md](qat.md)). On ROCm, where TileLang cannot run these MXFP4 kernels, both go to AITER's MXFP4 quantizer in
  its ceil scaling mode, the same recipe bit for bit (`verl/utils/qat/fp4_rocm.py`). The experts' inputs are
  fake-quantized to MXFP8 (1x32 blocks), while the rollout's DeepGEMM kernel quantizes them to FP8 in 1x128 groups
  with E8M0 scales.
- QAT quantizes the Transformer Engine modules of the layer spec that Megatron-Bridge builds for the model, the
  grouped expert linears included, so tensor and expert parallelism can be combined. All expert weights of a grouped
  linear are fake-quantized in one kernel launch.
- QAT and Transformer Engine FP8 training are mutually exclusive; the engine refuses the combination.
- `moe_activation: bf16` serves the MXFP4 experts with BF16 inputs (W4A16), and the QAT mode `mxfp4_w4a16`
  fake-quantizes only their weights; verl refuses a QAT mode that does not match the rollout quantization. The W4A16
  rollout is implemented for ROCm (see [AMD Instinct (ROCm)](#amd-instinct-rocm)); on CUDA such experts are left to
  vLLM's compressed-tensors MXFP4 MoE method, untested with verl.

## Example: Qwen3-30B-A3B on DAPO-Math-17k

[`examples/mxfp4/run_qwen3_30b_a3b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp4/run_qwen3_30b_a3b_megatron.sh)
is the MXFP4 counterpart of the MXFP8 Qwen3-30B-A3B example, with the same data and hyper-parameters on one
4-GPU Blackwell node (tested on 4×B300; 4×B200 untested): training TP2 × EP2 with full recompute and the whole
optimizer on the CPU, the rollout vLLM with TP2. `PRECISION` selects the arm: `mxfp4_qat` (MXFP4 rollout experts and
MXFP4 QAT), `mxfp4` (MXFP4 rollout experts only) and `bf16` (the baseline). The script reads DAPO-Math-17k in verl's
format from `TRAIN_FILE`, as
[verl-recipe's `dapo/prepare_dapo_data.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/prepare_dapo_data.sh)
downloads it, and runs without a validation set.

```bash
PRECISION=mxfp4_qat TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=mxfp4 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
PRECISION=bf16 TOTAL_TRAINING_STEPS=10 bash examples/mxfp4/run_qwen3_30b_a3b_megatron.sh
```

### Results

**Configuration**

- One 4×B300 machine for every arm, the same seed and data order, so steps pair across arms; 10 steps
- The script's hyper-parameters, as in the MXFP8 example: 64 prompts × 8 responses per step, prompts and responses up
  to 1024 and 8192 tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6; DAPO-Math-17k deduplicated
  to its 17,398 unique prompts
- verl main `e41dd697` plus this PR; the versions verl locks: vLLM 0.32.0, Transformer Engine
  2.20.2, Megatron-Core 0.19.2, Megatron-Bridge 0.6.2, ModelOpt 0.47.0, TileLang 0.1.15

"X/Y" is training precision X, rollout precision Y; BF16+QAT trains in BF16 with MXFP4 QAT on the routed experts.

| mean of steps 1–10 | BF16/BF16 | BF16/MXFP4 | BF16+QAT/MXFP4 |
|---|---|---|---|
| step time (s) | 827 | 848 | 1000 |
| generation (s) | 107 | 114 | 115 |
| `old_log_prob` (s) | 118 | 118 | 149 |
| `ref` (s) | 113 | 117 | 130 |
| `update_actor` (s) | 476 | 477 | 582 |
| `update_weights` (s) | 8 | 18 | 19 |
| `rollout_corr/kl` | 0.0018 | 0.0150 | 0.0030 |
| score | −0.195 | −0.161 | −0.183 |
| response length | 6517 | 6411 | 6438 |

The QAT arm's steps 1–4 overlapped another tenant on the same node (the machine's other four GPUs at 100 %), so they
are slower than the rest; its steps 5–10 ran on an idle machine. The same six steps for all three arms:

| mean of steps 5–10 | BF16/BF16 | BF16/MXFP4 | BF16+QAT/MXFP4 |
|---|---|---|---|
| step time (s) | 818 | 833 | 900 |
| `old_log_prob` (s) | 117 | 116 | 135 |
| `update_actor` (s) | 472 | 468 | 511 |
| `rollout_corr/kl` | 0.0018 | 0.0153 | 0.0030 |

![Qwen3-30B-A3B on DAPO-Math-17k, MXFP4 and QAT, per step](https://github.com/user-attachments/assets/e1ca6340-8646-4e11-abd5-d5413275d26b)
*Per step on one 4×B300 machine; the shaded steps of the QAT arm shared the node. wandb:
[QAT](https://wandb.ai/train0z5a/verl_grpo_dapo_mxfp4/runs/z52e8qr2),
[MXFP4](https://wandb.ai/train0z5a/verl_grpo_dapo_mxfp4/runs/mzx5r4ys),
[BF16](https://wandb.ai/train0z5a/verl_grpo_dapo_mxfp4/runs/xd7yxpnw)*

Results and observations:

- QAT cuts the rollout–trainer mismatch of the MXFP4 rollout by 80 %, to 1.7× the BF16 arm's. Per step,
  `rollout_corr/kl` ranged 0.0028–0.0033 with QAT, 0.0135–0.0173 without it and 0.0016–0.0020 in BF16.
- On the six idle steps QAT costs 8 % over the MXFP4 rollout alone (900 s vs 833 s; `old_log_prob` +19 s,
  `update_actor` +43 s); the ten-step means above include four contended QAT steps and show 18 %.
- The TileLang grouped export of the expert weights is why the MXFP4 arms' `update_weights` is 18–19 s here; the
  earlier per-expert export cost 45–46 s.
- The MXFP4 rollout does not speed this job up: generation takes about as long as in BF16 (+6 s) and the expert
  export adds about 10 s per step.
- Ten steps are enough for speed and mismatch, not for the score.

## AMD Instinct (ROCm)

On ROCm the trainer-side MXFP4 quantization runs AITER's MXFP4 quantizer (see the notes above), and the rollout
serves MXFP4 experts with MXFP8 inputs (`moe: MXFP4`) through AITER's a8w4 fused MoE instead of DeepGEMM. AITER
quantizes both expert GEMM inputs to FP8 in 1x32 blocks with E8M0 scales, the grid of the QAT input fake quant. The
expert weights stay registered in their compressed-tensors checkpoint layout: vLLM's reload writes each weight update
into them and then refreshes AITER's shuffled copies in place.

- GPUs: gfx950 (MI350X, MI355X). AITER's a8w4 MoE does not compile on gfx942, and verl refuses the MXFP4 × MXFP8
  rollout there.
- W4A16 (`moe_activation: bf16`): AITER's a16w4 SiLU MoE does not keep BF16 activations on gfx950, so the rollout
  dequantizes the experts to BF16, exactly, and runs vLLM's BF16 MoE kernel. W4A16 then matches the QAT numerics
  but runs at the BF16 rollout's speed.
- ModelOpt's QAT turns off fused wgrad accumulation on the quantized expert linears, so Transformer Engine runs them
  on its Triton grouped GEMM where `NVTE_USE_GROUPED_GEMM_TRITON=1`, as verl's ROCm image sets. Before Transformer
  Engine 2.21 its gfx950 configs for these shapes need more shared memory than the GPU has; the example script sets
  the variable to `0` for its QAT arms on ROCm, so they run on TE's default grouped GEMM, like the BF16 arm.

### Results on MI350X

**Configuration**

- The example script on 4×MI350X per arm, the same seed and data order: the BF16 arm on one host, the two QAT arms on
  the two halves of another host (sharing its CPUs), so only the QAT arms' step times are compared
- The script's defaults plus token-level TIS (`algorithm.rollout_correction.rollout_is=token`), R3 router replay
  (`actor_rollout_ref.actor.megatron.router_replay.mode=R3`, `actor_rollout_ref.rollout.enable_rollout_routing_replay=True`)
  and `actor_rollout_ref.rollout.free_cache_engine=False`: on some of these hosts, vLLM's sleep mode fails in
  `hipMemUnmap` with `invalid argument`, a HIP runtime regression in the image's ROCm 10.2 nightly
  (ROCm/rocm-systems#11606, fixed by ROCm/rocm-systems#13562); `rocm/primus:v26.10` has the fix, and the script's
  default works there;
  DAPO-Math-17k as `prepare_dapo_data.sh` downloads it, not deduplicated
- `rocm/primus:v26.9` (ROCm 10.2 nightly, PyTorch 2.16 pre-release) with vLLM 0.34.0, Transformer Engine 2.20.0.dev0,
  Megatron-Core 0.19.2, Megatron-Bridge 0.6.2, ModelOpt 0.48.0 and AITER 0.1.27; verl main `de742368` with these
  changes, before weight updates moved onto vLLM's reload (#8780)

| mean of steps 2–119 | BF16/BF16 | BF16+QAT/MXFP4 W4A8 | BF16+QAT/MXFP4 W4A16 |
|---|---|---|---|
| `rollout_corr/kl` | 0.00073 | 0.0024 | 0.00073 |
| tokens with TIS ratio > 2 | 2.4e-5 | 4.2e-4 | 2.5e-5 |
| score | 0.33 | 0.31 | 0.31 |
| entropy | 0.135 | 0.143 | 0.133 |
| generation (s) | – | 146 | 172 |
| step time (s) | – | 812 | 837 |

- The W4A8 arm's `rollout_corr/kl` is 3.3× the BF16 arm's and rises slowly over the run (0.0017–0.0029 per step;
  0.0021 on average over steps 2–58, 0.0026 over steps 50–119). Score, entropy and response length follow the BF16
  arm with no divergence, through step 119 (the BF16 arm's last) and to the W4A8 arm's last step, 127.
- The W4A16 arm's mismatch equals the BF16 arm's: its rollout experts are exactly the QAT forward's weights.
- On the same host, W4A8 generation takes 15 % less time than W4A16 (BF16 kernels), and the whole step 3 % less.
