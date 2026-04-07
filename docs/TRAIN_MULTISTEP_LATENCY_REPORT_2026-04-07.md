# Train Multi-step Offline Latency Report (2026-04-07)

## Scope

- Task: formal offline latency benchmark on Ascend NPU with real Bench2Drive samples
- Split: `train`
- Dataset source: local `Bench2Drive Mini` routes prepared into MindDrive info files
- Benchmark path: direct model inference on dataset samples, not `MinddriveAgent.run_step()`
- Goal: measure latency and check whether outputs remain basically reasonable against GT

## Why `train`

The local `Bench2Drive Mini` subset currently contains train routes only.

- `data/infos/b2d_infos_train.pkl`: 2295 samples
- `data/infos/b2d_infos_val.pkl`: 0 samples

So the formal offline benchmark was run on `train`.

## Command

```bash
source scripts/env_minddrive_b2d.sh
"${MINDDRIVE_PYTHON}" scripts/benchmark_minddrive_latency_offline.py \
  --split train \
  --steps 20 \
  --warmup-steps 4 \
  --sample-pool-size 20 \
  --start-index 0 \
  --output-dir results_latency_offline_train_steps20
```

## Environment

- Device: `npu`
- Visible device: `2`
- Device count seen by benchmark: `1`
- Config: `adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`
- Checkpoint: `ckpts/minddrive_rltrain.pth`

## Result Summary

### 1. System latency

This mode includes end-to-end sample fetch + collate + device transfer + model inference + postprocess.

- Total steps: `20`
- Warmup steps: `4`
- Effective measured steps: `16`
- Wall time: `41.541 s`
- E2E mean: `1396.061 ms`
- E2E p50: `1377.440 ms`
- E2E p90: `1458.431 ms`
- E2E p95: `1470.584 ms`
- Model mean: `634.962 ms`
- Prepare mean: `760.521 ms`
- Transfer mean: `4.168 ms`
- Post mean: `0.579 ms`

Interpretation:

- End-to-end latency is about `1.40 s / sample`
- Pure model forward is about `0.635 s`
- The main extra overhead in system mode comes from data/sample preparation, not postprocess

### 2. Pure inference latency

This mode reuses already prepared inputs and focuses on device transfer + model inference + postprocess.

- Total steps: `20`
- Warmup steps: `4`
- Effective measured steps: `16`
- Wall time: `12.788 s`
- E2E mean: `638.471 ms`
- E2E p50: `637.682 ms`
- E2E p90: `645.873 ms`
- E2E p95: `649.840 ms`
- Model mean: `634.439 ms`
- Prepare mean: `3.465 ms`
- Transfer mean: `3.464 ms`
- Post mean: `0.567 ms`

Interpretation:

- Pure inference is stable around `0.64 s / sample`
- Most of the latency gap between system mode and pure inference comes from real-data loading / collation / preparation

## Sanity and GT Reasonableness

Thresholds used by the benchmark:

- max ego FDE: `20.0 m`
- max path FDE: `25.0 m`
- max absolute trajectory magnitude: `150.0 m`

### System latency sanity

- Basic sanity pass: `16 / 16`
- GT reasonableness pass: `14 / 16`
- Ego FDE mean: `6.718 m`
- Ego FDE p95: `20.273 m`
- Ego FDE max: `20.878 m`
- Path FDE mean: `0.056 m`
- Path FDE p95: `0.067 m`
- Path FDE max: `0.074 m`

### Pure inference sanity

- Basic sanity pass: `16 / 16`
- GT reasonableness pass: `14 / 16`
- Ego FDE mean: `6.809 m`
- Ego FDE p95: `20.323 m`
- Ego FDE max: `20.891 m`
- Path FDE mean: `0.053 m`
- Path FDE p95: `0.063 m`
- Path FDE max: `0.072 m`

### Failed GT-reasonableness cases

Two effective samples slightly exceeded the `20.0 m` ego FDE threshold in both modes:

- sample `8`: ego FDE about `20.07 m` to `20.13 m`
- sample `9`: ego FDE about `20.88 m` to `20.89 m`

Notes:

- All `16 / 16` effective samples passed basic sanity checks
- Path prediction stayed very close to GT in all measured samples
- The two GT failures are threshold overflows on ego FDE, not numerical explosions or invalid tensors

## Artifact Files

- `results_latency_offline_train_steps20/combined_summary.json`
- `results_latency_offline_train_steps20/system_latency_summary.json`
- `results_latency_offline_train_steps20/pure_inference_latency_summary.json`
- `results_latency_offline_train_steps20/system_latency_records.json`
- `results_latency_offline_train_steps20/pure_inference_latency_records.json`

## Caveats

- This is an offline benchmark on real dataset samples, not closed-loop CARLA evaluation
- The benchmark is on `train`, because the prepared Mini subset has no val samples locally
- Checkpoint loading prints many mismatch warnings, but the benchmark completed successfully and produced stable measurements
