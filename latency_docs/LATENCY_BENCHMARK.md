# MindDrive Latency Benchmark

## Goal

This document describes the latency-oriented benchmark path added on top of the
original MindDrive Bench2Drive evaluation.

Target requirement:

- use **latency** as the primary metric
- use camera resolution **1280 x 704**
- keep the original closed-loop benchmark path available

The default MindDrive benchmark still focuses on driving score and success rate.
The latency path adds timing instrumentation and a dedicated launcher without
replacing the original flow.

## Implemented Changes

The latency adaptation is implemented in these files:

- `team_code/minddrive_b2d_agent.py`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`
- `scripts/run_minddrive_05b_benchmark.sh`
- `scripts/run_minddrive_05b_latency.sh`

### 1. Resolution control

The agent now supports environment-driven camera resolution:

- `MINDDRIVE_CAMERA_WIDTH`
- `MINDDRIVE_CAMERA_HEIGHT`

Default values remain the original:

- width `1600`
- height `900`

Latency mode sets:

- width `1280`
- height `704`

### 2. Latency instrumentation

The agent records per-step latency in `run_step()` with the following metrics:

- `tick_ms`
- `prepare_ms`
- `pre_ms`
- `model_ms`
- `post_ms`
- `e2e_ms`

The timing boundaries are:

- `tick_ms`: sensor extraction and `tick(input_data)`
- `prepare_ms`: pipeline, collation, and tensor movement before model forward
- `pre_ms`: total time before model forward
- `model_ms`: `self.model(...)` only
- `post_ms`: PID and control generation after model forward
- `e2e_ms`: full step latency from entering `run_step()` to control ready

### 3. Latency outputs

When `MINDDRIVE_ENABLE_LATENCY=1`, the agent writes:

- `latency_records.json`
- `latency_summary.json`

These files are written under `SAVE_PATH`, which in latency mode defaults to:

- `Bench2Drive/eval_minddrive_05b_latency_1280x704/`

The summary includes:

- `mean`
- `std`
- `p50`
- `p90`
- `p95`
- `p99`
- `max`
- total/effective sample counts
- warmup configuration
- camera resolution

### 4. 1280x704 config

The dedicated latency config is:

- `adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`

It overrides:

- `ida_aug_conf["H"] = 704`
- `ida_aug_conf["W"] = 1280`

It keeps:

- `final_dim = (320, 640)`

This means the raw sensor input is now `1280x704`, while the final tensor shape
fed into the model is still unchanged. This is deliberate, because it avoids
changing model tensor assumptions while still letting the benchmark reflect the
new capture resolution.

### 5. Projection matrix handling

The agent applies a simple width/height ratio scaling to the pixel-space rows of
`lidar2img` when the camera resolution changes.

This is a pragmatic adaptation for benchmark continuity, not a full intrinsic
recalibration. It is sufficient for a latency benchmark, but it should not be
treated as a rigorous camera re-calibration procedure.

## Two Latency Modes

There are two useful measurement definitions.

### Mode A: system latency

Measure the full online stack from `run_step()` entry to final control output.

This includes:

- sensor unpacking
- JPEG roundtrip in `tick()` if enabled
- mmcv preprocessing
- batch collation
- GPU transfer
- model forward
- PID controller

This is the best choice for deployment-facing end-to-end latency.

### Mode B: model-oriented latency

If you want the benchmark to focus more on model inference and less on CPU image
packing overhead, disable the JPEG roundtrip:

- `MINDDRIVE_KEEP_JPEG_ROUNDTRIP=0`

This keeps the same benchmark path, but removes the extra JPEG encode/decode
inside `tick()`.

## Default Behavior

The dedicated launcher defaults to:

- `MINDDRIVE_ENABLE_LATENCY=1`
- `MINDDRIVE_CAMERA_WIDTH=1280`
- `MINDDRIVE_CAMERA_HEIGHT=704`
- `MINDDRIVE_LATENCY_WARMUP_STEPS=20`
- `MINDDRIVE_KEEP_JPEG_ROUNDTRIP=1`

So the default latency script currently measures **system latency** at
`1280x704`.

## How To Run

### Default command

Run the latency benchmark with:

```bash
/mnt/42_store/zxz/HUAWEI/VLA/MindDrive/scripts/run_minddrive_05b_latency.sh \
  2 \
  leaderboard/data/drivetransformer_bench2drive_dev10 \
  30000 \
  50000 \
  3
```

Arguments:

1. model GPU id
2. route basename
3. CARLA port
4. traffic manager port
5. CARLA adapter id

### Pure inference oriented run

If you want to reduce preprocessing noise:

```bash
export MINDDRIVE_KEEP_JPEG_ROUNDTRIP=0
/mnt/42_store/zxz/HUAWEI/VLA/MindDrive/scripts/run_minddrive_05b_latency.sh \
  2 \
  leaderboard/data/drivetransformer_bench2drive_dev10 \
  30000 \
  50000 \
  3
```

### Warmup control

To change the number of ignored warmup frames:

```bash
export MINDDRIVE_LATENCY_WARMUP_STEPS=50
```

## Outputs To Check

After a run, check:

- benchmark result json under `minddrive_05b_latency_results/`
- latency records under `Bench2Drive/eval_minddrive_05b_latency_1280x704/`

The main latency summary file is:

- `Bench2Drive/eval_minddrive_05b_latency_1280x704/latency_summary.json`

## Recommended Reporting

If the target is a serious latency comparison, report both:

- `system latency @ 1280x704`
- `reduced-preprocess latency @ 1280x704`

At minimum include:

- GPU model
- CUDA / driver environment
- whether CARLA rendering startup is excluded
- warmup count
- whether JPEG roundtrip is enabled
- `mean`, `p50`, `p90`, `p95`, `p99`, `max`

## Validation Status

The current latency path has been smoke-checked for:

- shell syntax of both launcher scripts
- config import
- agent import under the project environment

It has not yet been reworked into a standalone offline microbenchmark. The
current path is still based on the Bench2Drive closed-loop evaluation loop, with
latency collection added on top.
