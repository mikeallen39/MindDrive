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

## Offline Pipeline Types

The offline latency path does not use every prompt style that appears in
training or validation configs. MindDrive currently contains several prompt /
pipeline variants, and they are not interchangeable for latency measurement.

### 1. `planning` pipeline

Typical pipeline node:

- `LoadAnnoatationMixCriticalVQATest(load_type=["planning"], desc_qa=False, use_gen_token=True, single=True, ...)`

What it does:

- builds a planning-only prompt
- asks the model to generate ego future waypoints
- keeps the offline benchmark aligned with the control/planning path that
  matters for latency

Input / output characteristics:

- one stitched conversation round for planning
- one image placeholder is expected in the final prompt
- one image feature tensor is enough for the sample
- output is planning-related text / waypoint token generation

This is the pipeline used by the working `0.5B` offline latency benchmark and
is also the correct choice for `3B` offline latency.

### 2. `critical_qa` pipeline

Typical pipeline node:

- `LoadAnnoatationMixCriticalVQATest(load_type=["critical_qa"], desc_qa=True, ...)`

What it does:

- asks scene understanding questions
- emphasizes critical objects, descriptions, and reasoning-oriented answers

Input / output characteristics:

- produces descriptive VQA rounds instead of planning-only generation
- is suitable for richer evaluation / debugging, not for the minimal latency
  path

### 3. `with_history_vqa=True`

When this flag is enabled, the pipeline appends extra history-oriented
questions, such as:

- previous-frame object changes
- traffic light changes
- speed changes
- previous behavior

This is useful for richer conversational evaluation, but it increases the
number of prompt rounds.

### 4. `single=True`

This flag changes how image placeholders are inserted.

For `LoadAnnoatationMixCriticalVQATest` with `use_gen_token=True`, the current
implementation adds `DEFAULT_IMAGE_TOKEN` to every human turn when
`single=True`.

That means:

- one planning-only round still works
- multiple stitched VQA rounds will introduce multiple image placeholders

The offline latency benchmark only passes one image feature tensor for each
sample, so a multi-round prompt with multiple image placeholders will break the
multimodal alignment logic.

## Why `0.5B` Worked But `3B` Failed Initially

The failure was caused by config mismatch, not by NPU and not by model size
alone.

`0.5B` latency inherits a planning-only test pipeline:

- `load_type=["planning"]`
- `desc_qa=False`
- `single=True`

This produces a single planning prompt and only one image placeholder, which
matches the one image feature tensor passed into the LLaVA-style multimodal
stack.

The original `3B` latency config inherited the wrong test pipeline from
`minddrive_qwen25_3B_infer.py`:

- `load_type=["critical_qa"]`
- `desc_qa=True`
- `with_history_vqa=True`
- `single=True`

That combination stitched multiple human turns into one conversation and added
multiple image placeholders, while the benchmark still supplied only one image
feature tensor. During generation, `prepare_inputs_labels_for_multimodal()`
tried to consume image feature `1` while only feature `0` existed, causing:

- `IndexError: index 1 is out of bounds for dimension 0 with size 1`

The offline latency benchmark now explicitly prefers
`cfg.inference_only_pipeline` when that pipeline is available, so the `3B`
latency path also follows the same planning-only prompt construction as `0.5B`
and matches the benchmark's single-image-feature assumption.

## `3B` NPU Stability Adaptation

The `3B` offline latency path required one more runtime adaptation on Ascend
NPU beyond the prompt-pipeline fix above.

### Why a second adaptation was needed

With the corrected planning-only pipeline, the first `3B` inference step could
already finish on NPU. However, a multi-step benchmark still failed on the
second step with an Ascend runtime allocation error.

The key observation was:

- the first step completed successfully
- after that step, NPU reserved memory stayed around the high-water mark
- the next step tried to allocate another large temporary block and failed

This was not caused by prompt mismatch anymore. It was a step-to-step memory
retention problem in the offline benchmark loop.

### What the benchmark now does

The offline benchmark script now supports two runtime controls:

- `--release-cache-per-step`
- `--print-step`

`--release-cache-per-step` performs explicit cleanup after each step:

- delete step-local outputs
- delete the moved device batch
- run Python GC
- call `torch.npu.empty_cache()` on NPU

This keeps the benchmark stable for multi-step `3B` runs on Ascend without
changing model semantics.

`--print-step` prints one concise line per step so long runs can be monitored
in real time.

### Practical implication

For `3B` on Ascend NPU:

- `system_latency` should be run with per-step cache release enabled
- `pure_inference_latency` should also keep the same cleanup policy for
  consistency across long runs

Without this explicit cleanup, a single-step smoke test can pass while a real
latency benchmark still fails on step 2.

## `3B` Formal Offline Result On Ascend NPU

The formal `3B` offline benchmark was run with:

- real Bench2Drive train samples
- `5` warmup steps
- `50` measured steps
- input resolution `1280 x 704`
- config `adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py`
- checkpoint `/cache/minddrive_ckpts/minddrive_3b_rltrain.pth`
- LLM base `/cache/minddrive_ckpts/llava-qwen2.5-3b`

Result directory:

- `results/npu/latency_offline_3b_train_steps55_warmup5/`

Measured summary:

- `system_latency.e2e_ms.mean = 1456.578`
- `system_latency.model_ms.mean = 710.052`
- `system_latency.prepare_ms.mean = 746.158`
- `pure_inference_latency.e2e_ms.mean = 717.734`
- `pure_inference_latency.model_ms.mean = 711.694`
- `pure_inference_latency.prepare_ms.mean = 5.682`
- both modes passed `basic_sanity = 50 / 50`
- both modes passed `gt_reasonableness = 49 / 50`

Interpretation:

- steady-state `3B` model forward on Ascend NPU is about `711ms`
- the gap between `system_latency` and `pure_inference_latency` mainly comes
  from sample loading, collation, and device transfer
- the first warmup step is much slower than steady-state because it includes
  one-time runtime initialization / compilation cost and should not be used as
  the representative latency number

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

The current latency path has been validated for:

- config import
- real-data offline dataset construction
- `0.5B` offline latency benchmark
- `3B` offline latency benchmark on Ascend NPU
- long-run multi-step stability with explicit per-step cache release

The current recommended path for NPU latency work is the standalone offline
microbenchmark in `scripts/benchmark_minddrive_latency_offline.py`, not the old
closed-loop CARLA evaluation loop.
