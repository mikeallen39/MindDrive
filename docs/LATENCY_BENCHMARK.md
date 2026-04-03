# MindDrive Latency Benchmark Notes

## Goal

If the target metric is **latency** rather than driving score / success rate, then the default Bench2Drive closed-loop benchmark is not sufficient by itself.

Bench2Drive mainly reports:

- Driving Score
- Success Rate
- Ability metrics
- Efficiency / smoothness

It does **not** directly define a latency benchmark protocol for MindDrive.

If the requirement is:

- use **latency** as the main metric
- use image resolution **1280 x 704**

then a separate latency measurement protocol should be defined on top of the current evaluation pipeline.

## Recommended Latency Protocol

There are two reasonable latency definitions.

### 1. System Latency

Measure the full time from agent input to final control output.

Recommended measurement boundary:

- start: entering `run_step(self, input_data, timestamp)`
- end: right before `return control`

This measures:

- sensor data handling
- image preprocessing
- pipeline / collate
- GPU transfer
- model forward
- PID / control postprocess

This is the best choice if the target is **deployment-facing end-to-end latency**.

### 2. Model Latency

Measure only the model forward path.

Recommended measurement boundary:

- start: immediately before `self.model(...)`
- end: immediately after `self.model(...)`

This excludes:

- sensor packing
- JPEG encode/decode
- mmcv pipeline overhead
- host/device staging
- PID controller

This is the best choice if the target is **pure inference speed**.

## Suggested Metrics

For either protocol, record at least:

- `latency_e2e_ms`
- `latency_pre_ms`
- `latency_model_ms`
- `latency_post_ms`
- `mean`
- `std`
- `p50`
- `p90`
- `p95`
- `p99`
- `max`

Also recommend:

- ignore warmup frames, e.g. first 20 frames
- report total sample count
- report hardware and software environment
- report whether CARLA render startup time is excluded

## Where To Instrument

The main entry point is:

- `MindDrive/team_code/minddrive_b2d_agent.py`

Relevant functions:

- `sensors()`
- `tick()`
- `run_step()`

Recommended timing split inside `run_step()`:

1. `tick(input_data)` and result packing
2. `self.inference_only_pipeline(results)`
3. batch collation and `.to(self.device)`
4. `self.model(...)`
5. PID controller and control generation

## Resolution 1280 x 704

To switch from the current `1600 x 900` setup to `1280 x 704`, changing only one place is not enough.

At least the following parts must be updated consistently.

### 1. Camera sensor resolution

File:

- `MindDrive/team_code/minddrive_b2d_agent.py`

Current camera sensors are defined as:

- width `1600`
- height `900`

These should be changed to:

- width `1280`
- height `704`

### 2. Config-side raw image geometry

File:

- `MindDrive/adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`

Current values:

- `ida_aug_conf["H"] = 900`
- `ida_aug_conf["W"] = 1600`

These should be changed to:

- `ida_aug_conf["H"] = 704`
- `ida_aug_conf["W"] = 1280`

Note:

- `final_dim` in the config is currently `(320, 640)`
- if this is kept unchanged, the input sensor resolution changes, but the final resized tensor shape remains the same
- this may reduce preprocessing cost but does not necessarily reduce model compute proportionally

### 3. Camera projection / intrinsic consistency

File:

- `MindDrive/team_code/minddrive_b2d_agent.py`

The agent currently contains hard-coded matrices:

- `lidar2img`
- `lidar2cam`

These values are written for the current camera setting and image geometry.

If the sensor resolution is changed to `1280 x 704` but these matrices are left unchanged, projection consistency may break.

At minimum, the image-plane dependent parts of `lidar2img` should be re-derived or rescaled consistently.

## Important Caveat About Current Preprocessing

Inside `tick()` the code currently does this for each RGB camera:

- `cv2.imencode('.jpg', img, ...)`
- `cv2.imdecode(...)`

This adds artificial CPU preprocessing overhead.

Therefore, before measuring latency, you must decide what exactly you want to measure.

### If the target is system latency

Keep this behavior.

Reason:

- it reflects the current actual agent path

### If the target is model latency

Remove or bypass this JPEG encode/decode step.

Reason:

- otherwise the reported latency is dominated by preprocessing noise instead of model inference

## Recommended Reporting Modes

### Mode A: System Latency Benchmark

Keep the current agent logic intact and report:

- end-to-end per-frame latency
- latency distribution after warmup
- resolution `1280 x 704`

Use this if the question is:

- "How long does the whole online driving stack take per step?"

### Mode B: Model Latency Benchmark

Bypass JPEG encode/decode and measure only model forward.

Use this if the question is:

- "How fast is MindDrive inference itself?"

## Practical Recommendation

For serious comparison, report both:

- `system latency @ 1280x704`
- `model latency @ 1280x704`

This avoids mixing:

- deployment overhead
- preprocessing overhead
- actual neural inference cost

## Implementation Recommendation

The cleanest next step is:

1. modify sensor resolution to `1280 x 704`
2. update config `H/W`
3. verify / adjust projection matrices
4. add latency logging in `run_step()`
5. output per-frame latency JSON
6. output summary statistics at the end

## Scope Clarification

Changing the input resolution to `1280 x 704` does **not** automatically make the benchmark a latency benchmark.

A latency benchmark needs:

- explicit timing boundaries
- explicit warmup policy
- explicit reporting statistics
- explicit statement of whether preprocessing is included

