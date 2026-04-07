#!/usr/bin/env python

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = str(REPO_ROOT / "adzoo" / "minddrive" / "configs" / "minddrive_qwen2_05B_latency.py")
DEFAULT_CKPT = str(REPO_ROOT / "ckpts" / "minddrive_rltrain.pth")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results_latency_offline_1280x704")
CUSTOM_FP16_FLAGS = {
    "map_head": False,
    "pts_bbox_head": False,
}


def ensure_repo_imports():
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def infer_device():
    requested = os.environ.get("MINDDRIVE_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_sync(device):
    if device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def device_info(device):
    info = {"device": device}
    if device == "npu" and hasattr(torch, "npu"):
        info["device_count"] = torch.npu.device_count()
        info["visible_devices"] = os.environ.get("NPU-VISIBLE-DEVICES", "")
    elif device == "cuda" and torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline latency benchmark for MindDrive using real Bench2Drive samples"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-pool-size", type=int, default=8)
    parser.add_argument("--max-ego-fde", type=float, default=20.0)
    parser.add_argument("--max-path-fde", type=float, default=25.0)
    parser.add_argument("--max-traj-abs-m", type=float, default=150.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    return parser.parse_args()


def import_cfg_plugins(cfg):
    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None)
    if not plugin_dir:
        return
    module_dir = os.path.dirname(plugin_dir).split("/")
    module_path = module_dir[0]
    for part in module_dir[1:]:
        module_path = f"{module_path}.{part}"
    importlib.import_module(module_path)


def repo_path(path_value):
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_dataset_cfg(cfg, split):
    if split == "train":
        dataset_cfg = cfg.data.val.copy()
        dataset_cfg["ann_file"] = cfg.data.train.ann_file
    else:
        dataset_cfg = cfg.data[split].copy()
    dataset_cfg["test_mode"] = True
    dataset_cfg["filter_empty_gt"] = False
    return dataset_cfg


def validate_dataset_assets(dataset_cfg):
    checks = {
        "data_root": repo_path(dataset_cfg["data_root"]),
        "ann_file": repo_path(dataset_cfg["ann_file"]),
        "map_file": repo_path(dataset_cfg["map_file"]),
    }
    missing = {name: str(path) for name, path in checks.items() if not path.exists()}
    if missing:
        missing_lines = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            "Real-data offline latency requires prepared Bench2Drive assets, but some required paths are missing:\n"
            f"{missing_lines}\n"
            "You need the raw `data/bench2drive` assets and prepared `data/infos/*.pkl` files. "
            "The repo's `docs/DATA_PREP.md` describes the expected layout and preparation step."
        )


def custom_wrap_fp16_model(model):
    for module in model.modules():
        if hasattr(module, "fp16_enabled"):
            module.fp16_enabled = True
    for module_name, enabled in CUSTOM_FP16_FLAGS.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = enabled


def build_runtime(config_path, ckpt_path, split):
    from mmcv import Config
    from mmcv.datasets import build_dataset
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint

    cfg = Config.fromfile(config_path)
    import_cfg_plugins(cfg)

    dataset_cfg = resolve_dataset_cfg(cfg, split)
    validate_dataset_assets(dataset_cfg)

    dataset = build_dataset(dataset_cfg)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")

    device = infer_device()
    model.to(device)
    model.eval()
    if device != "cpu":
        custom_wrap_fp16_model(model)

    return cfg, dataset, model, device


def make_sample_indices(dataset_len, steps, start_index, sample_pool_size):
    if dataset_len <= 0:
        raise ValueError("Dataset is empty; cannot run latency benchmark.")
    pool_size = min(max(sample_pool_size, 1), dataset_len)
    base = [(start_index + offset) % dataset_len for offset in range(pool_size)]
    return [base[step % pool_size] for step in range(steps)], base


def collate_sample(sample):
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form

    return mm_collate_to_batch_form([sample], samples_per_gpu=1)


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        moved = {}
        for key, value in data.items():
            moved[key] = value if key == "img_metas" else move_to_device(value, device)
        return moved
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return to_numpy(value[0])
        return np.asarray([to_numpy(item) for item in value], dtype=object)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def squeeze_leading_singletons(array_like):
    value = to_numpy(array_like)
    if value is None:
        return None
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    return value


def extract_gt_traj(batch_data, key):
    if key not in batch_data:
        return None
    value = squeeze_leading_singletons(batch_data[key])
    if value is None:
        return None
    while value.ndim > 2:
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != 2:
        return None
    return np.cumsum(value.astype(np.float64), axis=0)


def scalar_from_batch(batch_data, key):
    if key not in batch_data:
        return None
    value = to_numpy(batch_data[key])
    if value is None:
        return None
    if value.size == 0:
        return None
    return bool(value.reshape(-1)[0])


def compute_reasonableness(sample_index, batch_data, output, max_ego_fde, max_path_fde, max_traj_abs_m):
    pts_bbox = output.get("pts_bbox", {})
    metric_results = output.get("metric_results") or {}

    ego_pred = squeeze_leading_singletons(pts_bbox.get("ego_fut_preds"))
    pw_pred = squeeze_leading_singletons(pts_bbox.get("pw_ego_fut_pred"))
    ego_gt = extract_gt_traj(batch_data, "ego_fut_trajs")
    path_gt = extract_gt_traj(batch_data, "path_points_future")
    fut_valid_flag = scalar_from_batch(batch_data, "fut_valid_flag")

    result = {
        "sample_index": int(sample_index),
        "fut_valid_flag": bool(fut_valid_flag) if fut_valid_flag is not None else None,
        "ego_pred_present": ego_pred is not None,
        "pw_pred_present": pw_pred is not None,
        "metric_results": {
            key: float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
            for key, value in metric_results.items()
        },
    }

    def inspect_prediction(prefix, pred, gt, max_fde):
        info = {
            f"{prefix}_shape_ok": False,
            f"{prefix}_finite": False,
            f"{prefix}_within_bounds": False,
            f"{prefix}_gt_available": gt is not None,
            f"{prefix}_gt_reasonable": None,
        }
        if pred is None:
            return info
        pred = np.asarray(pred, dtype=np.float64)
        info[f"{prefix}_shape"] = list(pred.shape)
        info[f"{prefix}_shape_ok"] = pred.ndim == 2 and pred.shape[-1] == 2
        info[f"{prefix}_finite"] = bool(np.isfinite(pred).all())
        info[f"{prefix}_within_bounds"] = bool(np.max(np.abs(pred)) <= max_traj_abs_m) if pred.size else False
        if gt is not None and info[f"{prefix}_shape_ok"] and pred.shape == gt.shape:
            delta = pred - gt
            l2 = np.linalg.norm(delta, axis=-1)
            info[f"{prefix}_ade"] = float(np.mean(l2))
            info[f"{prefix}_fde"] = float(l2[-1])
            info[f"{prefix}_gt_reasonable"] = bool(l2[-1] <= max_fde)
        return info

    result.update(inspect_prediction("ego", ego_pred, ego_gt, max_ego_fde))
    result.update(inspect_prediction("path", pw_pred, path_gt, max_path_fde))

    basic_checks = [
        result["ego_pred_present"],
        result["ego_shape_ok"],
        result["ego_finite"],
        result["ego_within_bounds"],
    ]
    if result["pw_pred_present"]:
        basic_checks.extend(
            [result["path_shape_ok"], result["path_finite"], result["path_within_bounds"]]
        )
    result["basic_sanity_ok"] = bool(all(basic_checks))

    gt_checks = []
    if result["ego_gt_reasonable"] is not None:
        gt_checks.append(result["ego_gt_reasonable"])
    if result["path_gt_reasonable"] is not None:
        gt_checks.append(result["path_gt_reasonable"])
    result["gt_reasonableness_ok"] = bool(all(gt_checks)) if gt_checks else None
    return result


def summarize_metric(values):
    if not values:
        return {}
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values_np.mean()),
        "std": float(values_np.std()),
        "p50": float(np.percentile(values_np, 50)),
        "p90": float(np.percentile(values_np, 90)),
        "p95": float(np.percentile(values_np, 95)),
        "p99": float(np.percentile(values_np, 99)),
        "max": float(values_np.max()),
    }


def summarize_run(
    mode_name,
    records,
    sample_pool,
    split,
    device,
    wall_total_s,
    width,
    height,
    thresholds,
):
    effective_records = [
        item for item in records.values() if not item["warmup"]
    ]
    summary = {
        "mode": mode_name,
        "split": split,
        "camera_width": width,
        "camera_height": height,
        "wall_total_s": round(wall_total_s, 3),
        "count_total": len(records),
        "count_effective": len(effective_records),
        "sample_pool_indices": sample_pool,
        "thresholds": thresholds,
    }
    summary.update(device_info(device))

    latency_keys = [
        "sample_ms",
        "collate_ms",
        "transfer_ms",
        "prepare_ms",
        "model_ms",
        "post_ms",
        "e2e_ms",
    ]
    for key in latency_keys:
        summary[key] = summarize_metric([item[key] for item in effective_records])

    basic_ok = [item["sanity"]["basic_sanity_ok"] for item in effective_records]
    gt_ok = [
        item["sanity"]["gt_reasonableness_ok"]
        for item in effective_records
        if item["sanity"]["gt_reasonableness_ok"] is not None
    ]
    ego_fde_values = [
        item["sanity"]["ego_fde"] for item in effective_records if "ego_fde" in item["sanity"]
    ]
    path_fde_values = [
        item["sanity"]["path_fde"] for item in effective_records if "path_fde" in item["sanity"]
    ]
    summary["sanity"] = {
        "basic_sanity_pass": int(sum(1 for value in basic_ok if value)),
        "basic_sanity_total": len(basic_ok),
        "gt_reasonableness_pass": int(sum(1 for value in gt_ok if value)),
        "gt_reasonableness_total": len(gt_ok),
        "ego_fde_m": summarize_metric(ego_fde_values),
        "path_fde_m": summarize_metric(path_fde_values),
    }
    return summary


def load_sample(dataset, sample_index):
    try:
        return dataset[sample_index]
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset sample at index {sample_index}: {exc}") from exc


def preload_collated_batches(dataset, sample_pool):
    cached = {}
    for sample_index in sample_pool:
        sample = load_sample(dataset, sample_index)
        cached[sample_index] = collate_sample(sample)
    return cached


def run_mode(mode_name, args, dataset, model, device, iteration_indices, sample_pool):
    cached_batches = preload_collated_batches(dataset, sample_pool) if mode_name == "pure_inference_latency" else None
    records = OrderedDict()
    thresholds = {
        "max_ego_fde_m": args.max_ego_fde,
        "max_path_fde_m": args.max_path_fde,
        "max_traj_abs_m": args.max_traj_abs_m,
    }

    wall_start = time.perf_counter()
    with torch.no_grad():
        for step, sample_index in enumerate(iteration_indices):
            step_start = time.perf_counter()

            if cached_batches is None:
                sample_start = time.perf_counter()
                sample = load_sample(dataset, sample_index)
                sample_end = time.perf_counter()

                collate_start = sample_end
                batch_cpu = collate_sample(sample)
                collate_end = time.perf_counter()
            else:
                sample_start = step_start
                sample_end = step_start
                collate_start = step_start
                batch_cpu = cached_batches[sample_index]
                collate_end = step_start

            transfer_start = time.perf_counter()
            batch_device = move_to_device(batch_cpu, device)
            device_sync(device)
            transfer_end = time.perf_counter()

            model_start = transfer_end
            outputs = model(batch_device, return_loss=False)
            device_sync(device)
            model_end = time.perf_counter()

            sanity = compute_reasonableness(
                sample_index=sample_index,
                batch_data=batch_cpu,
                output=outputs[0],
                max_ego_fde=args.max_ego_fde,
                max_path_fde=args.max_path_fde,
                max_traj_abs_m=args.max_traj_abs_m,
            )
            post_end = time.perf_counter()

            records[str(step)] = {
                "step": int(step),
                "sample_index": int(sample_index),
                "warmup": bool(step < args.warmup_steps),
                "sample_ms": round((sample_end - sample_start) * 1000.0, 3),
                "collate_ms": round((collate_end - collate_start) * 1000.0, 3),
                "transfer_ms": round((transfer_end - transfer_start) * 1000.0, 3),
                "prepare_ms": round((transfer_end - step_start) * 1000.0, 3),
                "model_ms": round((model_end - model_start) * 1000.0, 3),
                "post_ms": round((post_end - model_end) * 1000.0, 3),
                "e2e_ms": round((post_end - step_start) * 1000.0, 3),
                "sanity": sanity,
            }

    wall_end = time.perf_counter()
    summary = summarize_run(
        mode_name=mode_name,
        records=records,
        sample_pool=sample_pool,
        split=args.split,
        device=device,
        wall_total_s=wall_end - wall_start,
        width=args.width,
        height=args.height,
        thresholds=thresholds,
    )
    return summary, records


def main():
    args = parse_args()
    ensure_repo_imports()
    torch.set_grad_enabled(False)
    os.chdir(REPO_ROOT)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, dataset, model, device = build_runtime(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        split=args.split,
    )
    iteration_indices, sample_pool = make_sample_indices(
        dataset_len=len(dataset),
        steps=args.steps,
        start_index=args.start_index,
        sample_pool_size=args.sample_pool_size,
    )

    all_summaries = {}
    for mode_name in ["system_latency", "pure_inference_latency"]:
        summary, records = run_mode(
            mode_name=mode_name,
            args=args,
            dataset=dataset,
            model=model,
            device=device,
            iteration_indices=iteration_indices,
            sample_pool=sample_pool,
        )
        summary["dataset_size"] = len(dataset)
        summary["config"] = str(pathlib.Path(args.config).resolve())
        summary["checkpoint"] = str(pathlib.Path(args.checkpoint).resolve())
        summary["effective_config_save_path"] = getattr(cfg.model, "save_path", None)
        all_summaries[mode_name] = summary

        with open(output_dir / f"{mode_name}_records.json", "w") as outfile:
            json.dump(records, outfile, indent=2)
        with open(output_dir / f"{mode_name}_summary.json", "w") as outfile:
            json.dump(summary, outfile, indent=2)

    with open(output_dir / "combined_summary.json", "w") as outfile:
        json.dump(all_summaries, outfile, indent=2)

    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
