#!/usr/bin/env python

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
import types
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RL_PROJECTS_DIR = REPO_ROOT / "rl_projects"
SCENARIO_RUNNER_DIR = RL_PROJECTS_DIR / "scenario_runner"
CARLA_MOCKS_DIR = REPO_ROOT / "rl_projects" / "scenario_runner" / "srunner" / "tests" / "carla_mocks"
DEFAULT_CONFIG = str(REPO_ROOT / "adzoo" / "minddrive" / "configs" / "minddrive_qwen2_05B_latency.py")
DEFAULT_CKPT = str(REPO_ROOT / "ckpts" / "minddrive_rltrain.pth")
CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class FakeRoadOption(IntEnum):
    LANEFOLLOW = 4


def ensure_repo_imports():
    for path in [str(REPO_ROOT), str(RL_PROJECTS_DIR), str(SCENARIO_RUNNER_DIR), str(CARLA_MOCKS_DIR)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_carla_stub():
    module = None
    try:
        import carla as module  # noqa: F401
    except ImportError:
        mock_path = CARLA_MOCKS_DIR / "carla.py"
        if mock_path.exists():
            spec = importlib.util.spec_from_file_location("carla", mock_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        if module is None:
            module = types.ModuleType("carla")
    if not hasattr(module, "VehicleControl"):
        class VehicleControl:
            def __init__(self):
                self.steer = 0.0
                self.throttle = 0.0
                self.brake = 0.0

        module.VehicleControl = VehicleControl
    sys.modules["carla"] = module


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
    parser = argparse.ArgumentParser(description="Offline latency benchmark for MindDrive 0.5B")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results_latency_offline_1280x704"),
    )
    return parser.parse_args()


def make_fake_hero_actor():
    def vector(x=0.0, y=0.0, z=0.0):
        return SimpleNamespace(x=x, y=y, z=z)

    rotation = SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0)
    transform = SimpleNamespace(
        location=vector(),
        rotation=rotation,
        get_forward_vector=lambda: vector(1.0, 0.0, 0.0),
        get_right_vector=lambda: vector(0.0, 1.0, 0.0),
    )
    return SimpleNamespace(
        get_acceleration=lambda: vector(),
        get_angular_velocity=lambda: vector(),
        get_transform=lambda: transform,
    )


def patch_agent_bootstrap():
    from leaderboard.autoagents import autonomous_agent

    autonomous_agent.AutonomousAgent.get_hero = lambda self: setattr(self, "hero_actor", make_fake_hero_actor())


def make_global_plan(lat_ref=42.0, lon_ref=2.0, num_points=8, delta_lon=1e-5):
    return [
        ({"lat": lat_ref, "lon": lon_ref + idx * delta_lon}, FakeRoadOption.LANEFOLLOW)
        for idx in range(num_points)
    ]


def make_static_input(width, height, seed):
    rng = np.random.default_rng(seed)
    input_data = {}
    for cam in CAM_NAMES:
        img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        input_data[cam] = (0.0, img)
    bev = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    imu = np.zeros(7, dtype=np.float32)
    gps = np.array([42.0, 2.0], dtype=np.float64)
    input_data["bev"] = (0.0, bev)
    input_data["GPS"] = (0.0, gps)
    input_data["SPEED"] = (0.0, {"speed": 2.0})
    input_data["IMU"] = (0.0, imu)
    return input_data


def build_agent(config_path, ckpt_path, width, height, warmup_steps, keep_jpeg_roundtrip):
    os.environ["IS_BENCH2DRIVE"] = "True"
    os.environ.pop("SAVE_PATH", None)
    os.environ["MINDDRIVE_ENABLE_LATENCY"] = "1"
    os.environ["MINDDRIVE_CAMERA_WIDTH"] = str(width)
    os.environ["MINDDRIVE_CAMERA_HEIGHT"] = str(height)
    os.environ["MINDDRIVE_LATENCY_WARMUP_STEPS"] = str(warmup_steps)
    os.environ["MINDDRIVE_KEEP_JPEG_ROUNDTRIP"] = "1" if keep_jpeg_roundtrip else "0"

    patch_agent_bootstrap()

    from team_code.minddrive_b2d_agent import MinddriveAgent
    from team_code.planner import RoutePlanner

    agent = MinddriveAgent("localhost", 2000, 0)
    agent.setup(f"{config_path}+{ckpt_path}")
    agent.hero_actor = make_fake_hero_actor()
    agent.lat_ref = 42.0
    agent.lon_ref = 2.0
    agent._global_plan = make_global_plan(lat_ref=agent.lat_ref, lon_ref=agent.lon_ref)
    agent._route_planner = RoutePlanner(4.0, 50.0, lat_ref=agent.lat_ref, lon_ref=agent.lon_ref)
    agent._route_planner.set_route(agent._global_plan, True)
    agent.initialized = True
    agent.metric_info = {}
    return agent


def run_mode(mode_name, config_path, ckpt_path, width, height, steps, warmup_steps, seed, keep_jpeg_roundtrip):
    agent = build_agent(
        config_path=config_path,
        ckpt_path=ckpt_path,
        width=width,
        height=height,
        warmup_steps=warmup_steps,
        keep_jpeg_roundtrip=keep_jpeg_roundtrip,
    )
    input_data = make_static_input(width=width, height=height, seed=seed)
    device = infer_device()

    wall_start = time.perf_counter()
    for step in range(steps):
        agent.run_step(input_data, step / 20.0)
        device_sync(device)
    wall_end = time.perf_counter()

    summary = agent._latency_summary()
    summary["mode"] = mode_name
    summary["steps_requested"] = steps
    summary["wall_total_s"] = round(wall_end - wall_start, 3)
    summary.update(device_info(device))
    agent.destroy()
    return summary, agent.latency_records


def main():
    args = parse_args()
    ensure_repo_imports()
    ensure_carla_stub()
    torch.set_grad_enabled(False)
    os.chdir(REPO_ROOT)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        ("system_latency", True, args.seed),
        ("pure_inference_latency", False, args.seed + 1),
    ]

    all_summaries = {}
    for mode_name, keep_jpeg_roundtrip, seed in modes:
        summary, records = run_mode(
            mode_name=mode_name,
            config_path=args.config,
            ckpt_path=args.checkpoint,
            width=args.width,
            height=args.height,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=seed,
            keep_jpeg_roundtrip=keep_jpeg_roundtrip,
        )
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
