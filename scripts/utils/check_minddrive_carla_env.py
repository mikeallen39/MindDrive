#!/usr/bin/env python

import argparse
import glob
import importlib
import os
import pathlib
import sys


def add_message(messages, ok, label, detail):
    prefix = "OK" if ok else "MISSING"
    messages.append(f"[{prefix}] {label}: {detail}")


def try_import(module_name):
    try:
        module = importlib.import_module(module_name)
        return True, getattr(module, "__file__", "builtin")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Preflight check for MindDrive CARLA closed-loop runtime")
    parser.add_argument("--require-runtime", action="store_true", help="Fail if CARLA runtime prerequisites are missing")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    carla_root = pathlib.Path(os.environ.get("CARLA_ROOT", str(repo_root / "carla")))
    messages = []
    missing = []

    carla_server = carla_root / "CarlaUE4.sh"
    python_api_dir = carla_root / "PythonAPI"
    python_api_carla_dir = python_api_dir / "carla"
    dist_glob = str(python_api_carla_dir / "dist" / "carla-*.egg")
    dist_matches = glob.glob(dist_glob)

    server_ok = carla_server.exists()
    python_api_ok = python_api_dir.is_dir()
    carla_pkg_ok = python_api_carla_dir.is_dir()
    dist_ok = len(dist_matches) > 0

    add_message(messages, carla_root.exists(), "CARLA_ROOT", str(carla_root))
    add_message(messages, server_ok, "CARLA server", str(carla_server))
    add_message(messages, python_api_ok, "PythonAPI dir", str(python_api_dir))
    add_message(messages, carla_pkg_ok, "PythonAPI/carla dir", str(python_api_carla_dir))
    add_message(messages, dist_ok, "PythonAPI egg", dist_matches[0] if dist_matches else dist_glob)

    if not carla_root.exists():
        missing.append("CARLA_ROOT")
    if args.require_runtime:
        if not server_ok:
            missing.append("CarlaUE4.sh")
        if not python_api_ok:
            missing.append("PythonAPI")
        if not carla_pkg_ok:
            missing.append("PythonAPI/carla")
        if not dist_ok:
            missing.append("PythonAPI egg")

    for module_name in [
        "carla",
        "agents.navigation.global_route_planner",
        "leaderboard.autoagents.autonomous_agent",
        "team_code.minddrive_b2d_agent",
    ]:
        ok, detail = try_import(module_name)
        add_message(messages, ok, f"import {module_name}", detail)
        if args.require_runtime and not ok:
            missing.append(f"import:{module_name}")

    print("\n".join(messages))

    if missing:
        missing_items = ", ".join(missing)
        print(
            "\nClosed-loop CARLA runtime is not ready.\n"
            "Install the official CARLA package in a path with enough free space, for example `/home/ma-user/work/carla`,\n"
            "then export `CARLA_ROOT` or rely on `scripts/env_minddrive_b2d.sh` auto-detection.\n"
            f"Missing items: {missing_items}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
