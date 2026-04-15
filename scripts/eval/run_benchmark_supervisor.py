#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def emit(message: str, log_fp=None) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    if log_fp is not None:
        print(line, file=log_fp, flush=True)


def load_checkpoint_state(checkpoint_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if not checkpoint_path.exists():
        return None, None, None

    try:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None

    progress = data.get("_checkpoint", {}).get("progress")
    entry_status = data.get("entry_status")
    if not isinstance(progress, list) or len(progress) != 2:
        return None, None, entry_status
    return progress[0], progress[1], entry_status


def is_finished(progress: Optional[int], total: Optional[int], entry_status: Optional[str]) -> bool:
    if progress is None or total is None or total <= 0:
        return False
    if progress < total:
        return False
    return entry_status in {"Finished", "Finished with missing data", "Finished with warnings", "Crashed"}


def kill_by_pattern(pattern: str, log_fp=None) -> None:
    protected_pids = {os.getpid(), os.getppid()}
    try:
        output = subprocess.check_output(["pgrep", "-f", "--", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        emit(f"cleanup pattern: {pattern} (no match)", log_fp=log_fp)
        return

    killed = []
    for token in output.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid in protected_pids:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    if killed:
        emit(f"cleanup pattern: {pattern} killed={killed}", log_fp=log_fp)
    else:
        emit(f"cleanup pattern: {pattern} (only protected/self matches)", log_fp=log_fp)


def cleanup_processes(patterns: Iterable[str], log_fp=None) -> None:
    for pattern in patterns:
        kill_by_pattern(pattern, log_fp=log_fp)


def stream_process_output(proc: subprocess.Popen, log_fp=None) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log_fp is not None:
            print(line, end="", file=log_fp, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart MindDrive benchmark automatically from checkpoint after CARLA or evaluator crashes."
    )
    parser.add_argument("--runner", required=True, help="Shell runner script, for example scripts/eval/run_minddrive_05b_benchmark.sh")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint json used by leaderboard resume")
    parser.add_argument("--log-file", default="", help="Optional supervisor log file")
    parser.add_argument("--max-restarts", type=int, default=50, help="Maximum retry attempts after incomplete runs")
    parser.add_argument("--restart-delay", type=int, default=15, help="Seconds to wait before restarting")
    parser.add_argument(
        "--max-stagnant-attempts",
        type=int,
        default=5,
        help="Abort if checkpoint progress does not increase for this many attempts",
    )
    parser.add_argument(
        "--kill-pattern",
        action="append",
        default=[],
        help="Process pattern passed to pgrep/pkill logic before retries. Can be specified multiple times.",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, help="Arguments passed to the runner after '--'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runner = Path(args.runner).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if not runner.exists():
        print(f"runner script not found: {runner}", file=sys.stderr)
        return 2

    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]

    log_fp = None
    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")

    child = None

    def handle_signal(signum, _frame):
        nonlocal child
        emit(f"received signal {signum}, stopping supervisor", log_fp=log_fp)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                time.sleep(2)
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        cleanup_processes(args.kill_pattern, log_fp=log_fp)
        if log_fp is not None:
            log_fp.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    previous_progress = None
    stagnant_attempts = 0
    attempt = 0

    while True:
        progress, total, entry_status = load_checkpoint_state(checkpoint)
        if is_finished(progress, total, entry_status):
            emit(f"checkpoint finished: progress={progress}/{total}, entry_status={entry_status}", log_fp=log_fp)
            if log_fp is not None:
                log_fp.close()
            return 0

        if attempt > args.max_restarts:
            emit(
                f"abort: exceeded max restarts ({args.max_restarts}), last checkpoint={progress}/{total}, entry_status={entry_status}",
                log_fp=log_fp,
            )
            if log_fp is not None:
                log_fp.close()
            return 1

        attempt += 1
        emit(
            f"attempt {attempt}: start runner with checkpoint={checkpoint} progress={progress}/{total} entry_status={entry_status}",
            log_fp=log_fp,
        )

        cleanup_processes(args.kill_pattern, log_fp=log_fp)
        time.sleep(2)

        child = subprocess.Popen(
            ["bash", str(runner), *runner_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        try:
            stream_process_output(child, log_fp=log_fp)
        finally:
            child.wait()

        rc = child.returncode
        progress_after, total_after, entry_status_after = load_checkpoint_state(checkpoint)
        emit(
            f"attempt {attempt}: runner exit_code={rc}, checkpoint={progress_after}/{total_after}, entry_status={entry_status_after}",
            log_fp=log_fp,
        )

        if is_finished(progress_after, total_after, entry_status_after):
            emit(f"benchmark completed after attempt {attempt}", log_fp=log_fp)
            if log_fp is not None:
                log_fp.close()
            return 0

        if progress_after is not None and previous_progress is not None and progress_after <= previous_progress:
            stagnant_attempts += 1
        else:
            stagnant_attempts = 0
        previous_progress = progress_after

        if stagnant_attempts >= args.max_stagnant_attempts:
            emit(
                f"abort: checkpoint progress stalled for {stagnant_attempts} consecutive attempts at {progress_after}/{total_after}",
                log_fp=log_fp,
            )
            if log_fp is not None:
                log_fp.close()
            return 1

        emit(f"sleep {args.restart_delay}s before retry", log_fp=log_fp)
        cleanup_processes(args.kill_pattern, log_fp=log_fp)
        time.sleep(args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
