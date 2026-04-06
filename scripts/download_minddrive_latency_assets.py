#!/usr/bin/env python

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Download MindDrive latency benchmark assets")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    ckpt_dir = repo_root / "ckpts"
    data_dir = repo_root / "data"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_model:
        hf_hub_download(
            repo_id="poleyzdk/Minddrive",
            filename="minddrive_rltrain.pth",
            local_dir=str(ckpt_dir),
            local_dir_use_symlinks=False,
            repo_type="model",
        )
        snapshot_download(
            repo_id="poleyzdk/Minddrive",
            allow_patterns=[
                "llava-qwen2-0.5b/config.json",
                "llava-qwen2-0.5b/generation_config.json",
                "llava-qwen2-0.5b/merges.txt",
                "llava-qwen2-0.5b/model.safetensors",
                "llava-qwen2-0.5b/special_tokens_map.json",
                "llava-qwen2-0.5b/tokenizer.json",
                "llava-qwen2-0.5b/tokenizer_config.json",
                "llava-qwen2-0.5b/vocab.json",
                "llava-qwen2-0.5b/added_tokens.json",
            ],
            local_dir=str(ckpt_dir),
            local_dir_use_symlinks=False,
            repo_type="model",
        )

    if not args.skip_dataset:
        hf_hub_download(
            repo_id="poleyzdk/Chat-B2D",
            filename="ChatB2D-plus.zip",
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,
            repo_type="dataset",
        )


if __name__ == "__main__":
    main()
