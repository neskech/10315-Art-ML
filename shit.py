#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if you place script elsewhere
JSON_PATH = PROJECT_ROOT / "vae_features" / "train" / "reconstruction_images.json"
POSES_ROOT = PROJECT_ROOT / "data" / "poses"
DOWNLOADED_PINS_ROOT = POSES_ROOT / "downloaded_pins"
PINTEREST_ROOT = POSES_ROOT / "pinterest"


def to_source_path(raw_path: str) -> Path:
    p = Path(raw_path)

    # Absolute path in JSON
    if p.is_absolute():
        return p

    # Repo-relative path in JSON, e.g. data/poses/...
    if raw_path.startswith("data/poses/"):
        return PROJECT_ROOT / raw_path

    # Fallback: treat as relative to repo root
    return PROJECT_ROOT / raw_path


def main() -> None:
    PINTEREST_ROOT.mkdir(parents=True, exist_ok=True)

    paths = json.loads(JSON_PATH.read_text())
    copied = 0
    missing = 0

    for raw in paths:
        src = to_source_path(raw)

        if not src.exists():
            print(f"[missing] {raw}")
            missing += 1
            continue

        # Map downloaded_pins/<subdir>/file -> pinterest/<subdir>/file
        try:
            rel_to_downloaded_pins = src.resolve().relative_to(
                DOWNLOADED_PINS_ROOT.resolve()
            )
        except ValueError:
            # Fallback: preserve path under data/poses when possible.
            # If source is outside data/poses entirely, use filename only.
            try:
                rel_to_downloaded_pins = src.resolve().relative_to(POSES_ROOT.resolve())
            except ValueError:
                rel_to_downloaded_pins = Path(src.name)

        dst = PINTEREST_ROOT / rel_to_downloaded_pins
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"[copied] {src} -> {dst}")

    print(f"\nDone. Copied: {copied}, Missing: {missing}")


if __name__ == "__main__":
    main()