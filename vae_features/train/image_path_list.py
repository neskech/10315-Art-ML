"""Shared helpers for JSON lists of pose image paths (under data/poses)."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_pose_relative_path(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if normalized.startswith("data/"):
        normalized = normalized[len("data/") :]
    if normalized.startswith("poses/"):
        return normalized
    return f"poses/{normalized}"


def load_resolved_image_paths_from_json(json_path: Path, data_dir: Path) -> list[str]:
    """Load a JSON array of path strings; return absolute paths that exist, in file order."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Image list JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array of path strings: {json_path}")

    data_dir = Path(data_dir)
    resolved: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            continue
        pose_rel = normalize_pose_relative_path(item)
        abs_path = (data_dir / pose_rel).resolve()
        if abs_path.exists():
            resolved.append(str(abs_path))
    return resolved
