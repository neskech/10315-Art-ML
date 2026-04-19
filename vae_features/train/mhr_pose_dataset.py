from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from vae_features.utils.angleFormat import euler_to_6d
from vae_features.utils.mhrToJointData import JointData, PostProcessor
from vae_features.utils.skeletonFormat import SkeletonFormat


@dataclass
class PoseMetadata:
    image_path: str
    image_path_abs: str | None
    mhr_parameters: torch.Tensor
    pred_cam: torch.Tensor
    pred_cam_t: torch.Tensor
    focal_length: torch.Tensor
    pred_keypoints_2d: dict[str, tuple[float, float]]


def pose_image_abs_path(meta: PoseMetadata, data_root: Path | None) -> str | None:
    """Absolute filesystem path to the pose image for thumbnails / OpenCV, if it exists."""
    if meta.image_path_abs is not None:
        p = Path(meta.image_path_abs)
        if p.exists():
            return str(p)
    if data_root is None:
        return None
    fallback = Path(data_root) / "poses" / str(meta.image_path).lstrip("/")
    return str(fallback) if fallback.exists() else None


@dataclass
class MHRBatch:
    joint_data: JointData
    joint_angles: torch.Tensor
    metadata: list[PoseMetadata]


def _parse_value(value: Any):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _joint_angles_from_raw(raw: torch.Tensor) -> torch.Tensor:
    # Raw layout has 7 values per joint: [tx, ty, tz, rx, ry, rz, ?]
    # Joint rotations are the euler triplets at offsets [3, 4, 5].
    batch_size, total_dim = raw.shape
    if total_dim % 7 != 0:
        raise ValueError(f"Expected raw pose dim to be divisible by 7, got {total_dim}")
    joint_count = total_dim // 7
    return raw.view(batch_size, joint_count, 7)[:, :, 3:6]


class MHRPoseDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        skeleton_format: SkeletonFormat,
        joint_names_path: str | Path,
        data_root: str | Path | None = None,
        max_samples: int | None = None,
        device: torch.device | None = None,
        use_6d_rotations: bool = False,
    ):
        self.parquet_path = Path(parquet_path)
        self.data_root = Path(data_root) if data_root is not None else None
        self.skeleton_format = skeleton_format
        self.device = device if device is not None else torch.device("cpu")
        self.use_6d_rotations = use_6d_rotations
        self.post_processor = PostProcessor(self.device)

        self.df = pd.read_parquet(self.parquet_path)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        with Path(joint_names_path).open("r", encoding="utf-8") as f:
            source_joint_names = json.load(f)["joint_names"]
        target_joint_names = self.skeleton_format.get_joint_names()
        source_idx_by_name = {name: idx for idx, name in enumerate(source_joint_names)}
        self.reorder_indices = torch.tensor(
            [source_idx_by_name[name] for name in target_joint_names], dtype=torch.long
        )
        self.inverse_reorder_indices = torch.argsort(self.reorder_indices)

        self._image_key_to_index: dict[str, int] = {}
        for i in range(len(self.df)):
            key = self._canonical_image_key(str(self.df.iloc[i]["image_path"]))
            self._image_key_to_index.setdefault(key, i)

    def _canonical_image_key(self, path: str | Path) -> str:
        """Normalize any path or parquet `image_path` value to a single lookup key."""
        p = Path(path)
        if self.data_root is not None:
            poses_root = (self.data_root / "poses").resolve()
            try:
                if p.is_absolute():
                    rel = p.resolve().relative_to(poses_root)
                    return str(rel).replace("\\", "/")
            except ValueError:
                pass
        s = str(path).replace("\\", "/").strip()
        while s.startswith("./"):
            s = s[2:]
        for prefix in ("data/poses/", "poses/"):
            if s.startswith(prefix):
                s = s[len(prefix) :]
                break
        return s.lstrip("/")

    def get_batch_by_image_paths(
        self, image_paths: list[str]
    ) -> tuple[MHRBatch | None, list[str]]:
        """
        Build a batch from parquet rows matching the given paths (any format: absolute,
        or relative under poses/). Order matches ``image_paths``; missing rows are omitted.

        Returns:
            (collated batch, list of paths that had no matching parquet row)
        """
        if len(image_paths) == 0:
            return None, []

        items: list[dict[str, Any]] = []
        missing: list[str] = []
        for raw_path in image_paths:
            key = self._canonical_image_key(raw_path)
            idx = self._image_key_to_index.get(key)
            if idx is None:
                missing.append(raw_path)
                continue
            items.append(self.__getitem__(idx))

        if len(items) == 0:
            return None, missing

        return self.collate_fn(items), missing

    def __len__(self) -> int:
        return len(self.df)

    def _to_abs_image_path(self, relative_path: str) -> str | None:
        if self.data_root is None:
            return None
        clean_path = relative_path.lstrip("/")
        return str(self.data_root / "poses" / clean_path)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        mhr_parameters = torch.tensor(
            _parse_value(row["mhr_parameters"]), dtype=torch.float32
        )
        pred_cam = torch.tensor(_parse_value(row["pred_cam"]), dtype=torch.float32)
        pred_cam_t = torch.tensor(_parse_value(row["pred_cam_t"]), dtype=torch.float32)
        focal_length = torch.tensor(
            _parse_value(row["focal_length"]), dtype=torch.float32
        )
        pred_keypoints_2d = _parse_value(row["pred_keypoints_2d"])
        image_path = str(row["image_path"])

        metadata = PoseMetadata(
            image_path=image_path,
            image_path_abs=self._to_abs_image_path(image_path),
            mhr_parameters=mhr_parameters,
            pred_cam=pred_cam,
            pred_cam_t=pred_cam_t,
            focal_length=focal_length,
            pred_keypoints_2d=pred_keypoints_2d,
        )

        return {"mhr_parameters": mhr_parameters, "metadata": metadata}

    def collate_fn(self, batch: list[dict[str, Any]]) -> MHRBatch:
        mhr_parameters = torch.stack([item["mhr_parameters"] for item in batch], dim=0)
        metadata = [item["metadata"] for item in batch]

        joint_data = self.post_processor.mhr_to_joint_data(
            mhr_parameters.to(self.device), zero_out_root_rotation=False
        )
        # PostProcessor currently returns a compressed joint angle tensor; recover full per-joint
        # angles directly from raw to get (B, 127, 3).
        full_joint_angles = _joint_angles_from_raw(joint_data.raw)
        reordered_joint_angles = full_joint_angles[
            :, self.reorder_indices.to(self.device), :
        ]
        if self.use_6d_rotations:
            reordered_joint_angles = euler_to_6d(reordered_joint_angles)

        # joint_angles: (B, J, 3) Euler XYZ, or (B, J, 6) 6D when use_6d_rotations is True.
        fixed_joint_data = JointData(
            raw=joint_data.raw,
            joint_angles=reordered_joint_angles,
            root_translation=joint_data.root_translation,
            root_rotation=joint_data.root_rotation,
        )
        return MHRBatch(
            joint_data=fixed_joint_data,
            joint_angles=reordered_joint_angles,
            metadata=metadata,
        )
