"""Squared Euclidean distance in PCA space of skeleton-reordered joint angles."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from pose_module.interpret_mhr_params import PoseData
from vae_features.train.mhr_pose_dataset import _joint_angles_from_raw
from vae_features.utils.angleFormat import euler_to_6d
from vae_features.utils.mhrToJointData import PostProcessor
from vae_features.utils.skeletonFormat import SkeletonFormat

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"
TRAIN_DIR = PROJECT_ROOT / "vae_features" / "train"
SKELETON_JSON_PATH = TRAIN_DIR / "mhr_skeleton_format.json"
JOINT_NAMES_JSON_PATH = TRAIN_DIR / "joint_names.json"


def _build_reorder_indices(skeleton_format: SkeletonFormat) -> torch.Tensor:
    with JOINT_NAMES_JSON_PATH.open("r", encoding="utf-8") as f:
        source_joint_names = json.load(f)["joint_names"]
    target_joint_names = skeleton_format.get_joint_names()
    source_idx_by_name = {name: idx for idx, name in enumerate(source_joint_names)}
    return torch.tensor(
        [source_idx_by_name[name] for name in target_joint_names], dtype=torch.long
    )


def getPCAFeatureMetric(output_tag: str):
    """Load PCA embeddings from ``data/pca_features_{tag}.parquet`` and model from
    ``data/pca_model_{tag}.joblib``.

    Query poses not in the parquet are encoded with the same PostProcessor +
    reorder + optional 6D path used when the model was trained.
    """
    parquet_path = os.path.join(DATA_PATH, f"pca_features_{output_tag}.parquet")
    model_path = os.path.join(DATA_PATH, f"pca_model_{output_tag}.joblib")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Missing PCA features parquet: {parquet_path}. "
            f"Run data_generation/write_pca_features.py (e.g. --output-tag {output_tag})."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing PCA model: {model_path}. "
            f"Run data_generation/write_pca_features.py with the same tag."
        )

    bundle = joblib.load(model_path)
    pca = bundle["pca"]
    use_6d: bool = bool(bundle["use_6d_rotations"])

    df = pd.read_parquet(parquet_path)
    feature_map = dict(zip(df["image_path"], df["pca_features"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skeleton_format = SkeletonFormat.from_json_file(SKELETON_JSON_PATH)
    reorder_indices = _build_reorder_indices(skeleton_format).to(device)
    post_processor = PostProcessor(device)
    cached_feats: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _pose_to_feature(pose: PoseData) -> np.ndarray:
        mhr_parameters = pose.mhr_parameters.to(device).float().unsqueeze(0)
        joint_data = post_processor.mhr_to_joint_data(
            mhr_parameters, zero_out_root_rotation=False
        )
        full_joint_angles = _joint_angles_from_raw(joint_data.raw)
        reordered = full_joint_angles[:, reorder_indices, :]
        if use_6d:
            reordered = euler_to_6d(reordered)
        flat = reordered.reshape(1, -1).detach().cpu().numpy().astype(np.float64)
        if np.isnan(flat).any():
            flat = np.nan_to_num(flat, nan=0.0)
        z = pca.transform(flat).astype(np.float32).squeeze(0)
        return z

    def pcaFeatureMetric(pose1: PoseData, pose2: PoseData):
        feat1 = feature_map.get(pose1.relative_image_path)
        feat2 = feature_map.get(pose2.relative_image_path)

        if feat1 is None:
            if pose1.relative_image_path in cached_feats:
                feat1 = cached_feats[pose1.relative_image_path]
            else:
                feat1 = _pose_to_feature(pose1)
                cached_feats[pose1.relative_image_path] = feat1
        if feat2 is None:
            if pose2.relative_image_path in cached_feats:
                feat2 = cached_feats[pose2.relative_image_path]
            else:
                feat2 = _pose_to_feature(pose2)
                cached_feats[pose2.relative_image_path] = feat2

        d = feat1.astype(np.float64, copy=False) - feat2.astype(np.float64, copy=False)
        return float(np.square(d).sum())

    return pcaFeatureMetric
