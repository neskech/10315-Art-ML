import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from pose_module.interpret_mhr_params import PoseData
from vae_features.model.feedForwardVae import FeedForwardVAE
from vae_features.model.graphVae import GraphVAE
from vae_features.utils.angleFormat import euler_to_6d
from vae_features.utils.feedForward import Norm
from vae_features.utils.mhrToJointData import PostProcessor
from vae_features.utils.skeletonFormat import SkeletonFormat

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"
TRAIN_DIR = PROJECT_ROOT / "vae_features" / "train"
SKELETON_JSON_PATH = TRAIN_DIR / "mhr_skeleton_format.json"
JOINT_NAMES_JSON_PATH = TRAIN_DIR / "joint_names.json"


def _load_model_from_checkpoint(checkpoint_name: str, device: torch.device):
    checkpoint_path = CHECKPOINTS_PATH / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    skeleton_format = SkeletonFormat.from_json_file(SKELETON_JSON_PATH)

    if config["USE_GRAPH_VAE"]:
        model = GraphVAE(
            skeleton_format=skeleton_format,
            num_layers=config["GRAPH_NUM_LAYERS"],
            joint_embedding_dimension=config["GRAPH_JOINT_EMBED_DIM"],
            bone_embedding_dimension=config["GRAPH_BONE_EMBED_DIM"],
            decoder_joint_embedding_dimension=config["GRAPH_DECODER_JOINT_EMBED_DIM"],
            num_attention_heads=config["GRAPH_NUM_HEADS"],
            bottleneck_dimensions=config["GRAPH_BOTTLENECK_DIM"],
            bottleneck_activation=torch.nn.GELU(),
            dropout=config["GRAPH_DROPOUT"],
            device=device,
            initial_concentration=float(config.get("INITIAL_CONCENTRATION", 1.0)),
            use_6d_rotation_format=config["USE_6D_ROTATIONS"],
        )
    else:
        norm: Norm = (
            "layerNorm"
            if str(config.get("FF_NORMALIZATION", "layer")).lower() == "layer"
            else "batchNorm"
        )
        model = FeedForwardVAE(
            skeletonFormat=skeleton_format,
            encoderSizes=config["FF_ENCODER_SIZES"],
            dropout=config["FF_DROPOUT"],
            use_residuals=config["FF_USE_RESIDUALS"],
            activation=torch.nn.GELU(),
            normalization=norm,
            device=device,
            use_6d_rotation_format=config["USE_6D_ROTATIONS"],
            initial_concentration=float(config.get("INITIAL_CONCENTRATION", 1.0)),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).float().eval()
    return model, config, skeleton_format


def _build_reorder_indices(skeleton_format: SkeletonFormat) -> torch.Tensor:
    import json

    with JOINT_NAMES_JSON_PATH.open("r", encoding="utf-8") as f:
        source_joint_names = json.load(f)["joint_names"]
    target_joint_names = skeleton_format.get_joint_names()
    source_idx_by_name = {name: idx for idx, name in enumerate(source_joint_names)}
    return torch.tensor([source_idx_by_name[name] for name in target_joint_names], dtype=torch.long)


def _joint_angles_from_raw(raw: torch.Tensor) -> torch.Tensor:
    batch_size, total_dim = raw.shape
    if total_dim % 7 != 0:
        raise ValueError(f"Expected raw pose dim divisible by 7, got {total_dim}")
    joint_count = total_dim // 7
    return raw.view(batch_size, joint_count, 7)[:, :, 3:6]


def getVAEFeatureMetric(checkpoint_name: str):
    parquet_path = os.path.join(
        DATA_PATH, f"vae_features_{Path(checkpoint_name).stem}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Missing VAE features parquet: {parquet_path}. "
            f"Run data_generation/write_vae_features.py --checkpoint-name {checkpoint_name} first."
        )

    df = pd.read_parquet(parquet_path)
    feature_map = dict(zip(df["image_path"], df["vae_features"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, skeleton_format = _load_model_from_checkpoint(checkpoint_name, device)
    reorder_indices = _build_reorder_indices(skeleton_format).to(device)
    post_processor = PostProcessor(device)
    cached_feats: dict[str, list[float]] = {}

    @torch.no_grad()
    def _pose_to_feature(pose: PoseData):
        mhr_parameters = pose.mhr_parameters.to(device).float().unsqueeze(0)
        joint_data = post_processor.mhr_to_joint_data(
            mhr_parameters, zero_out_root_rotation=False
        )
        full_joint_angles = _joint_angles_from_raw(joint_data.raw)
        reordered_joint_angles = full_joint_angles[:, reorder_indices, :]

        if config["USE_6D_ROTATIONS"]:
            reordered_joint_angles = euler_to_6d(reordered_joint_angles)

        if config["USE_GRAPH_VAE"]:
            _, latent, _ = model.encode(reordered_joint_angles)
        else:
            flat_input = reordered_joint_angles.reshape(reordered_joint_angles.shape[0], -1)
            _, latent, _ = model.encode(flat_input)
        return latent.squeeze(0).detach().cpu().numpy()

    def vaeFeatureMetric(pose1: PoseData, pose2: PoseData):
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

        t1 = torch.tensor(feat1).float()
        t2 = torch.tensor(feat2).float()
        sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
        return 1.0 - sim.item()

    return vaeFeatureMetric
