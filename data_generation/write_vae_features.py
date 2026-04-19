import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_features.model.feedForwardVae import FeedForwardVAE
from vae_features.model.graphVae import GraphVAE
from vae_features.train.mhr_pose_dataset import MHRPoseDataset
from vae_features.utils.feedForward import Norm
from vae_features.utils.skeletonFormat import SkeletonFormat

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"
TRAIN_DIR = PROJECT_ROOT / "vae_features" / "train"
SKELETON_JSON_PATH = TRAIN_DIR / "mhr_skeleton_format.json"
JOINT_NAMES_JSON_PATH = TRAIN_DIR / "joint_names.json"
POSES_PARQUET_PATH = DATA_PATH / "processed_poses.parquet"


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain a 'config' key."
        )

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


@torch.no_grad()
def _extract_features(
    model,
    config: dict,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict] = []
    use_graph = bool(config["USE_GRAPH_VAE"])

    for batch in tqdm(dataloader, desc="Extracting VAE latent features"):
        joint_angles = batch.joint_angles.to(device).float()
        if use_graph:
            distribution, latent, _ = model.encode(joint_angles)
        else:
            flat_input = joint_angles.reshape(joint_angles.shape[0], -1)
            distribution, latent, _ = model.encode(flat_input)
        _ = distribution  # keep return unpacking explicit

        latent_np = latent.detach().cpu().numpy()
        for meta, emb in zip(batch.metadata, latent_np, strict=False):
            records.append({"image_path": meta.image_path, "vae_features": emb})

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Write VAE latent features for processed poses."
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        required=True,
        help="Checkpoint filename under checkpoints/ (e.g. mhr_vae_best.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_PATH / args.checkpoint_name

    model, config, skeleton_format = _load_model_from_checkpoint(checkpoint_path, device)

    dataset = MHRPoseDataset(
        parquet_path=POSES_PARQUET_PATH,
        skeleton_format=skeleton_format,
        joint_names_path=JOINT_NAMES_JSON_PATH,
        data_root=DATA_PATH,
        max_samples=None,
        device=device,
        use_6d_rotations=bool(config["USE_6D_ROTATIONS"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collate_fn,
    )

    features_df = _extract_features(model, config, dataloader, device)
    output_path = DATA_PATH / f"vae_features_{Path(args.checkpoint_name).stem}.parquet"
    features_df.to_parquet(output_path, compression="zstd")
    print(f"Wrote {len(features_df)} VAE features to {output_path}")


if __name__ == "__main__":
    main()
