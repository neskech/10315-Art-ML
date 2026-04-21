"""Fit PCA on all corpus joint-angle vectors and write per-pose PCA embeddings.

Loads every row from the poses parquet (via MHRPoseDataset), stacks flattened
reordered joint angles, fits ``sklearn.decomposition.PCA`` on the full matrix,
then transforms each pose and writes a parquet plus a joblib model for
retrieval-time transforms of query poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_features.train.mhr_pose_dataset import MHRPoseDataset
from vae_features.utils.skeletonFormat import SkeletonFormat

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
TRAIN_DIR = PROJECT_ROOT / "vae_features" / "train"
SKELETON_JSON_PATH = TRAIN_DIR / "mhr_skeleton_format.json"
JOINT_NAMES_JSON_PATH = TRAIN_DIR / "joint_names.json"
DEFAULT_POSES_PARQUET_PATH = DATA_PATH / "processed_poses.parquet"


def _collect_joint_matrix(
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Return (N, D) float matrix and parallel list of image_path strings."""
    chunks: list[np.ndarray] = []
    paths: list[str] = []
    for batch in tqdm(dataloader, desc="Loading poses for PCA"):
        ja = batch.joint_angles.to(device).float()
        if torch.isnan(ja).any():
            ja = torch.nan_to_num(ja, nan=0.0)
        flat = ja.reshape(ja.shape[0], -1).detach().cpu().numpy()
        chunks.append(flat)
        paths.extend(meta.image_path for meta in batch.metadata)
    if not chunks:
        raise RuntimeError("No samples in dataloader.")
    X = np.vstack(chunks).astype(np.float64, copy=False)
    return X, paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit PCA on all pose joint-angle vectors and write embeddings + model."
    )
    parser.add_argument(
        "--poses-parquet",
        type=Path,
        default=None,
        help="Parquet of poses (default: data/processed_poses.parquet).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="PCA output dimension (must be <= num_samples - 1 and input dim).",
    )
    parser.add_argument(
        "--use-6d-rotations",
        action="store_true",
        help="Use 6D rotation features per joint (must match retrieval metric).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Tag for output files (default: '<n_components>c', e.g. 32c).",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.n_components < 1:
        raise ValueError("n-components must be >= 1")

    poses_parquet = (
        args.poses_parquet.resolve()
        if args.poses_parquet is not None
        else DEFAULT_POSES_PARQUET_PATH.resolve()
    )
    if not poses_parquet.is_file():
        raise FileNotFoundError(f"Poses parquet not found: {poses_parquet}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skeleton_format = SkeletonFormat.from_json_file(SKELETON_JSON_PATH)

    dataset = MHRPoseDataset(
        parquet_path=poses_parquet,
        skeleton_format=skeleton_format,
        joint_names_path=JOINT_NAMES_JSON_PATH,
        data_root=DATA_PATH,
        max_samples=None,
        device=device,
        use_6d_rotations=bool(args.use_6d_rotations),
    )
    n_samples = len(dataset)
    if n_samples < 2:
        raise RuntimeError("Need at least 2 poses to fit PCA.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collate_fn,
    )

    X, image_paths = _collect_joint_matrix(dataloader, device)
    if X.shape[0] != len(image_paths):
        raise RuntimeError("Path / matrix row count mismatch.")
    feature_dim = X.shape[1]

    n_comp = min(args.n_components, n_samples - 1, feature_dim)
    if n_comp != args.n_components:
        print(
            f"Note: reducing n_components from {args.n_components} to {n_comp} "
            f"(min of n_samples-1={n_samples - 1} and feature_dim={feature_dim}).",
            flush=True,
        )

    tag = args.output_tag if args.output_tag else f"{n_comp}c"
    out_parquet = DATA_PATH / f"pca_features_{tag}.parquet"
    out_model = DATA_PATH / f"pca_model_{tag}.joblib"

    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)
    Z = pca.transform(X).astype(np.float32)

    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "pca_features": list(Z),
        }
    )
    df.to_parquet(out_parquet, compression="zstd")
    payload = {
        "pca": pca,
        "use_6d_rotations": bool(args.use_6d_rotations),
        "n_components": n_comp,
        "feature_dim": int(feature_dim),
        "poses_parquet": str(poses_parquet),
    }
    joblib.dump(payload, out_model)

    print(f"Wrote {len(df)} rows to {out_parquet}")
    print(f"Wrote PCA model to {out_model} (explained_variance_ratio sum={pca.explained_variance_ratio_.sum():.4f})")


if __name__ == "__main__":
    main()
