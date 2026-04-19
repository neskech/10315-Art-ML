from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_features.train.image_path_list import load_resolved_image_paths_from_json
from vae_features.train.mhr_pose_dataset import MHRPoseDataset
from vae_features.utils.skeletonFormat import SkeletonFormat

_POSE_INTERPRETER = None
_RENDERER_CLASS = None


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_image_path(meta: dict[str, Any], data_dir: str | Path) -> str | None:
    img_path = meta.get("image_path_abs") or meta.get("image_path")
    if img_path is not None:
        path = Path(img_path)
        if path.exists():
            return str(path)

    rel = meta.get("image_path")
    if rel is None:
        return None
    fallback = Path(data_dir) / "poses" / str(rel).lstrip("/")
    if fallback.exists():
        return str(fallback)
    return None


def _render_mesh_overlay(image_bgr: np.ndarray, pose_data, renderer_class, faces: np.ndarray) -> np.ndarray:
    renderer = renderer_class(focal_length=_to_numpy(pose_data.focal_length), faces=faces)
    rendered = renderer(
        _to_numpy(pose_data.pred_vertices),
        _to_numpy(pose_data.pred_cam_t),
        image_bgr.copy(),
        mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
        scene_bg_color=(1, 1, 1),
    )
    return (rendered * 255).astype(np.uint8)


def _build_pose_dictionary(meta: dict[str, Any], mhr_parameters) -> dict[str, Any]:
    return {
        "mhr_parameters": _to_numpy(mhr_parameters).tolist(),
        "pred_cam": _to_numpy(meta["pred_cam"]).tolist(),
        "pred_cam_t": _to_numpy(meta["pred_cam_t"]).tolist(),
        "focal_length": _to_numpy(meta["focal_length"]).tolist(),
        "pred_keypoints_2d": meta["pred_keypoints_2d"],
        "image_path": meta["image_path"],
    }


def _save_triptych_matplotlib(
    original_image_bgr: np.ndarray,
    original_mhr_render_bgr: np.ndarray,
    reconstructed_mhr_render_bgr: np.ndarray,
    output_path: Path,
):
    images_rgb = [
        cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(original_mhr_render_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(reconstructed_mhr_render_bgr, cv2.COLOR_BGR2RGB),
    ]
    titles = ["Original Image", "Original MHR", "Reconstructed MHR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, image, title in zip(axes, images_rgb, titles, strict=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _get_runtime(project_root: str | Path):
    global _POSE_INTERPRETER
    global _RENDERER_CLASS

    if _POSE_INTERPRETER is not None and _RENDERER_CLASS is not None:
        return _POSE_INTERPRETER, _RENDERER_CLASS

    project_root = Path(project_root)
    sam3d_repo_path = project_root / "pose_module" / "sam3d"
    if str(sam3d_repo_path) not in sys.path:
        sys.path.insert(0, str(sam3d_repo_path))

    from pose_module.interpret_mhr_params import PoseDataInterpreter
    from sam_3d_body.visualization.renderer import Renderer

    _POSE_INTERPRETER = PoseDataInterpreter()
    _RENDERER_CLASS = Renderer
    return _POSE_INTERPRETER, _RENDERER_CLASS


class ReconstructionImageList:
    """JSON-driven list of pose image paths used for reconstruction triptychs."""

    DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "reconstruction_images.json"

    def __init__(self, json_path: str | Path | None = None) -> None:
        self.json_path = Path(json_path) if json_path is not None else self.DEFAULT_JSON_PATH

    def get_image_paths(self, data_dir: str | Path) -> list[str]:
        """Resolved absolute paths (in JSON order) that exist on disk."""
        return load_resolved_image_paths_from_json(self.json_path, Path(data_dir))


class ReconstructionVisualizer:
    """Render original vs reconstructed MHR meshes for an ordered list of sample dicts."""

    def save(
        self,
        samples: list[dict[str, Any]],
        epoch: int,
        output_dir: str | Path,
        project_root: str | Path,
        data_dir: str | Path,
    ) -> list[str]:
        if len(samples) == 0:
            return []

        pose_interpreter, renderer_class = _get_runtime(project_root=project_root)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faces_path = Path(project_root) / "visualization" / "faces.json"
        with faces_path.open("r", encoding="utf-8") as f:
            faces = np.array(json.load(f))

        saved_paths: list[str] = []
        for idx, sample in enumerate(samples, start=1):
            img_path = _resolve_image_path(sample, data_dir=data_dir)
            if img_path is None:
                continue

            original_image = cv2.imread(img_path)
            if original_image is None:
                continue

            original_pose_data = pose_interpreter.interpret_pose_dictionary(
                _build_pose_dictionary(sample, sample["original_mhr_parameters"])
            )
            reconstructed_pose_data = pose_interpreter.interpret_pose_dictionary(
                _build_pose_dictionary(sample, sample["reconstructed_mhr_parameters"])
            )

            original_mhr_render = _render_mesh_overlay(
                original_image,
                original_pose_data,
                renderer_class,
                faces,
            )
            reconstructed_mhr_render = _render_mesh_overlay(
                original_image,
                reconstructed_pose_data,
                renderer_class,
                faces,
            )

            out_path = output_dir / f"epoch_{epoch + 1}_sample_{idx}.png"
            _save_triptych_matplotlib(
                original_image_bgr=original_image,
                original_mhr_render_bgr=original_mhr_render,
                reconstructed_mhr_render_bgr=reconstructed_mhr_render,
                output_path=out_path,
            )
            saved_paths.append(str(out_path))

        return saved_paths


def save_reconstruction_visualizations(
    samples: list[dict[str, Any]],
    epoch: int,
    output_dir: str | Path,
    project_root: str | Path,
    data_dir: str | Path,
    image_list_json_path: str | Path | None = None,
) -> list[str]:
    """
    Backward-compatible wrapper: ``image_list_json_path`` is ignored; pass pre-filtered
    ``samples`` from the training loop (see ``ReconstructionImageList`` + dataset batching).
    """
    _ = image_list_json_path
    return ReconstructionVisualizer().save(
        samples=samples,
        epoch=epoch,
        output_dir=output_dir,
        project_root=project_root,
        data_dir=data_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Debug one-sample joint->MHR reconstruction round-trip. "
            "Uses original joint angles as reconstructed output."
        )
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Dataset row index to test.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    parquet_path = data_dir / "processed_poses.parquet"
    joint_names_path = project_root / "vae_features" / "train" / "joint_names.json"
    skeleton_path = project_root / "vae_features" / "train" / "mhr_skeleton_format.json"
    output_dir = project_root / "vae_features" / "train" / "validation" / "reconstruction"
    test_output_path = output_dir / "test_reconstruction.png"

    skeleton_format = SkeletonFormat.from_json_file(skeleton_path)
    dataset = MHRPoseDataset(
        parquet_path=parquet_path,
        skeleton_format=skeleton_format,
        joint_names_path=joint_names_path,
        data_root=data_dir,
        device=torch.device("cpu"),
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index out of bounds: {args.index}, dataset size={len(dataset)}")

    single_item = dataset[args.index]
    batch = dataset.collate_fn([single_item])
    metadata = batch.metadata[0]

    # Debug round-trip: "reconstructed" joint angles are exactly the original ones.
    reconstructed_joint_angles = batch.joint_angles
    inverse_idx = dataset.inverse_reorder_indices.to(reconstructed_joint_angles.device)
    reconstructed_joint_angles_mhr_order = reconstructed_joint_angles[:, inverse_idx, :]
    _, reconstructed_mhr_params = dataset.post_processor.joint_angles_to_mhr_parameters(
        joint_angles_mhr_order=reconstructed_joint_angles_mhr_order,
        base_raw=batch.joint_data.raw,
    )

    mhr_mse = torch.mean((reconstructed_mhr_params[0] - metadata.mhr_parameters) ** 2).item()
    print(f"MHR parameter MSE at index {args.index}: {mhr_mse:.8f}")

    sample = {
        "image_path": metadata.image_path,
        "image_path_abs": metadata.image_path_abs,
        "pred_cam": metadata.pred_cam,
        "pred_cam_t": metadata.pred_cam_t,
        "focal_length": metadata.focal_length,
        "pred_keypoints_2d": metadata.pred_keypoints_2d,
        "original_mhr_parameters": metadata.mhr_parameters,
        "reconstructed_mhr_parameters": reconstructed_mhr_params[0],
    }

    paths = ReconstructionVisualizer().save(
        samples=[sample],
        epoch=0,
        output_dir=output_dir,
        project_root=project_root,
        data_dir=data_dir,
    )
    if len(paths) == 0:
        print("No reconstruction image was produced.")
    else:
        produced_path = Path(paths[0])
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        produced_path.replace(test_output_path)
        print(f"Saved reconstruction debug image: {test_output_path}")


if __name__ == "__main__":
    main()
