import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# Allow imports from the sam3d repo to work
parent_dir = Path(__file__).resolve().parent.parent
sam3d_repo_path = parent_dir / "pose_module" / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))


def _load_runtime_dependencies():
    from pose_module.inference import SAM3DBodyInference
    from pose_module.interpret_mhr_params import PoseDataInterpreter
    from pose_module.sam3d.tools.vis_utils import (
        visualize_on_white,
        visualize_pose_with_camera_reference,
        visualize_sample_together,
    )

    return (
        SAM3DBodyInference,
        PoseDataInterpreter,
        visualize_on_white,
        visualize_pose_with_camera_reference,
        visualize_sample_together,
    )


(
    SAM3DBodyInference,
    PoseDataInterpreter,
    visualize_on_white,
    visualize_pose_with_camera_reference,
    visualize_sample_together,
) = _load_runtime_dependencies()

CURRENT_DIRECTORY = Path(__file__).parent
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"
PARQUET_PATH = CURRENT_DIRECTORY.parent / "data" / "processed_poses.parquet"


def _load_image(image_path: str):
    full_path = os.path.join(POSES_DIRECTORY, image_path)
    image = cv2.imread(full_path)

    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {full_path}")

    return image


def _load_faces():
    with open(CURRENT_DIRECTORY / "faces.json", "r") as f:
        return np.array(json.load(f))


def visualize_sample(pose_data, output_filename: str, mode: str):
    visuals_dir = CURRENT_DIRECTORY / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    image = _load_image(pose_data.relative_image_path)
    faces = _load_faces()

    if mode == "white":
        rendered_image = visualize_on_white(image, [asdict(pose_data)], faces)
    else:
        rendered_image = visualize_sample_together(image, [asdict(pose_data)], faces)

    save_path = visuals_dir / output_filename
    cv2.imwrite(str(save_path), rendered_image.astype(np.uint8))
    print(f"Successfully saved {mode} visualization to {save_path}")


def visualize_pose_in_camera_reference(
    camera_reference_pose_data,
    pose_reference_pose_data,
    output_filename: str,
):
    visuals_dir = CURRENT_DIRECTORY / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    camera_reference_image = _load_image(camera_reference_pose_data.relative_image_path)
    faces = _load_faces()

    rendered_image = visualize_pose_with_camera_reference(
        camera_reference_image,
        asdict(camera_reference_pose_data),
        asdict(pose_reference_pose_data),
        faces,
        white_background=True,
        center_on_camera=True,
    )

    save_path = visuals_dir / output_filename
    cv2.imwrite(str(save_path), rendered_image.astype(np.uint8))
    print(f"Successfully saved camera-reference visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D pose from a processed parquet."
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        required=True,
        help="The relative image path to visualize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sample.png",
        help="The output filename (default: sample.png).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["together", "white", "camera-reference"],
        default="together",
        help=(
            "Visualization mode: 'together' (grid), 'white' (3D model on white background), "
            "or 'camera-reference' (render pose image at camera image viewpoint)."
        ),
    )
    parser.add_argument(
        "--camera-image-path",
        type=str,
        default=None,
        help="Camera reference image path. Required when --mode is 'camera-reference'.",
    )
    args = parser.parse_args()

    estimator = SAM3DBodyInference(torch.device("cuda"), use_torch_compile=False)
    interpreter = PoseDataInterpreter()

    if args.mode == "camera-reference":
        if args.camera_image_path is None:
            parser.error(
                "--camera-image-path is required when --mode is 'camera-reference'."
            )

        camera_image = cv2.imread(args.camera_image_path)
        pose_image = cv2.imread(args.image_path)

        if camera_image is None:
            raise Exception(f"Could not read camera image at {args.camera_image_path}")
        if pose_image is None:
            raise Exception(f"Could not read pose image at {args.image_path}")

        outputs = estimator.predict([camera_image, pose_image], use_bbox_detector=False)

        camera_output = outputs[0]
        camera_output["image_path"] = args.camera_image_path
        camera_pose_data = interpreter.interpret_pose_dictionary(camera_output)

        pose_output = outputs[1]
        pose_output["image_path"] = args.image_path
        pose_pose_data = interpreter.interpret_pose_dictionary(pose_output)

        visualize_pose_in_camera_reference(
            camera_pose_data, pose_pose_data, args.output
        )
        return

    image = cv2.imread(args.image_path)

    if image is None:
        raise Exception(f"Could not read image at {args.image_path}")

    outputs = estimator.predict([image], use_bbox_detector=False)
    output = outputs[0]
    output["image_path"] = args.image_path
    pose_data = interpreter.interpret_pose_dictionary(output)

    visualize_sample(pose_data, args.output, args.mode)


if __name__ == "__main__":
    main()
