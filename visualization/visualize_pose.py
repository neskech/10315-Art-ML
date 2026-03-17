import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

# Allow imports from the sam3d repo to work
parent_dir = Path(__file__).resolve().parent.parent
sam3d_repo_path = parent_dir / "pose_module" / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))

from pose_module.inference import SAM3DBodyInference
from pose_module.interpret_mhr_params import PoseData, PoseDataInterpreter
from pose_module.sam3d.tools.vis_utils import visualize_sample_together, visualize_on_white

CURRENT_DIRECTORY = Path(__file__).parent
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"
PARQUET_PATH = CURRENT_DIRECTORY.parent / "data" / "processed_poses.parquet"


def visualize_sample(pose_data: PoseData, output_filename: str, mode: str):
    visuals_dir = CURRENT_DIRECTORY / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    full_path = os.path.join(POSES_DIRECTORY, pose_data.relative_image_path)
    image = cv2.imread(full_path)
    
    if image is None:
        # Fallback for when the image-path provided is absolute/local rather than in the data dir
        image = cv2.imread(pose_data.relative_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {full_path}")

    with open(CURRENT_DIRECTORY / "faces.json", "r") as f:
        faces = np.array(json.load(f))

    if mode == "white":
        rendered_image = visualize_on_white(image, [asdict(pose_data)], faces)
    else:
        rendered_image = visualize_sample_together(image, [asdict(pose_data)], faces)

    save_path = visuals_dir / output_filename
    cv2.imwrite(str(save_path), rendered_image.astype(np.uint8))
    print(f"Successfully saved {mode} visualization to {save_path}")


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
        choices=["together", "white"],
        default="together",
        help="Visualization mode: 'together' (grid) or 'white' (3D model on white background).",
    )
    args = parser.parse_args()

    estimator = SAM3DBodyInference(torch.device("cuda"), use_torch_compile=False)
    image = cv2.imread(args.image_path)
    
    if image is None:
        raise Exception(f"Could not read image at {args.image_path}")
        
    outputs = estimator.predict([image], use_bbox_detector=False)

    interpreter = PoseDataInterpreter()
    output = outputs[0]
    output["image_path"] = args.image_path
    pose_data = interpreter.interpret_pose_dictionary(output)
    
    visualize_sample(pose_data, args.output, args.mode)


if __name__ == "__main__":
    main()