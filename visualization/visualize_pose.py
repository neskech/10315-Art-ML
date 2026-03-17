import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# Allow imports from the sam3d repo to work
parent_dir = Path(__file__).resolve().parent.parent
sam3d_repo_path = parent_dir / "pose_module" / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))


from pose_module.interpret_mhr_params import PoseData, PoseDataInterpreter  # noqa: E402
from pose_module.sam3d.tools.vis_utils import visualize_sample_together  # noqa: E402

CURRENT_DIRECTORY = Path(__file__).parent
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"
CSV_PATH = CURRENT_DIRECTORY.parent / "data" / "processed_poses.csv"


def visualize_sample(poseData: PoseData):
    visuals_dir = CURRENT_DIRECTORY / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    full_path = os.path.join(POSES_DIRECTORY, poseData.relative_image_path)
    image = cv2.imread(full_path)

    with open(CURRENT_DIRECTORY / "faces.json", "r") as f:
        faces = np.array(json.load(f))

    rendered_image = visualize_sample_together(
        image, [asdict(poseData)], faces
    )

    save_path = visuals_dir / "sample.png"
    cv2.imwrite(str(save_path), rendered_image.astype(np.uint8))
    print(f"Successfully saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D pose from a processed CSV."
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        required=True,
        help="The relative image path to visualize (e.g., 'gesture/1234.png').",
    )
    args = parser.parse_args()

    target_path = args.image_path

    df = pd.read_csv(CSV_PATH)
    matched_rows = df[df["image_path"] == target_path]

    if matched_rows.empty:
        print(f"Error: No data found for image path '{target_path}' in the CSV.")
        return

    row_dict = matched_rows.iloc[0].to_dict()
    interpreter = PoseDataInterpreter()
    pose_data = interpreter.interpret_pose_dictionary(row_dict)

    visualize_sample(pose_data)


if __name__ == "__main__":
    main()
