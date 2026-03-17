import argparse
import os
import cv2
import pandas as pd
from pathlib import Path
import torch
import tqdm
from pose_module.inference import SAM3DBodyInference

CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"


def _predict_poses(
    poses_path: str,
    existingDataframe: pd.DataFrame,
    batch_size: int,
    processing_limit: int,
) -> pd.DataFrame:
    data = []
    estimator = SAM3DBodyInference(device=torch.device('cuda'), use_torch_compile=False)

    image_paths = []
    for root, _, files in os.walk(poses_path):
        for file in files:
            path = os.path.join(root, file)
            relative_path = path.removeprefix(poses_path)

            if not path.endswith(".png"):
                continue

            if (
                not existingDataframe.empty
                and relative_path in existingDataframe["image_path"].values
            ):
                print(f"{relative_path} has already been processed!")
                continue

            image_paths.append(os.path.join(root, file))

    if processing_limit != -1:
        image_paths = image_paths[:processing_limit]

    progress_bar = tqdm.tqdm(
        range(0, len(image_paths), batch_size),
        "Processing batches...",
        total=len(image_paths) // batch_size,
    )
    for i in progress_bar:
        batch_end = min(i + batch_size, len(image_paths))
        batch = image_paths[i:batch_end]

        cv2_images = [cv2.imread(path) for path in batch]
        filtered_cv2_images = [img for img in cv2_images if img is not None]
        outputs = estimator.predict(filtered_cv2_images, use_bbox_detector=True)

        for output, path in zip(outputs, batch):
            relative_path = path.removeprefix(poses_path)
            output["image_path"] = relative_path

        data += outputs

    return pd.DataFrame(data)


def _write_poses(
    data_path: str,
    batch_size: int,
    processing_limit: int,
):
    processed_data_path = os.path.join(data_path, "processed_poses.csv")
    if os.path.exists(processed_data_path):
        existingDataframe = pd.read_csv(processed_data_path)
    else:
        existingDataframe = pd.DataFrame([])

    poses_path = os.path.join(data_path, "poses/")
    new_dataframe = _predict_poses(
        poses_path, existingDataframe, batch_size, processing_limit
    )
    combined_dataframe = pd.concat([existingDataframe, new_dataframe])

    combined_dataframe.to_csv(processed_data_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process poses using SAM 3D Body Inference."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images to process in a single batch (default: 1).",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=-1,
        help="Maximum number of new images to process. Set to -1 for no limit (default: -1).",
    )

    args = parser.parse_args()

    print("Starting to read and process poses with sam 3d body...")
    _write_poses(DATA_PATH, args.batch_size, args.image_limit)
    print("Successfully read and processed poses with sam 3d body...")


if __name__ == "__main__":
    main()
