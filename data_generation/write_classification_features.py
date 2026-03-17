import os
import pandas as pd
from pathlib import Path

import tqdm
from classification_features.inference import PoseClassificationFeatures

CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"


def _get_features_dataframe(
    poses_dataframe: pd.DataFrame, existingDataframe: pd.DataFrame
) -> pd.DataFrame:
    data = []
    extractor = PoseClassificationFeatures()

    progress_bar = tqdm.tqdm(poses_dataframe.iterrows(), "Processing...", total=len(poses_dataframe))
    for _, row in progress_bar:
        relative_image_path = row["image_path"]

        # Skip if already processed
        if (
            not existingDataframe.empty
            and relative_image_path in existingDataframe["image_path"].values
        ):
            continue

        keypoints_2d = row["pred_keypoints_2d"]
        embedding = extractor.extract_embedding(
            keypoints_2d
        )  # This is your numpy array

        # Append as a dictionary
        data.append(
            {"image_path": relative_image_path, "classification_features": embedding}
        )

    # Create the final dataframe
    return pd.DataFrame(data)


def _write_poses(data_path: str):
    classification_features_path = os.path.join(
        data_path, "classification_features.parquet"
    )
    if os.path.exists(classification_features_path):
        existingDataframe = pd.read_parquet(classification_features_path)
    else:
        existingDataframe = pd.DataFrame([])

    poses_path = os.path.join(data_path, "processed_poses.parquet")
    poses_dataframe = pd.read_parquet(poses_path)

    new_dataframe = _get_features_dataframe(poses_dataframe, existingDataframe)
    combined_dataframe = pd.concat([existingDataframe, new_dataframe])
    combined_dataframe.to_parquet(classification_features_path, compression="zstd")


def main():
    print("Starting to read and process poses with sam 3d body...")
    _write_poses(DATA_PATH)
    print("Successfully read and processed poses with sam 3d body...")


if __name__ == "__main__":
    main()
