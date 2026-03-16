import os
import pandas as pd

from classification_features.inference import PoseClassificationFeatures

DATA_PATH = '../data/'


def _get_features_dataframe(poses_dataframe: pd.DataFrame, existingDataframe: pd.DataFrame) -> pd.DataFrame:
    data = []
    extractor = PoseClassificationFeatures()

    for _, row in poses_dataframe.iterrows():
        relative_image_path = row['image_path']

        # Skip if already processed
        if not existingDataframe.empty and relative_image_path in existingDataframe['image_path'].values:
            continue

        keypoints_2d = row['keypoints_2d']
        embedding = extractor.extract_embedding(keypoints_2d)  # This is your numpy array

        # Append as a dictionary
        data.append({
            'image_path': relative_image_path,
            'classification_features': embedding
        })

    # Create the final dataframe
    return pd.DataFrame(data)


def _write_poses(data_path: str):
    classification_features_path = os.path.join(data_path, 'classification_features.csv')
    if os.path.exists(classification_features_path):
        existingDataframe = pd.read_csv(classification_features_path)
    else:
        existingDataframe = pd.DataFrame([])

    classification_features_path = os.path.join(data_path, '')

    poses_path = os.path.join(data_path, 'processed_poses.csv')
    poses_dataframe = pd.read_csv(poses_path)

    new_dataframe = _get_features_dataframe(poses_dataframe, existingDataframe)
    combined_dataframe = pd.concat([existingDataframe, new_dataframe])
    combined_dataframe.to_csv(classification_features_path)


def main():
    print("Starting to read and process poses with sam 3d body...")
    _write_poses(DATA_PATH)
    print("Successfully read and processed poses with sam 3d body...")


if __name__ == '__main__':
    main()
