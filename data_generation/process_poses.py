import os
import cv2
import pandas as pd

from pose_module.inference import SAM3DBodyInference

DATA_PATH = '../data/'


def _predict_poses(poses_path: str, existingDataframe: pd.DataFrame) -> pd.DataFrame:
    data = []
    estimator = SAM3DBodyInference()

    for root, _, files in os.walk(poses_path):
        for file in files:
            if not file.endswith('.png'):
                continue

            path = os.path.join(root, file)
            relative_path = path.removeprefix(poses_path)
            if not existingDataframe.empty and relative_path in existingDataframe['image_path'].values:
                print(f"{relative_path} has already been processed!")
                continue

            image = cv2.imread(path)
            if not image:
                print(f"Failed to read {relative_path}!")
                continue

            output = estimator.predict(image)
            if not output:
                print(f"Sam-3D-Body could not find any people in {relative_path}!")
                continue

            output['image_path'] = relative_path
            data.append(output)

    return pd.DataFrame(data)


def _write_poses(data_path: str):
    processed_data_path = os.path.join(data_path, 'processed_poses.csv')
    if os.path.exists(processed_data_path):
        existingDataframe = pd.read_csv(processed_data_path)
    else:
        existingDataframe = pd.DataFrame([])

    poses_path = os.path.join(data_path, 'poses/')
    new_dataframe = _predict_poses(poses_path, existingDataframe)
    combined_dataframe = pd.concat([existingDataframe, new_dataframe])

    combined_dataframe.to_csv(processed_data_path)


def main():
    print("Starting to read and process poses with sam 3d body...")
    _write_poses(DATA_PATH)
    print("Successfully read and processed poses with sam 3d body...")


if __name__ == '__main__':
    main()
