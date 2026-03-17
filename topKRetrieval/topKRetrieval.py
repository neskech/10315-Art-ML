import os
from pathlib import Path
import cv2
import pandas as pd
from pose_module.inference import SAM3DBodyInference
from pose_module.interpret_mhr_params import PoseData, PoseDataInterpreter
import torch


CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"
POSES_PATH = DATA_PATH / "poses"


def _topKRetrieval(
    inputPose: PoseData,
    poses_dataframe: pd.DataFrame,
    pose_intepreter: PoseDataInterpreter,
    distanceFunction,
    k: int,
) -> list[PoseData]:
    """
    Retrieves closest k poses to inputPose

    :param inputPose: User pose data
    :param otherPoses: Dictionary of pose dataset
    :param distanceFunction: Metric to calculate distance between poses
    :param k: Number of desired poses to ouput
    """

    pose_data_with_scores = []

    # Dictionary mapping poses in otherPoses with relative distance to inputPose
    for _, pose_dict in poses_dataframe.iterrows():
        data = pose_intepreter.interpret_pose_dictionary(pose_dict)
        relDist = distanceFunction(inputPose, data)
        pose_data_with_scores.append((data, relDist))

    pose_data_with_scores.sort(key=lambda x: x[1])

    return [x[0] for x in pose_data_with_scores[:k]]


def runTopKRetrieval(pose_image_path: str, distanceFunction, k: int) -> list[PoseData]:
    parquet_path = os.path.join(DATA_PATH, "processed_poses.parquet")
    dataframe = pd.read_parquet(parquet_path)

    estimator = SAM3DBodyInference(torch.device("cuda"), use_torch_compile=False)
    image = cv2.imread(pose_image_path)
    outputs = estimator.predict([image], use_bbox_detector=False)

    intepreter = PoseDataInterpreter()
    output = outputs[0]
    output["image_path"] = pose_image_path.removeprefix(str(POSES_PATH))
    poseData = intepreter.interpret_pose_dictionary(output)

    return poseData, _topKRetrieval(
        poseData, dataframe, intepreter, distanceFunction, k
    )
