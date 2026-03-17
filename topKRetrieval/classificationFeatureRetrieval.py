
from pose_module.interpret_mhr_params import PoseData


def classificationFeatureMetric(pose1: PoseData, pose2: PoseData):
    return (pose1.pred_keypoints_3d - pose2.pred_keypoints_3d).square().sum()
