"""
Modal wrapper for PoseC3D 2D pose embedding extraction.
"""
from pathlib import Path
from typing import Dict, Tuple


import numpy as np
import torch

import mmengine
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmaction.apis import init_recognizer
from ..pose_module.sam3d.sam_3d_body.metadata.mhr70 import mhr_names


# MHR70 format provided:
# 9: left-hip, 10: right-hip, 11: left-knee, 12: right-knee, 13: left-ankle, 14: right-ankle
# 41: right-wrist, 62: left-wrist

MHR70_TO_COCO_MAPPING = {
    0: 0,   # nose -> nose
    1: 1,   # left-eye -> left_eye
    2: 2,   # right-eye -> right_eye
    3: 3,   # left-ear -> left_ear
    4: 4,   # right-ear -> right_ear
    5: 5,   # left-shoulder -> left_shoulder
    6: 6,   # right-shoulder -> right_shoulder
    7: 7,   # left-elbow -> left_elbow
    8: 8,   # right-elbow -> right_elbow
    62: 9,  # left-wrist -> left_wrist (Corrected)
    41: 10,  # right-wrist -> right_wrist (Corrected)
    9: 11,  # left-hip -> left_hip
    10: 12,  # right-hip -> right_hip
    11: 13,  # left-knee -> left_knee
    12: 14,  # right-knee -> right_knee
    13: 15,  # left-ankle -> left_ankle
    14: 16,  # right-ankle -> right_ankle
}


def _load_model():
    """Load PoseC3D model from checkpoint."""
    # Should be backend/ locally and /root on modal
    parent = Path(__file__).resolve().parent.parent
    config_path = str(parent / 'classification_features' / 'configs' / 'skeleton' / 'posec3d' /
                      'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py')
    checkpoint_path = str(parent / 'checkpoints' / 'posec3d' /
                          'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth')

    config = mmengine.Config.fromfile(config_path)
    init_default_scope(config.get('default_scope', 'mmaction'))

    # Disable pretrained weights if specified
    if hasattr(config.model, 'backbone') and config.model.backbone.get('pretrained', None):
        config.model.backbone.pretrained = None

    model = init_recognizer(config, checkpoint_path, device='cuda:0')
    model.eval()

    return model, config


class PoseClassificationFeatures:
    def __init__(self) -> None:
        """Initialize the PoseC3D model."""
        print("Loading PoseC3D model...")
        self.model, self.config = _load_model()

        # Get test pipeline from config
        init_default_scope(self.config.get('default_scope', 'mmaction'))
        self.test_pipeline = Compose(self.config.test_pipeline)

        print("PoseC3D model loaded successfully!")

    def extract_embedding(
            self,
            pose_dict: Dict[str, Tuple[float, float]],
            img_shape: Tuple[int, int] = (480, 640),
    ) -> np.ndarray:
        """
        Extract embedding from a 2D pose dictionary.

        Args:
            pose_dict: Dictionary mapping MHR70 joint names to (x, y) coordinates
                      from SAM 3D Body output.
            img_shape: Image shape (height, width) used for normalization.
                      Defaults to (480, 640).

        Returns:
            Embedding vector as numpy array.
        """
        # Handle empty pose dict
        if not pose_dict:
            # Return zero embedding (512 dim from backbone)
            return np.zeros(512, dtype=np.float32)

        # Map MHR70 joints to COCO 17 format
        coco_keypoints = np.zeros((17, 2), dtype=np.float32)
        coco_scores = np.ones(17, dtype=np.float32)

        for mhr_idx, coco_idx in MHR70_TO_COCO_MAPPING.items():
            joint_name = mhr_names[mhr_idx]
            if joint_name in pose_dict:
                x, y = pose_dict[joint_name]
                coco_keypoints[coco_idx] = [x, y]
            else:
                # Missing joint - set score to 0
                coco_scores[coco_idx] = 0.0

        # Convert to format expected by PoseC3D
        # Format: [M x T x V x C] where M=persons, T=frames, V=keypoints, C=coords
        # For single frame, repeat 48 times to match PoseC3D's expected temporal dimension
        num_frames = 48
        num_persons = 1
        num_keypoints = 17

        # Create keypoints: [T, M, V, C] then transpose to [M, T, V, C]
        keypoints = np.tile(coco_keypoints[np.newaxis, np.newaxis, :, :],
                            (num_frames, num_persons, 1, 1))
        keypoint_scores = np.tile(coco_scores[np.newaxis, np.newaxis, :],
                                  (num_frames, num_persons, 1))

        # Transpose to [M, T, V, C] format
        keypoints = keypoints.transpose((1, 0, 2, 3))  # [M, T, V, C]
        keypoint_scores = keypoint_scores.transpose((1, 0, 2))  # [M, T, V]

        # Create fake annotation dict
        h, w = img_shape
        fake_anno = dict(
            frame_dict='',
            label=-1,
            img_shape=(h, w),
            origin_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=num_frames,
            keypoint=keypoints,  # [M, T, V, C] = [1, 48, 17, 2]
            keypoint_score=keypoint_scores,  # [M, T, V] = [1, 48, 17]
        )

        # Process through test pipeline
        data = self.test_pipeline(fake_anno)
        data = pseudo_collate([data])

        # Extract features
        # pseudo_collate returns a dict with 'inputs' (list) and 'data_samples' keys
        # Convert inputs list to tensor if needed
        with torch.no_grad():
            inputs = data['inputs']
            # If inputs is a list, stack it into a tensor
            if isinstance(inputs, list):
                inputs = torch.stack(inputs)
            inputs = inputs.to('cuda:0')
            # Extract backbone features
            features, _ = self.model.extract_feat(inputs,
                                                  stage='backbone',
                                                  test_mode=True)  # type: ignore

            # Average pool spatial and temporal dimensions
            # features shape: [batch_size, channels, temporal, spatial_h, spatial_w]
            # We want to get a single embedding vector per sample
            if len(features.shape) == 5:
                # Global average pooling over spatial and temporal dimensions
                embedding = features.mean(dim=(2, 3,
                                               4))  # [batch_size, channels]
            else:
                # Fallback: flatten and average
                embedding = features.view(features.size(0), features.size(1),
                                          -1).mean(dim=2)

            # Convert to numpy and return first (and only) sample
            embedding_np = embedding[0].cpu().numpy().astype(np.float32)
            return embedding_np
