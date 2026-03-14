"""
Modal wrapper for SAM 3D Body 2D pose inference.
"""
from pathlib import Path
from typing import Any, Dict, Optional
from sam3d.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam3d.sam_3d_body.metadata.mhr70 import mhr_names
from sam3d.tools.build_detector import HumanDetector
import numpy as np


class SAM3DBodyInference:
    def __init__(self) -> None:
        self.model, self.model_cfg = _load_model()

        # Store joint names for keypoint mapping
        self.joint_names = mhr_names

        self.human_detector = HumanDetector(name="vitdet", device="cuda")
        print("SAM 3D Body model loaded successfully!")

    def predict_2d_pose(
        self,
        image: np.ndarray,
        use_bbox_detector: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict 3D pose keypoints, root translation, root rotation, and 3D
        euler angle joint angles from an imahe

        Args:
            image: Input image as numpy array in RGB format (H, W, 3)
            use_bbox_detector: If True, use bounding box detector to find person.
                             If False, use full image as bounding box.

        Returns:
            Dictionary mapping joint names to (x, y, z) coordinates,
            root translation, root rotation, and joint angles dictionary
            mapping joint names to (x, y, z) relative euler angles.
            Returns None if no person is detected.
        """
        # Validate and convert image to numpy array
        img = np.asarray(image).copy()

        # Validate image shape
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got {img.shape}")

        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=self.human_detector if use_bbox_detector else None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # Process image
        outputs = estimator.process_one_image(img)

        # Handle no person detected
        if not outputs or len(outputs) == 0:
            return None

        # Get first person's output
        person_output = outputs[0]

        # Extract 3d keypoints
        keypoints_3d = person_output["pred_keypoints_3d"]  # Shape: [70, 2]
        assert len(self.joint_names == keypoints_3d.shape[0])

        # Map keypoints to joint names
        keypoints_3d_dict = {}
        for idx, joint_name in enumerate(self.joint_names):
            x = keypoints_3d[idx][0]
            y = keypoints_3d[idx][1]
            z = keypoints_3d[idx][2]
            keypoints_3d_dict[joint_name] = (x, y, z)

        # Extract MHR parameters
        mhr_parameters = person_output["mhr_model_params"]
        root_translation = (mhr_parameters[0], mhr_parameters[1], mhr_parameters[2])
        root_rotation = (mhr_parameters[3], mhr_parameters[4], mhr_parameters[5])
        joint_angles_dict = {
            joint_name: (mhr_parameters[6 + i * 3 + 0],
                         mhr_parameters[6 + i * 3 + 1],
                         mhr_parameters[6 + i * 3 + 2],)
            for i, joint_name in enumerate(self.joint_names)
        }

        return {
            'keypoints': keypoints_3d_dict,
            'root_translation': root_translation,
            'root_rotation': root_rotation,
            'joint_angles_dict': joint_angles_dict
        }


def _load_model():
    parent = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = str(parent / 'checkpoints' / 'sam-3d-body-dinov3' /
                          'model.ckpt')
    MHR_PATH = str(parent / 'checkpoints' / 'sam-3d-body-dinov3' / 'assets' /
                   'mhr_model.pt')
    return load_sam_3d_body(checkpoint_path=CHECKPOINT_PATH, mhr_path=MHR_PATH)
