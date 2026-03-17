"""
Modal wrapper for SAM 3D Body 2D pose inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from pose_module.inferFull import infer_full
from pose_module.preprocess import prepare_batch_correctly
from pose_module.sam3d.sam_3d_body.data.transforms.common import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from pose_module.sam3d.sam_3d_body.utils.dist import recursive_to
from sam3d.sam_3d_body import load_sam_3d_body
from sam3d.sam_3d_body.metadata.mhr70 import mhr_names
from sam3d.tools.build_detector import HumanDetector
import numpy as np
import torch
from torchvision.transforms import ToTensor


class SAM3DBodyInference:
    def __init__(self, device: torch.device, use_torch_compile: bool) -> None:
        self.device = device
        self.model, self.model_cfg = _load_model()

        # Store joint names for keypoint mapping
        self.joint_names = mhr_names

        if use_torch_compile:
            print("Applying torch compile to model...")
            self.model = torch.compile(self.model, "max-autotune")

        # Sam3 is better but bigger and slower, so we opt not to use it
        self.human_detector = HumanDetector(name="vitdet", device=self.device)

        # Transform applied to each batch element
        self.target_image_size = self.model_cfg.MODEL.IMAGE_SIZE
        self.transform_batch_element = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(
                    input_size=self.target_image_size,
                    use_udp=False,
                ),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        # Internal transform used by sam-3d-body
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(
                    input_size=self.target_image_size,
                    use_udp=False,
                ),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def predict(
        self,
        images: list[np.ndarray],
        use_bbox_detector: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict 3D pose keypoints, root translation, root rotation, and 3D
        euler angle joint angles from an imahe

        Args:
            images: Input image list of numpys array in RGB format (H, W, 3)
            use_bbox_detector: If True, use bounding box detector to find person.
                             If False, use full image as bounding box.

        Returns:
            Dictionary mapping joint names to (x, y, z) coordinates,
            root translation, root rotation, and joint angles dictionary
            mapping joint names to (x, y, z) relative euler angles.
            Returns None if no person is detected.
        """
        batch_size = len(images)

        # Prepare batch
        detector = self.human_detector if use_bbox_detector else None
        input_batch = prepare_batch_correctly(
            images,
            self.transform_batch_element,
            detector,
        )

        # Transfer to device and initialize
        input_batch = recursive_to(input_batch, self.device)
        self.model._initialize_batch(input_batch)

        # Remove the singleton person dimension
        image_batch = input_batch["img"].squeeze(1)
        outputs = infer_full(
            self.model, image_batch, input_batch, self.transform_hand, self.device
        )

        output_dictionaries = []
        for i in range(batch_size):
            print("FUCKING SHAPE", outputs["pred_keypoints_3d"].shape)
            # Extract 3d and 2d keypoints
            keypoints_3d = outputs["pred_keypoints_3d"][i]  # Shape: [70, 2]
            assert len(self.joint_names == keypoints_3d.shape[0])
            keypoints_2d = outputs["pred_keypoints_2d"][i]
            assert len(self.joint_names == keypoints_2d.shape[0])

            # Map keypoints to joint names
            keypoints_3d_dict = {}
            for idx, joint_name in enumerate(self.joint_names):
                x = keypoints_3d[idx][0]
                y = keypoints_3d[idx][1]
                z = keypoints_3d[idx][2]
                keypoints_3d_dict[joint_name] = (x, y, z)

            # Map keypoints to joint names
            keypoints_2d_dict = {}
            for idx, joint_name in enumerate(self.joint_names):
                x = keypoints_2d[idx][0]
                y = keypoints_2d[idx][1]
                keypoints_2d_dict[joint_name] = (x, y)

            # Extract MHR parameters
            mhr_parameters = outputs["mhr_model_params"][i]
            root_translation = (mhr_parameters[0], mhr_parameters[1], mhr_parameters[2])
            root_rotation = (mhr_parameters[3], mhr_parameters[4], mhr_parameters[5])
            joint_angles_dict = {
                joint_name: (
                    mhr_parameters[6 + i * 3 + 0],
                    mhr_parameters[6 + i * 3 + 1],
                    mhr_parameters[6 + i * 3 + 2],
                )
                for i, joint_name in enumerate(self.joint_names)
            }

            output_dictionaries.append(
                {
                    "keypoints_3d": keypoints_3d_dict,
                    "keypoints_2d": keypoints_2d,
                    "root_translation": root_translation,
                    "root_rotation": root_rotation,
                    "joint_angles": joint_angles_dict,
                }
            )

        return output_dictionaries


def _load_model():
    parent = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = str(
        parent / "checkpoints" / "sam3d" / "sam-3d-body-dinov3" / "model.ckpt"
    )
    MHR_PATH = str(
        parent / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    )
    return load_sam_3d_body(checkpoint_path=CHECKPOINT_PATH, mhr_path=MHR_PATH)
