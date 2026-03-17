"""
Modal wrapper for SAM 3D Body 2D pose inference.
"""

from pathlib import Path
import sys

from pose_module.sam3d.tools.build_detector import HumanDetector


# Allow imports from the sam3d repo to work
current_dir = Path(__file__).resolve().parent
sam3d_repo_path = current_dir / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))


from typing import Any, Dict, Optional  # noqa: E402
from pose_module.inferFull import infer_full  # noqa: E402
from pose_module.preprocess import prepare_batch_correctly  # noqa: E402
from pose_module.sam3d.sam_3d_body.data.transforms.common import (  # noqa: E402
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from pose_module.sam3d.sam_3d_body.utils.dist import recursive_to  # noqa: E402
from pose_module.sam3d.sam_3d_body import load_sam_3d_body  # noqa: E402
from pose_module.sam3d.sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.transforms import ToTensor  # noqa: E402


class SAM3DBodyInference:
    def __init__(self, device: torch.device, use_torch_compile: bool) -> None:
        self.device = device
        self.model, self.model_cfg = _load_model()
        self.model = self.model.eval()

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

    @torch.no_grad
    def predict(
        self,
        images: list[np.ndarray],
        use_bbox_detector: bool
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
        outputs = outputs["mhr"]

        output_dictionaries = []
        for i in range(batch_size):
            # Extract MHR parameters
            # See sam3d/sam_3d_body/models/heads/mhr_head.py
            # The first 3 params are root translation, then root rotation
            # then 130 euler angles. The rest are for bone length and body shape
            mhr_parameters = outputs["mhr_model_params"][i].tolist()

            # Extract camera parameters
            pred_cam = outputs["pred_cam"][i].tolist()
            pred_cam_t = outputs["pred_cam_t"][i].tolist()
            focal_length = outputs["focal_length"][i].tolist()

            # Extract 2D keypoints
            keypoints = outputs["pred_keypoints_2d"][i].tolist()
            keypoints_2d_dict = {}
            for idx, joint_name in enumerate(self.joint_names):
                x = keypoints[idx][0]
                y = keypoints[idx][1]
                keypoints_2d_dict[joint_name] = (x, y)

          

            output_dictionaries.append(
                {
                    "mhr_parameters": mhr_parameters,
                    "pred_cam": pred_cam,
                    "pred_cam_t": pred_cam_t,
                    "focal_length": focal_length,
                    "pred_keypoints_2d": keypoints_2d_dict
                }
            )
        return output_dictionaries


def _load_model():
    parent = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = str(parent / "checkpoints" / "sam3d" / "dinov3" / "model.ckpt")
    MHR_PATH = str(
        parent / "checkpoints" / "sam3d" / "dinov3" / "assets" / "mhr_model.pt"
    )
    return load_sam_3d_body(checkpoint_path=CHECKPOINT_PATH, mhr_path=MHR_PATH)
