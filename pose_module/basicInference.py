import sys
from pathlib import Path

from pose_module.sam3d.tools.build_detector import HumanDetector

current_dir = Path(__file__).resolve().parent
sam3d_repo_path = current_dir / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))

from typing import Any, Dict, List  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pose_module.sam3d.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402
from pose_module.sam3d.sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402


class SAM3DBodyInferenceBasic:
    def __init__(
        self,
        device: torch.device,
        use_human_detector: bool,
        use_torch_compile: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.model_cfg = _load_model()
        self.model = self.model.to(self.device).eval()
        self.joint_names = mhr_names
        self.human_detector = (
            HumanDetector(name="vitdet", device=self.device)
            if use_human_detector
            else None
        )

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=self.human_detector,
            human_segmentor=None,
            fov_estimator=None,
        )

        if use_torch_compile:
            self.model = torch.compile(self.model, mode="max-autotune")

    @torch.no_grad()
    def predict(
        self,
        images: List[np.ndarray],
        use_bbox_detector: bool = True,
    ) -> List[Dict[str, Any]]:
        output_dictionaries = []

        for img in images:
            img_rgb = np.asarray(img).copy()
            self.estimator.detector = self.human_detector if use_bbox_detector else None
            outputs = self.estimator.process_one_image(img_rgb)

            if not outputs:
                continue

            person_output = outputs[0]
            mhr_parameters = person_output["mhr_model_params"].tolist()

            focal_length = person_output["focal_length"]

            keypoints_2d = person_output["pred_keypoints_2d"]
            keypoints_2d_dict = {}
            for idx, joint_name in enumerate(self.joint_names):
                keypoints_2d_dict[joint_name] = (
                    float(keypoints_2d[idx][0]),
                    float(keypoints_2d[idx][1]),
                )

            output_dictionaries.append(
                {
                    "mhr_parameters": mhr_parameters,
                    "pred_cam": person_output["pred_pose_raw"].tolist(),
                    "pred_cam_t": person_output["pred_cam_t"].tolist(),
                    "focal_length": focal_length,
                    "pred_keypoints_2d": keypoints_2d_dict,
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
