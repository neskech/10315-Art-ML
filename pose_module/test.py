import sys
import os
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)
# Allow imports from the sam3d repo to work
current_dir = Path(__file__).resolve().parent
sam3d_repo_path = current_dir / "sam3d"

if str(sam3d_repo_path) not in sys.path:
    sys.path.insert(0, str(sam3d_repo_path))


from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import mhr_names
from tools.vis_utils import visualize_sample_together
from tools.build_detector import HumanDetector


class SAM3DInferencePipeline:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

        parent = Path(__file__).resolve().parent.parent
        checkpoint_path = str(
            parent / "checkpoints" / "sam3d" / "dinov3" / "model.ckpt"
        )
        mhr_path = str(
            parent / "checkpoints" / "sam3d" / "dinov3" / "assets" / "mhr_model.pt"
        )

        print(f"Loading SAM 3D Body from: {checkpoint_path}")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path, mhr_path=mhr_path, device=device
        )
        self.model.eval()
        self.joint_names = mhr_names
        self.human_detector = None # HumanDetector(name="vitdet", device=self.device)

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=self.human_detector,
            human_segmentor=None,
            fov_estimator=None,
        )

    def run_and_visualize(self, image_path: str, output_path: str):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        print(f"Processing: {image_path}")
        outputs = self.estimator.process_one_image(
            img_bgr,
            bbox_thr=0.5,
            use_mask=False,
        )

        if not outputs:
            print("No persons detected.")
            return []

        processed_results = []
        for out in outputs:
            kp2d = out["pred_keypoints_2d"]
            kp2d_dict = {
                name: (float(kp2d[i][0]), float(kp2d[i][1]))
                for i, name in enumerate(self.joint_names)
            }

            processed_results.append(
                {
                    "mhr_parameters": out["mhr_model_params"].tolist(),
                    "pred_cam": out["pred_pose_raw"].tolist(),
                    "pred_cam_t": out["pred_cam_t"].tolist(),
                    "focal_length": float(out["focal_length"]),
                    "pred_keypoints_2d": kp2d_dict,
                    "bbox": out["bbox"].tolist(),
                }
            )

        print("Rendering results...")
        rend_img = visualize_sample_together(img_bgr, outputs, self.estimator.faces)
        cv2.imwrite(output_path, rend_img.astype(np.uint8))
        print(f"Visualization saved to {output_path}")

        return processed_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default="output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = SAM3DInferencePipeline(device=device)
    pipeline.run_and_visualize(args.image_path, args.output_path)


if __name__ == "__main__":
    main()
