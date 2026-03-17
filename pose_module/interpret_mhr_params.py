import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PoseData:
    pred_cam: torch.Tensor
    pred_cam_t: torch.Tensor
    focal_length: torch.Tensor
    mhr_parameters: torch.Tensor
    root_translation: torch.Tensor
    root_rotation: torch.Tensor
    joint_angles: torch.Tensor
    pred_vertices: torch.Tensor
    pred_keypoints_2d: torch.Tensor
    pred_keypoints_3d: torch.Tensor


class PoseDataInterpreter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoseDataInterpreter, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Loads the MHR model on the first instantiation."""
        parent = Path(__file__).resolve().parent.parent
        mhr_model_path = (
            parent / "checkpoints" / "sam3d" / "dinov3" / "assets" / "mhr_model.pt"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading scripted MHR model from {mhr_model_path} onto {self.device}...")

        self.mhr_model = torch.jit.load(str(mhr_model_path), map_location=self.device)
        self.mhr_model.eval()

    @torch.no_grad()
    def interpret_pose_dictionary(self, pose_dict: dict[str, Any]) -> PoseData:
        """
        Takes a perfectly formatted pose dictionary (all values are lists),
        runs the MHR model, and returns a PoseData dataclass.
        """
        # 1. Straight to tensors
        mhr_params = torch.tensor(pose_dict["mhr_parameters"], dtype=torch.float32).to(
            self.device
        )

        root_translation = mhr_params[0:3]
        root_rotation = mhr_params[3:6]
        joint_angles = mhr_params[6:136]

        # 2. MHR Forward Pass
        id_coeffs = torch.zeros(1, 45, device=self.device)
        pose_params = mhr_params.unsqueeze(0)
        expr_coeffs = torch.zeros(1, 72, device=self.device)

        vertices, skeleton_state = self.mhr_model(id_coeffs, pose_params, expr_coeffs)

        vertices = vertices.squeeze(0).cpu()
        skeleton_state = skeleton_state.squeeze(0).cpu()
        pred_keypoints_3d = skeleton_state[..., :3]

        # 3. Build and return
        return PoseData(
            pred_cam=torch.tensor(pose_dict["pred_cam"], dtype=torch.float32),
            pred_cam_t=torch.tensor(
                pose_dict["pred_cam_translation"], dtype=torch.float32
            ),
            focal_length=torch.tensor(pose_dict["focal_length"], dtype=torch.float32),
            mhr_parameters=mhr_params.cpu(),
            root_translation=root_translation.cpu(),
            root_rotation=root_rotation.cpu(),
            joint_angles=joint_angles.cpu(),
            pred_vertices=vertices,
            pred_keypoints_2d=torch.tensor(
                pose_dict["keypoints_2d"], dtype=torch.float32
            ),
            pred_keypoints_3d=pred_keypoints_3d,
        )
