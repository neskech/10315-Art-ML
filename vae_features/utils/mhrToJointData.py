import torch
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np


@dataclass
class JointData:
    # (B, 889)
    raw: torch.Tensor
    # (B, 127, 3) euler angles
    joint_angles: torch.Tensor
    # (B, 3,)
    root_rotation: torch.Tensor
    # (B, 3,)
    root_translation: torch.Tensor


NUM_JOINTS = 127
# The first joint index just contains a bunch of 0's
ROOT_JOINT_IDX = 1


class PostProcessor:
    def __init__(self, device: torch.device):
        self.device = device

        script_dir = str(Path(__file__).resolve().parent)
        transform_path = os.path.join(script_dir, "parameterTransform.npy")
        parameter_transform = np.load(transform_path)
        # Shape (889, 321)
        self.parameter_transform = torch.from_numpy(parameter_transform).to(self.device)
        self.inv_parameter_transform = torch.linalg.pinv(self.parameter_transform.T)

    def _mhr_to_raw_joint_data(self, mhr_parameters: torch.Tensor):
        """
        Post processes mhr parameters of shape (B, 204) into
        joint parameters of shape (B, 889).

        The resulting array has 7 parameters per joint, for a total
        of 127 different joints

        All code here is derived from mhr.py in the Meta mhr repo
        """
        batch_size = mhr_parameters.shape[0]
        num_padding = 117  # The number of blend shapes in MHR

        # Shape (B, 321)
        padded_parameters = torch.cat(
            (mhr_parameters, torch.zeros(batch_size, num_padding).to(self.device)),
            dim=-1,
        )

        # Shape (B, 889)
        return padded_parameters @ self.parameter_transform.transpose(0, 1)

    def mhr_to_joint_data(
        self,
        mhr_parameters: torch.Tensor,
        zero_out_root_rotation: bool,
    ) -> JointData:
        raw = self._mhr_to_raw_joint_data(mhr_parameters)
        assert raw.shape[1] == NUM_JOINTS * 7

        batch_size = raw.shape[0]
        rotations = raw[:, :3].unfold(1, 3, 7).reshape(batch_size, -1)

        root_translation = raw[:, ROOT_JOINT_IDX * 7 : ROOT_JOINT_IDX * 7 + 3]
        root_rotation = raw[:, ROOT_JOINT_IDX * 7 + 3 : ROOT_JOINT_IDX * 7 + 6]

        if zero_out_root_rotation:
            rotations[:, ROOT_JOINT_IDX : ROOT_JOINT_IDX + 3] = 0

        return JointData(
            raw=raw,
            joint_angles=rotations,
            root_translation=root_translation,
            root_rotation=root_rotation,
        )

    def raw_to_mhr_parameters(self, raw: torch.Tensor):
        """
        Inverts raw joint parameters (B, 889) back to MHR parameters (B, 204).
        """
        padded_reconstructed = raw @ self.inv_parameter_transform
        mhr_parameters = padded_reconstructed[:, :204]
        return mhr_parameters

    def patch_raw_with_joint_angles(
        self,
        base_raw: torch.Tensor,
        joint_angles_mhr_order: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a new raw tensor where only Euler angle slots are replaced.

        Args:
            base_raw: Tensor with shape (B, 889).
            joint_angles_mhr_order: Euler angles in MHR joint ordering with shape
                (B, 127, 3).
        """
        if base_raw.ndim != 2 or base_raw.shape[1] != NUM_JOINTS * 7:
            raise ValueError(f"Expected base_raw shape (B, {NUM_JOINTS * 7}), got {base_raw.shape}")
        if joint_angles_mhr_order.ndim != 3 or joint_angles_mhr_order.shape[1:] != (NUM_JOINTS, 3):
            raise ValueError(
                f"Expected joint_angles_mhr_order shape (B, {NUM_JOINTS}, 3), got "
                f"{joint_angles_mhr_order.shape}"
            )
        if base_raw.shape[0] != joint_angles_mhr_order.shape[0]:
            raise ValueError(
                f"Batch mismatch between raw ({base_raw.shape[0]}) and joint angles "
                f"({joint_angles_mhr_order.shape[0]})"
            )

        base = base_raw.view(base_raw.shape[0], NUM_JOINTS, 7)
        patched = torch.cat(
            [base[:, :, 0:3], joint_angles_mhr_order, base[:, :, 6:7]],
            dim=-1,
        )
        return patched.reshape(base_raw.shape[0], NUM_JOINTS * 7)

    def joint_angles_to_mhr_parameters(
        self,
        joint_angles_mhr_order: torch.Tensor,
        base_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts reconstructed joint angles to MHR parameters using base raw tensor.

        Returns:
            tuple[patched_raw, mhr_parameters]
        """
        patched_raw = self.patch_raw_with_joint_angles(
            base_raw=base_raw,
            joint_angles_mhr_order=joint_angles_mhr_order,
        )
        mhr_parameters = self.raw_to_mhr_parameters(patched_raw)
        return patched_raw, mhr_parameters