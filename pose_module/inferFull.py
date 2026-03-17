import numpy as np
import roma
import torch
from typing import Dict, List

from preprocess import prepare_batch_correctly
from sam3d.sam_3d_body.data.transforms.common import Compose
from sam3d.sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody
from sam3d.sam_3d_body.models.modules.mhr_utils import (
    fix_wrist_euler,
    rotation_angle_difference,
)
from sam3d.sam_3d_body.utils.dist import recursive_to


def _tensor_to_numpy_list(tensor_batch: torch.Tensor) -> List[np.ndarray]:
    """
    Helper: Converts a batch of tensors (B, C, H, W) to a list of numpy arrays (H, W, C).
    Required because OpenCV transforms in prepare_batch_correctly cannot handle Tensors.
    """
    # 1. Detach from graph and move to CPU
    # 2. Permute (B, C, H, W) -> (B, H, W, C)
    imgs_np = tensor_batch.detach().cpu().permute(0, 2, 3, 1).numpy()

    # 3. If the tensor was float (0-1), keep as float.
    #    If it was uint8-like (0-255), keep as is.
    #    We return a list of individual arrays.
    return [imgs_np[i] for i in range(imgs_np.shape[0])]


def infer_full(
    model: SAM3DBody,
    image_batch: torch.Tensor,
    batch: Dict,
    transform_hand: Compose,
    device: torch.device,
    thresh_wrist_angle=1.4,
):
    """
    Performs full-body inference with hand refinement for a 3D body model using a multi-stage process.

    This function first runs the body decoder to estimate the full-body pose,
    then refines the hand poses by running the hand decoder on cropped hand regions.
    The hand outputs are unflipped and merged back into the body pose, with several validity
    checks (wrist angle, box size, keypoint location, and wrist distance) to determine whether
    to use the refined hand predictions. Optionally, keypoint prompting is used to further refine
    the body pose using wrist and elbow keypoints. The function updates the pose, scale, and
    shape parameters, runs forward kinematics, and projects the final 3D keypoints to 2D.

    NOTE: This is an adapted function from the Sam3DBody run_inference member function. The
    original function did not support batched inputs. This adaptation is an AI generated fix
    of the old code, so problems may arise

    Args:
        model (SAM3DBody): The 3D body model with body and hand decoders.
        image_batch (torch.Tensor): Batch of input images of shape (B, C, H, W).
        batch (Dict): Batch dictionary containing camera intrinsics and other metadata.
        transform_hand (Compose): Transform to apply to hand crops.
        device (torch.device): Device to run computations on.
        thresh_wrist_angle (float, optional): Threshold for wrist angle difference to accept hand refinement. Default is 1.4.

    Returns:
        Dict: Updated pose output dictionary with refined body and hand predictions, including 3D/2D keypoints, vertices, and model parameters.
    """
    # FIX 1: Extract dimensions from Batch Tensor (B, C, H, W)
    # Original code assumed (H, W, C) numpy array
    batch_size, _, height, width = image_batch.shape

    cam_int = batch["cam_int"].clone()

    # Step 1. For full-body inference, we first inference with the body decoder.
    pose_output = model.forward_step(batch, decoder_type="body")
    left_xyxy, right_xyxy = model._get_hand_box(pose_output, batch)
    ori_local_wrist_rotmat = roma.euler_to_rotmat(
        "XZY",
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
            1, (2, 3)
        ),
    )

    # Step 2. Re-run with each hand
    ## Left... Flip image & box

    # FIX 2: Batched Image Flipping
    # img_batch is (B, C, H, W). Flip on Width dimension (dim 3).
    # Original: flipped_img = img[:, ::-1]
    flipped_img_batch = torch.flip(image_batch, [3])

    tmp = left_xyxy.copy()
    left_xyxy[:, 0] = width - tmp[:, 2] - 1
    left_xyxy[:, 2] = width - tmp[:, 0] - 1

    batch_lhand = prepare_batch_correctly(
        images=_tensor_to_numpy_list(flipped_img_batch),
        transform=transform_hand,
        boxes=left_xyxy,
        cam_int=cam_int.clone(),
    )
    batch_lhand = recursive_to(batch_lhand, device)
    lhand_output = model.forward_step(batch_lhand, decoder_type="hand")

    # Unflip output
    ## Flip scale
    ### Get MHR values
    scale_r_hands_mean = model.head_pose.scale_mean[8].item()
    scale_l_hands_mean = model.head_pose.scale_mean[9].item()
    scale_r_hands_std = model.head_pose.scale_comps[8, 8].item()
    scale_l_hands_std = model.head_pose.scale_comps[9, 9].item()
    ### Apply
    lhand_output["mhr_hand"]["scale"][:, 9] = (
        (
            scale_r_hands_mean
            + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
        )
        - scale_l_hands_mean
    ) / scale_l_hands_std
    ## Get the right hand global rotation, flip it, put it in as left.
    lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
        "joint_global_rots"
    ][:, 42].clone()
    lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
    ### Flip hand pose
    lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][:, 54:]
    ### Unflip box
    batch_lhand["bbox_center"][:, :, 0] = (
        width - batch_lhand["bbox_center"][:, :, 0] - 1
    )

    ## Right...
    batch_rhand = prepare_batch_correctly(
        images=_tensor_to_numpy_list(image_batch),
        transform=transform_hand,
        boxes=right_xyxy,
        cam_int=cam_int.clone(),
    )
    batch_rhand = recursive_to(batch_rhand, device)
    rhand_output = model.forward_step(batch_rhand, decoder_type="hand")

    # Step 3. replace hand pose estimation from the body decoder.
    ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
    joint_rotations = pose_output["mhr"]["joint_global_rots"]
    ### Get lowarm
    lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()  # left, right
    lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3
    ### Get zero-wrist pose
    wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()  # left, right
    wrist_zero_rot_pose = (
        lowarm_joint_rotations @ model.head_pose.joint_rotation[wrist_twist_joint_idxs]
    )
    ### Get globals from left & right
    left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
    right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
    pred_global_wrist_rotmat = torch.stack(
        [
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ],
        dim=1,
    )
    ### Get the local poses that lead to the wrist being pred_global_wrist_rotmat
    fused_local_wrist_rotmat = torch.einsum(
        "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
    )
    angle_difference = rotation_angle_difference(
        ori_local_wrist_rotmat, fused_local_wrist_rotmat
    )  # B x 2 x 3 x3
    angle_difference_valid_mask = angle_difference < thresh_wrist_angle

    ## CRITERIA 2: hand box size
    hand_box_size_thresh = 64
    hand_box_size_valid_mask = torch.stack(
        [
            (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
            (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
        ],
        dim=1,
    )

    ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
    hand_kps2d_thresh = 0.5
    hand_kps2d_valid_mask = torch.stack(
        [
            lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2))
            < hand_kps2d_thresh,
            rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2))
            < hand_kps2d_thresh,
        ],
        dim=1,
    )

    ## CRITERIA 4: 2D wrist distance.
    hand_wrist_kps2d_thresh = 0.25
    kps_right_wrist_idx = 41
    kps_left_wrist_idx = 62
    # FIX 4: Use batch-aware width for flip logic
    right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand
    body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_left_wrist_idx]
    ].clone()
    right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
        dim=-1
    ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
    left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
        dim=-1
    ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
    hand_wrist_kps2d_valid_mask = torch.stack(
        [
            left_kps_dist < hand_wrist_kps2d_thresh,
            right_kps_dist < hand_wrist_kps2d_thresh,
        ],
        dim=1,
    )
    ## Left-right
    hand_valid_mask = (
        angle_difference_valid_mask
        & hand_box_size_valid_mask
        & hand_kps2d_valid_mask
        & hand_wrist_kps2d_valid_mask
    )

    # Keypoint prompting with the body decoder.
    # We use the wrist location from the hand decoder and the elbow location
    # from the body decoder as prompts to get an updated body pose estimation.
    batch_size, num_person = batch["img"].shape[:2]
    model.hand_batch_idx = []
    model.body_batch_idx = list(range(batch_size * num_person))

    ## Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
    kps_right_wrist_idx = 41
    kps_left_wrist_idx = 62
    right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand

    # Next, get them to crop-normalized space.
    right_kps_crop = model._full_to_crop(batch, right_kps_full)
    left_kps_crop = model._full_to_crop(batch, left_kps_full)

    # Get right & left elbow keypoints from crops; full image. Each are B x 1 x 2
    kps_right_elbow_idx = 8
    kps_left_elbow_idx = 7
    right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_right_elbow_idx]
    ].clone()
    left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_left_elbow_idx]
    ].clone()

    # Next, get them to crop-normalized space.
    right_kps_elbow_crop = model._full_to_crop(batch, right_kps_elbow_full)
    left_kps_elbow_crop = model._full_to_crop(batch, left_kps_elbow_full)

    # Assemble them into keypoint prompts
    keypoint_prompt = torch.cat(
        [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
        dim=1,
    )
    keypoint_prompt = torch.cat([keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1)
    keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
    keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
    keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
    keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

    if keypoint_prompt.shape[0] > 1:
        # Replace invalid keypoints to dummy prompts
        invalid_prompt = (
            (keypoint_prompt[..., 0] < -0.5)
            | (keypoint_prompt[..., 0] > 0.5)
            | (keypoint_prompt[..., 1] < -0.5)
            | (keypoint_prompt[..., 1] > 0.5)
            | (~hand_valid_mask[..., [1, 0, 1, 0]])
        ).unsqueeze(-1)
        dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
        dummy_prompt[:, :, -1] = -2
        keypoint_prompt[:, :, :2] = torch.clamp(
            keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
        )  # [-0.5, 0.5] --> [0, 1]
        keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
    else:
        # Only keep valid keypoints
        valid_keypoint = (
            torch.all(
                (keypoint_prompt[:, :, :2] > -0.5) & (keypoint_prompt[:, :, :2] < 0.5),
                dim=2,
            )
            & hand_valid_mask[..., [1, 0, 1, 0]]
        ).squeeze()
        keypoint_prompt = keypoint_prompt[:, valid_keypoint]
        keypoint_prompt[:, :, :2] = torch.clamp(
            keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
        )  # [-0.5, 0.5] --> [0, 1]

    if keypoint_prompt.numel() != 0:
        pose_output, _ = model.run_keypoint_prompt(batch, pose_output, keypoint_prompt)

    ##############################################################################

    # Drop in hand pose
    left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
    right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
    updated_hand_pose = torch.cat(
        [left_hand_pose_params, right_hand_pose_params], dim=1
    )

    # Drop in hand scales
    updated_scale = pose_output["mhr"]["scale"].clone()
    updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
    updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
    updated_scale[:, 18:] = (
        lhand_output["mhr_hand"]["scale"][:, 18:]
        + rhand_output["mhr_hand"]["scale"][:, 18:]
    ) / 2

    # Update hand shape
    updated_shape = pose_output["mhr"]["shape"].clone()
    updated_shape[:, 40:] = (
        lhand_output["mhr_hand"]["shape"][:, 40:]
        + rhand_output["mhr_hand"]["shape"][:, 40:]
    ) / 2

    ############################ Doing IK ############################

    # First, forward just FK
    joint_rotations = model.head_pose.mhr_forward(
        global_trans=pose_output["mhr"]["global_rot"] * 0,
        global_rot=pose_output["mhr"]["global_rot"],
        body_pose_params=pose_output["mhr"]["body_pose"],
        hand_pose_params=updated_hand_pose,
        scale_params=updated_scale,
        shape_params=updated_shape,
        expr_params=pose_output["mhr"]["face"],
        return_joint_rotations=True,
    )[1]

    # Get lowarm
    lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()  # left, right
    lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3

    # Get zero-wrist pose
    wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()  # left, right
    wrist_zero_rot_pose = (
        lowarm_joint_rotations @ model.head_pose.joint_rotation[wrist_twist_joint_idxs]
    )

    # Get globals from left & right
    left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
    right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
    pred_global_wrist_rotmat = torch.stack(
        [
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ],
        dim=1,
    )

    # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
    fused_local_wrist_rotmat = torch.einsum(
        "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
    )
    wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat))

    # Put it in.
    angle_difference = rotation_angle_difference(
        ori_local_wrist_rotmat, fused_local_wrist_rotmat
    )  # B x 2 x 3 x3
    valid_angle = angle_difference < thresh_wrist_angle
    valid_angle = valid_angle & hand_valid_mask
    valid_angle = valid_angle.unsqueeze(-1)

    body_pose = pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
        1, (2, 3)
    )
    updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
    pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
        updated_body_pose.flatten(1, 2)
    )

    hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
    pose_output["mhr"]["hand"] = torch.where(
        valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
    ).flatten(1, 2)

    hand_scale = torch.stack(
        [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
        dim=1,
    )
    updated_hand_scale = torch.stack([updated_scale[:, 9], updated_scale[:, 8]], dim=1)
    masked_hand_scale = torch.where(
        valid_angle.squeeze(-1), updated_hand_scale, hand_scale
    )
    pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
    pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

    # Replace shared shape and scale
    pose_output["mhr"]["scale"][:, 18:] = torch.where(
        valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
        (
            lhand_output["mhr_hand"]["scale"][:, 18:] * valid_angle.squeeze(-1)[:, [0]]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
            * valid_angle.squeeze(-1)[:, [1]]
        )
        / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
        pose_output["mhr"]["scale"][:, 18:],
    )
    pose_output["mhr"]["shape"][:, 40:] = torch.where(
        valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
        (
            lhand_output["mhr_hand"]["shape"][:, 40:] * valid_angle.squeeze(-1)[:, [0]]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
            * valid_angle.squeeze(-1)[:, [1]]
        )
        / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
        pose_output["mhr"]["shape"][:, 40:],
    )

    ########################################################

    # Re-run forward
    with torch.no_grad():
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
            model.head_pose.mhr_forward(
                global_trans=pose_output["mhr"]["global_rot"] * 0,
                global_rot=pose_output["mhr"]["global_rot"],
                body_pose_params=pose_output["mhr"]["body_pose"],
                hand_pose_params=pose_output["mhr"]["hand"],
                scale_params=pose_output["mhr"]["scale"],
                shape_params=pose_output["mhr"]["shape"],
                expr_params=pose_output["mhr"]["face"],
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
            )
        )
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference
        jcoords[..., [1, 2]] *= -1
        pose_output["mhr"]["pred_keypoints_3d"] = j3d
        pose_output["mhr"]["pred_vertices"] = verts
        pose_output["mhr"]["pred_joint_coords"] = jcoords
        pose_output["mhr"]["pred_pose_raw"][...] = (
            0  # pred_pose_raw is not valid anymore
        )
        pose_output["mhr"]["mhr_model_params"] = mhr_model_params

    ########################################################
    # Project to 2D
    pred_keypoints_3d_proj = (
        pose_output["mhr"]["pred_keypoints_3d"]
        + pose_output["mhr"]["pred_cam_t"][:, None, :]
    )
    pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
        :, None, None
    ]
    pred_keypoints_3d_proj[:, :, [0, 1]] = (
        pred_keypoints_3d_proj[:, :, [0, 1]]
        + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
            None, None, :
        ]
        * pred_keypoints_3d_proj[:, :, [2]]
    )
    pred_keypoints_3d_proj[:, :, :2] = (
        pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
    )
    pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

    return pose_output
