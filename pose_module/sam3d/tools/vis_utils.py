# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _root_rotation_matrix_in_vertex_frame(root_rotation):
    root_rotation = _to_numpy(root_rotation).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(root_rotation)
    vertex_frame_transform = np.diag([1.0, -1.0, -1.0])
    return vertex_frame_transform @ rotation_matrix @ vertex_frame_transform


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img


def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(
            person_output["pred_vertices"] + person_output["pred_cam_t"]
        )
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (
        np.max(all_pred_vertices[-2 * 18439 :], axis=0)
        + np.min(all_pred_vertices[-2 * 18439 :], axis=0)
    ) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        * 255
    )

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img


def visualize_on_white(img_cv2, outputs, faces):
    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(
            person_output["pred_vertices"] + person_output["pred_cam_t"]
        )
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)

    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    fake_pred_cam_t = (
        np.max(all_pred_vertices, axis=0) + np.min(all_pred_vertices, axis=0)
    ) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    white_img = np.ones_like(img_cv2) * 255
    renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)

    img_white = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=False,
        )
        * 255
    )

    return img_white.astype(np.uint8)


def visualize_pose_with_camera_reference(
    camera_reference_img_cv2,
    camera_reference_output,
    pose_reference_output,
    faces,
    white_background=False,
    center_on_camera=False,
):
    base_img = (
        np.ones_like(camera_reference_img_cv2) * 255
        if white_background
        else camera_reference_img_cv2.copy()
    )

    focal_length = _to_numpy(pose_reference_output["focal_length"])
    pose_translation = _to_numpy(pose_reference_output["pred_cam_t"]).copy()
    pred_vertices = _to_numpy(pose_reference_output["pred_vertices"]).copy()

    reference_rotation_matrix = _root_rotation_matrix_in_vertex_frame(
        camera_reference_output["root_rotation"]
    )
    pose_rotation_matrix = _root_rotation_matrix_in_vertex_frame(
        pose_reference_output["root_rotation"]
    )
    rotation_delta = reference_rotation_matrix @ pose_rotation_matrix.T

    upright_correction, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi], dtype=np.float32))
    rotation_delta = upright_correction @ rotation_delta

    mesh_center = np.mean(pred_vertices, axis=0)
    pred_vertices = (pred_vertices - mesh_center) @ rotation_delta.T + mesh_center

    pred_vertices[:, 0] *= -1.0
    render_faces = faces[:, [0, 2, 1]]

    render_translation = pose_translation
    if center_on_camera:
        bounds_center = (
            np.max(pred_vertices, axis=0) + np.min(pred_vertices, axis=0)
        ) / 2
        pred_vertices = pred_vertices - bounds_center

        focal_value = float(np.mean(np.reshape(focal_length, -1)))
        img_h, img_w = base_img.shape[:2]
        frame_margin = 0.85
        half_w = max((img_w * 0.5) * frame_margin, 1.0)
        half_h = max((img_h * 0.5) * frame_margin, 1.0)

        req_z_x = (
            focal_value * np.abs(pred_vertices[:, 0]) / half_w - pred_vertices[:, 2]
        )
        req_z_y = (
            focal_value * np.abs(pred_vertices[:, 1]) / half_h - pred_vertices[:, 2]
        )
        target_z = max(float(np.max(req_z_x)), float(np.max(req_z_y)), 0.1) + 0.05

        render_translation = np.array([0.0, 0.0, target_z], dtype=np.float32)

    renderer = Renderer(focal_length=focal_length, faces=render_faces)
    rendered_img = (
        renderer(
            pred_vertices,
            render_translation,
            base_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    return rendered_img.astype(np.uint8)
