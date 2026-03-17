import numpy as np
import torch
from torch.utils.data import default_collate
from pose_module.sam3d.sam_3d_body.data.transforms.common import Compose
from pose_module.sam3d.tools.build_detector import HumanDetector


def prepare_batch_correctly(
    images: list[np.ndarray],
    transform: Compose,
    human_detector: HumanDetector | None = None,
    boxes=None,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """
    Prepares a batch of images and associated data for model inference, applying transforms and formatting.

    Args:
        images (List[np.ndarray]): List of images of length B.
        transform (Compose): Transform pipeline to apply to each image and associated data.
        human_detector (HumanDetector | None): A detector for detecting bounding boxes around the subjects
            of the image. If none, no detection will be done
        boxes (optional): Precomputed bounding boxes of shape (B, 4), or None.
        masks (optional): Optional masks for each image, or None.
        masks_score (optional): Optional mask scores for each image, or None.
        cam_int (optional): Optional camera intrinsics, or None.

    Returns:
        dict: Batched and transformed data ready for model input.
    """
    batch_size = len(images)

    bounding_boxes = []
    for i, image in enumerate(images):
        height, width = image.shape[:2]

        if human_detector is not None:
            detected_boxes = human_detector.run_human_detection(image)
            if detected_boxes is not None and len(detected_boxes) > 0:
                box = np.array(detected_boxes[0], dtype=np.float32).flatten()
            else:
                box = np.array([0, 0, width, height], dtype=np.float32)
            bounding_boxes.append(box)
            
        elif boxes is not None:
            box = np.array(boxes[i], dtype=np.float32).flatten()
            bounding_boxes.append(box)
            
        else:
            box = np.array([0, 0, width, height], dtype=np.float32)
            bounding_boxes.append(box)

    bounding_boxes = np.stack(bounding_boxes, axis=0)

    # Make a list of dictionaries
    batch = []
    for batch_idx in range(batch_size):
        batch_element = dict()

        # Image and bounding box
        batch_element["img"] = images[batch_idx]
        batch_element["bbox"] = bounding_boxes[batch_idx]
        batch_element["bbox_format"] = "xyxy"

        # Mask: We do not use masking, set default values unless provided
        # Masks must be on same device as images (GPU) for the transform
        # at the end of the loop (TopdownAffineGPU)
        if masks is None:
            height, width = images[batch_idx].shape[:2]
            batch_element["mask"] = np.zeros(
                (height, width, 1),
                dtype=np.uint8,
            )
        else:
            batch_element["mask"] = masks[batch_idx]

        if masks_score is None:
            batch_element["mask_score"] = np.array(
                0.0,
                dtype=np.float32,
            )
        else:
            batch_element["mask_score"] = masks_score[batch_idx]

        batch_element = transform(batch_element)
        for key in ["bbox_center", "bbox_scale", "ori_img_size", "img_size"]:
            if key in batch_element and isinstance(batch_element[key], np.ndarray):
                if batch_element[key].ndim > 1:
                    batch_element[key] = batch_element[key].reshape(-1)
        batch.append(batch_element)

    # Becomes a dictionary mapping keys to batched tensors
    batch = default_collate(batch)

    # Sam 3D expects batches to be (Batch Size, Num People Per Image, ...)
    # We will only accept one person per image for now, so we make that
    # second dimension equal to 1
    keys_to_unsqueeze = [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]
    for key in keys_to_unsqueeze:
        batch[key] = batch[key].unsqueeze(1).float()

    # Padding mask for people. E.g. if the maximum number of people
    # is 3 then the batch must be padded to (B, 3, ...). This mask
    # is of shape (B, max_people). Max people = 1 right now.
    batch["person_valid"] = torch.ones(batch_size, 1)

    # Add person dimension to the masks
    batch["mask"] = batch["mask"].unsqueeze(2)

    # Default camera instrinsics
    if cam_int is None:
        cam_ints = []
        for image in images:
            h, w = image.shape[:2]
            cam_ints.append(
                [
                    [(h**2 + w**2) ** 0.5, 0, w / 2.0],
                    [0, (h**2 + w**2) ** 0.5, h / 2.0],
                    [0, 0, 1],
                ]
            )
        batch["cam_int"] = torch.tensor(cam_ints, dtype=torch.float32)
    else:
        batch["cam_int"] = cam_int

    return batch