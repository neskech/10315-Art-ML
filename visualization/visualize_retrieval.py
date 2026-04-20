import argparse
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from topKRetrieval.topKRetrieval import runTopKRetrieval
from topKRetrieval.squaredDistanceRetrieval import squaredDistanceMetric
from topKRetrieval.classificationFeatureRetrieval import getClassificationFeatureMetric
from topKRetrieval.vaeFeatureRetrieval import getVAEFeatureMetric

CURRENT_DIRECTORY = Path(__file__).parent.resolve()
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"


def _data_relative_pose_image_path(relative_image_path: str) -> str:
    """Repo-relative path under data/ (e.g. data/poses/...)."""
    rel = Path(relative_image_path).as_posix()
    return f"data/poses/{rel}"


def _load_rgb_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_delta(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    return float(diff.mean() / 255.0)


def _pose_squared_distance(pose_a, pose_b) -> float:
    """Sum of squared differences of 3D keypoints (same notion as squaredDistanceMetric).

    Runs on CPU so mixed GPU/CPU tensors from the dataset vs query do not break.
    """
    a = pose_a.pred_keypoints_3d.detach().cpu().reshape(-1).float()
    b = pose_b.pred_keypoints_3d.detach().cpu().reshape(-1).float()
    return float(((a - b) ** 2).sum().item())


def _dedupe_results_by_image(
    results,
    epsilon: float,
    target_k: int,
    query_image: np.ndarray | None = None,
    *,
    sq_distance_threshold: float | None = None,
):
    """Filter retrieval hits: path, then squared 3D-keypoint distance, then pixels.

    Order of checks for each candidate (first failure drops the candidate):
      1. Same ``relative_image_path`` as an already-kept result.
      2. Squared keypoint distance below ``sq_distance_threshold`` vs any kept pose
         (disabled when threshold is ``None`` or ``<= 0``).
      3. Near-duplicate of the query image (mean normalized pixel delta).
      4. Near-duplicate of a previously kept result image (same pixel test).
    """
    unique_results = []
    unique_images = []
    seen_paths: set[str] = set()
    use_sq = sq_distance_threshold is not None and sq_distance_threshold > 0.0

    for res_pose in results:
        path_key = Path(res_pose.relative_image_path).as_posix()
        if path_key in seen_paths:
            continue

        res_full_path = str(POSES_DIRECTORY / res_pose.relative_image_path)
        res_img = _load_rgb_image(res_full_path)
        if res_img is None:
            continue

        if use_sq and any(
            _pose_squared_distance(res_pose, kept) < sq_distance_threshold
            for kept in unique_results
        ):
            continue

        if query_image is not None and _image_delta(res_img, query_image) < epsilon:
            continue

        is_duplicate = any(
            _image_delta(res_img, existing) < epsilon for existing in unique_images
        )
        if is_duplicate:
            continue

        unique_results.append(res_pose)
        seen_paths.add(path_key)
        unique_images.append(res_img)

        if len(unique_results) >= target_k:
            break

    return unique_results


def render_results_table(
    query_path, results, query_pose, metric_func, metric_name, output_filename
):
    k = len(results)
    cols = 3
    rows = (k // cols) + (1 if k % cols != 0 else 0)

    fig = plt.figure(figsize=(15, 5 * (rows + 1)))
    gs = fig.add_gridspec(rows + 1, cols)

    fig.suptitle(
        f"Top-{k} Pose Retrieval\nMetric: {metric_name}", fontsize=24, fontweight="bold"
    )

    query_img = _load_rgb_image(query_path)
    if query_img is None:
        raise FileNotFoundError(f"Could not load query image at {query_path}")
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_img)
    ax_query.set_title("QUERY", fontsize=20, fontweight="bold", pad=15)
    ax_query.axis("off")

    for i, res_pose in enumerate(results):
        r = (i // cols) + 1
        c = i % cols

        res_full_path = str(POSES_DIRECTORY / res_pose.relative_image_path)
        res_img = _load_rgb_image(res_full_path)

        if res_img is not None:
            dist = metric_func(query_pose, res_pose)

            ax = fig.add_subplot(gs[r, c])
            ax.imshow(res_img)
            ax.set_title(f"Rank {i + 1}\nDist: {dist:.4f}", fontsize=12)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    visuals_dir = CURRENT_DIRECTORY / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    output_path = visuals_dir / output_filename

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Result table saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-path", type=str, required=True)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument(
        "--metric",
        type=str,
        default="squared",
        choices=["squared", "classification", "vae"],
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="mhr_vae_best.pt",
        help="Checkpoint filename under checkpoints/ used for VAE retrieval features.",
    )
    parser.add_argument(
        "--dedupe-epsilon",
        type=float,
        default=0.01,
        help="Two results are considered duplicate images if mean normalized pixel delta is below this threshold.",
    )
    parser.add_argument(
        "--dedupe-sq-distance-threshold",
        type=float,
        default=1e-4,
        help=(
            "Also drop a candidate if sum of squared 3D keypoint differences to "
            "any already-kept pose is below this (same family as squaredDistanceMetric). "
            "Use 0 to disable pose-distance deduplication."
        ),
    )
    parser.add_argument(
        "--overfetch-factor",
        type=int,
        default=5,
        help="Retrieve this many times K candidates before deduplication.",
    )
    parser.add_argument("-o", "--output", type=str, default="retrieval_results.png")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("k must be greater than 0")
    if args.overfetch_factor < 1:
        raise ValueError("overfetch-factor must be at least 1")
    if args.dedupe_epsilon < 0:
        raise ValueError("dedupe-epsilon must be non-negative")
    if args.dedupe_sq_distance_threshold < 0:
        raise ValueError("dedupe-sq-distance-threshold must be non-negative")

    if args.metric == "squared":
        selected_metric = squaredDistanceMetric
    elif args.metric == "classification":
        selected_metric = getClassificationFeatureMetric()
    else:
        selected_metric = getVAEFeatureMetric(checkpoint_name=args.vae_checkpoint)
    overfetch_k = max(args.k, args.k * args.overfetch_factor)

    query_pose, overfetched_results = runTopKRetrieval(
        pose_image_path=args.image_path, distanceFunction=selected_metric, k=overfetch_k
    )

    query_image = _load_rgb_image(args.image_path)
    if query_image is None:
        raise FileNotFoundError(f"Could not load query image at {args.image_path}")

    sq_thr = (
        None
        if args.dedupe_sq_distance_threshold == 0
        else args.dedupe_sq_distance_threshold
    )
    results = _dedupe_results_by_image(
        overfetched_results,
        epsilon=args.dedupe_epsilon,
        target_k=args.k,
        query_image=query_image,
        sq_distance_threshold=sq_thr,
    )

    if len(results) < args.k:
        print(
            f"Warning: Only {len(results)} unique results found after deduplication "
            f"(requested {args.k})."
        )

    print(f"Top-{len(results)} retrieved images (paths relative to repo, under data/):")
    for rank, res_pose in enumerate(results, start=1):
        print(f'"{_data_relative_pose_image_path(res_pose.relative_image_path)}"')

    render_results_table(
        args.image_path, results, query_pose, selected_metric, args.metric, args.output
    )


if __name__ == "__main__":
    main()
