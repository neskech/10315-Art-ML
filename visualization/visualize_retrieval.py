import argparse
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from topKRetrieval.topKRetrieval import runTopKRetrieval
from topKRetrieval.squaredDistanceRetrieval import squaredDistanceMetric
from topKRetrieval.classificationFeatureRetrieval import getClassificationFeatureMetric

CURRENT_DIRECTORY = Path(__file__).parent.resolve()
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"


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


def _dedupe_results_by_image(
    results, epsilon: float, target_k: int, query_image: np.ndarray | None = None
):
    unique_results = []
    unique_images = []

    for res_pose in results:
        res_full_path = str(POSES_DIRECTORY / res_pose.relative_image_path)
        res_img = _load_rgb_image(res_full_path)
        if res_img is None:
            continue

        if query_image is not None and _image_delta(res_img, query_image) < epsilon:
            continue

        is_duplicate = any(
            _image_delta(res_img, existing) < epsilon for existing in unique_images
        )
        if is_duplicate:
            continue

        unique_results.append(res_pose)
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
        "--metric", type=str, default="squared", choices=["squared", "classification"]
    )
    parser.add_argument(
        "--dedupe-epsilon",
        type=float,
        default=0.01,
        help="Two results are considered duplicate images if mean normalized pixel delta is below this threshold.",
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

    metric_map = {
        "squared": squaredDistanceMetric,
        "classification": getClassificationFeatureMetric(),
    }
    selected_metric = metric_map[args.metric]
    overfetch_k = max(args.k, args.k * args.overfetch_factor)

    query_pose, overfetched_results = runTopKRetrieval(
        pose_image_path=args.image_path, distanceFunction=selected_metric, k=overfetch_k
    )

    query_image = _load_rgb_image(args.image_path)
    if query_image is None:
        raise FileNotFoundError(f"Could not load query image at {args.image_path}")

    results = _dedupe_results_by_image(
        overfetched_results,
        epsilon=args.dedupe_epsilon,
        target_k=args.k,
        query_image=query_image,
    )

    if len(results) < args.k:
        print(
            f"Warning: Only {len(results)} unique results found after deduplication "
            f"(requested {args.k})."
        )

    render_results_table(
        args.image_path, results, query_pose, selected_metric, args.metric, args.output
    )


if __name__ == "__main__":
    main()
