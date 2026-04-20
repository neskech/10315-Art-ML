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


# Letterbox + multi-scale RMS; sizes chosen so crop vs full still overlap at coarse scales.
_DEDUPE_SIZES: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256)


def _letterbox_square_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Fit ``img`` inside ``size``×``size`` with aspect ratio preserved; pad with black."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _rms_l2_letterboxed_rgb(img_a: np.ndarray, img_b: np.ndarray, size: int) -> float:
    """RMS L2 in [0,1] after letterboxing both to ``size``×``size``."""
    a = _letterbox_square_rgb(img_a, size).astype(np.float32) / 255.0
    b = _letterbox_square_rgb(img_b, size).astype(np.float32) / 255.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _min_rms_multiscale(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Minimum RMS across scales; helps when one image is a crop / different resolution."""
    return min(_rms_l2_letterboxed_rgb(img_a, img_b, s) for s in _DEDUPE_SIZES)


def _max_ncc_if_crop(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """If one image is a spatial crop of the other, return max TM_CCOEFF_NORMED score.

    Otherwise (same size, unrelated) returns a low value. Capped resolution for speed.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    if ha * wa == 0 or hb * wb == 0:
        return 0.0
    # Smaller image = template
    if ha * wa > hb * wb:
        img_a, img_b = img_b, img_a
        ha, wa, hb, wb = hb, wb, ha, wa
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    max_side = 640
    if max(hb, wb) > max_side:
        scale = max_side / max(hb, wb)
        gray_b = cv2.resize(gray_b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_a = cv2.resize(gray_a, (max(1, int(wa * scale)), max(1, int(ha * scale))), interpolation=cv2.INTER_AREA)
    if gray_a.shape[0] < 3 or gray_a.shape[1] < 3:
        return 0.0
    if gray_a.shape[0] > gray_b.shape[0] or gray_a.shape[1] > gray_b.shape[1]:
        return 0.0
    res = cv2.matchTemplate(gray_b, gray_a, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else 0.0


def _pair_is_duplicate(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    epsilon: float,
    ncc_threshold: float,
) -> bool:
    """Duplicate if multiscale min-RMS is low OR NCC suggests crop/subregion match."""
    if _min_rms_multiscale(img_a, img_b) < epsilon:
        return True
    if ncc_threshold > 0.0 and _max_ncc_if_crop(img_a, img_b) >= ncc_threshold:
        return True
    return False


def _dedupe_results_by_image(
    results,
    epsilon: float,
    target_k: int,
    query_image: np.ndarray | None = None,
    *,
    ncc_threshold: float = 0.88,
):
    """Drop near-duplicates using letterboxed multiscale L2 + optional crop (NCC) match.

    Stretch-only resize fails when one image is a crop of another with different
    aspect ratio or resolution; letterboxing + min RMS over scales aligns that.
    Template matching catches ``small image ≈ patch of large image`` pairs.
    """
    unique_results: list = []
    unique_images: list[np.ndarray] = []
    seen_paths: set[str] = set()

    for res_pose in results:
        path_key = Path(res_pose.relative_image_path).as_posix()
        if path_key in seen_paths:
            continue

        res_full_path = str(POSES_DIRECTORY / res_pose.relative_image_path)
        res_img = _load_rgb_image(res_full_path)
        if res_img is None:
            continue

        if query_image is not None and _pair_is_duplicate(
            query_image, res_img, epsilon=epsilon, ncc_threshold=ncc_threshold
        ):
            continue

        if any(
            _pair_is_duplicate(kept, res_img, epsilon=epsilon, ncc_threshold=ncc_threshold)
            for kept in unique_images
        ):
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
        default=0.05,
        help=(
            "Min multiscale RMS L2 (letterboxed RGB in [0,1]) below this ⇒ duplicate. "
            f"Scales: {list(_DEDUPE_SIZES)}."
        ),
    )
    parser.add_argument(
        "--dedupe-ncc-threshold",
        type=float,
        default=0.88,
        help=(
            "If > 0, also treat a pair as duplicate when OpenCV matchTemplate NCC "
            "(smaller image vs larger) reaches this—catches crop vs full image. "
            "Use 0 to disable."
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
    if args.dedupe_ncc_threshold < 0 or args.dedupe_ncc_threshold > 1:
        raise ValueError("dedupe-ncc-threshold must be in [0, 1]")

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

    results = _dedupe_results_by_image(
        overfetched_results,
        epsilon=args.dedupe_epsilon,
        target_k=args.k,
        query_image=query_image,
        ncc_threshold=args.dedupe_ncc_threshold,
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
