import argparse
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from topKRetrieval.topKRetrieval import runTopKRetrieval
from topKRetrieval.squaredDistanceRetrieval import squaredDistanceMetric
from topKRetrieval.classificationFeatureRetrieval import classificationFeatureMetric

CURRENT_DIRECTORY = Path(__file__).parent.resolve()
POSES_DIRECTORY = CURRENT_DIRECTORY.parent / "data" / "poses"


def render_results_table(query_path, results, query_pose, metric_func, metric_name, output_filename):
    k = len(results)
    cols = 3
    rows = (k // cols) + (1 if k % cols != 0 else 0)

    fig = plt.figure(figsize=(15, 5 * (rows + 1)))
    gs = fig.add_gridspec(rows + 1, cols)

    fig.suptitle(f"Top-{k} Pose Retrieval\nMetric: {metric_name}", fontsize=24, fontweight="bold")

    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_img)
    ax_query.set_title("QUERY", fontsize=20, fontweight="bold", pad=15)
    ax_query.axis("off")

    for i, res_pose in enumerate(results):
        r = (i // cols) + 1
        c = i % cols

        res_full_path = str(POSES_DIRECTORY / res_pose.relative_image_path)
        res_img = cv2.imread(res_full_path)

        if res_img is not None:
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
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
    parser.add_argument("-o", "--output", type=str, default="retrieval_results.png")
    args = parser.parse_args()

    metric_map = {
        "squared": squaredDistanceMetric,
        "classification": classificationFeatureMetric,
    }
    selected_metric = metric_map[args.metric]
    query_pose, results = runTopKRetrieval(
        pose_image_path=args.image_path, distanceFunction=selected_metric, k=args.k
    )

    render_results_table(args.image_path, results, query_pose, selected_metric, args.metric, args.output)


if __name__ == "__main__":
    main()