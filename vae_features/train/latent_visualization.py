from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
import umap

from vae_features.train.image_path_list import load_resolved_image_paths_from_json


def _fit_reduce(latents: np.ndarray, method: str = "umap") -> np.ndarray:
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D array for latents, got shape {latents.shape}")
    if latents.shape[0] < 2:
        raise ValueError("Need at least 2 samples for 2D projection")

    m = method.lower()
    if m == "umap":
        n = latents.shape[0]
        if n < 3:
            raise ValueError(
                "UMAP (cosine) latent projection requires at least 3 samples; "
                f"got {n}. Add more images to latent_projection_images.json or use method='pca'."
            )
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        )
        return reducer.fit_transform(latents)

    if m == "pca":
        reducer = PCA(n_components=2)
        return reducer.fit_transform(latents)

    raise ValueError(f"Unsupported projection method: {method}")


def save_latent_projection_with_images(
    latents: np.ndarray,
    image_paths: list[str],
    output_path: str | Path,
    title: str,
    method: str = "umap",
    thumbnail_zoom: float = 0.14,
):
    if len(latents) != len(image_paths):
        raise ValueError(
            f"latents/image_paths size mismatch: {len(latents)} != {len(image_paths)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize latent vectors for angular geometry before linear projection.
    norms = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    normalized_latents = latents / norms
    coords = _fit_reduce(normalized_latents, method=method)

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_title(title)
    if method.lower() == "umap":
        ax.set_xlabel("UMAP 1 (cosine)")
        ax.set_ylabel("UMAP 2 (cosine)")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    x_pad = (coords[:, 0].max() - coords[:, 0].min()) * 0.1 + 1e-4
    y_pad = (coords[:, 1].max() - coords[:, 1].min()) * 0.1 + 1e-4
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

    for (x, y), image_path in zip(coords, image_paths, strict=False):
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((96, 96))
            imagebox = OffsetImage(np.array(img), zoom=thumbnail_zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        except Exception:
            # Keep plotting resilient if an image is missing/corrupt.
            ax.scatter([x], [y], s=20, alpha=0.7, color="gray")

    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=320)
    plt.close(fig)

    return str(output_path)


class LatentProjectionImageList:
    """JSON-driven list of pose image paths used for latent projection thumbnails."""

    DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "latent_projection_images.json"

    def __init__(self, json_path: str | Path | None = None) -> None:
        self.json_path = Path(json_path) if json_path is not None else self.DEFAULT_JSON_PATH

    def get_image_paths(self, data_dir: str | Path) -> list[str]:
        """Resolved absolute paths (in JSON order) that exist on disk."""
        return load_resolved_image_paths_from_json(self.json_path, Path(data_dir))


class LatentProjectionVisualizer:
    """Save a 2D latent projection plot with image thumbnails for a pre-aligned batch."""

    def save_projection(
        self,
        embeddings: np.ndarray,
        image_paths: list[str],
        output_path: str | Path,
        title: str,
        method: str = "umap",
        thumbnail_zoom: float = 0.14,
    ) -> str | None:
        if len(embeddings) < 2:
            return None
        if len(embeddings) != len(image_paths):
            raise ValueError(
                f"embeddings and image_paths length mismatch: {len(embeddings)} vs {len(image_paths)}"
            )
        return save_latent_projection_with_images(
            latents=embeddings,
            image_paths=image_paths,
            output_path=output_path,
            title=title,
            method=method,
            thumbnail_zoom=thumbnail_zoom,
        )
