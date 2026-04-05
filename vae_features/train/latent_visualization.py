from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA

_FIXED_LATENT_IMAGE_KEYS: set[str] | None = None


def _fit_reduce(latents: np.ndarray, method: str = "pca") -> np.ndarray:
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D array for latents, got shape {latents.shape}")
    if latents.shape[0] < 2:
        raise ValueError("Need at least 2 samples for 2D projection")

    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        return reducer.fit_transform(latents)

    raise ValueError(f"Unsupported projection method: {method}")


def save_latent_projection_with_images(
    latents: np.ndarray,
    image_paths: list[str],
    output_path: str | Path,
    title: str,
    method: str = "pca",
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


def _resolve_image_path(meta: dict[str, Any], data_dir: str | Path) -> str | None:
    image_path_abs = meta.get("image_path_abs")
    if image_path_abs is not None:
        abs_path = Path(image_path_abs)
        if abs_path.exists():
            return str(abs_path)

    image_path = meta.get("image_path")
    if image_path is not None:
        rel_path = Path(image_path)
        if rel_path.exists():
            return str(rel_path)

        fallback = Path(data_dir) / "poses" / str(image_path).lstrip("/")
        if fallback.exists():
            return str(fallback)

    return None


def save_fixed_latent_projection_with_images(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    num_samples: int,
    seed: int,
    data_dir: str | Path,
    method: str = "pca",
    thumbnail_zoom: float = 0.14,
) -> str | None:
    global _FIXED_LATENT_IMAGE_KEYS

    if len(embeddings) < 2:
        return None

    valid_samples: list[tuple[np.ndarray, str]] = []
    for emb, meta in zip(embeddings, metadata, strict=False):
        image_path = _resolve_image_path(meta, data_dir=data_dir)
        if image_path is None:
            continue
        valid_samples.append((emb, image_path))

    if len(valid_samples) < 2:
        return None

    if _FIXED_LATENT_IMAGE_KEYS is None:
        rng = np.random.default_rng(seed)
        sample_size = min(int(num_samples), len(valid_samples))
        candidate_keys = [path for _, path in valid_samples]
        chosen_idx = rng.choice(len(candidate_keys), size=sample_size, replace=False)
        _FIXED_LATENT_IMAGE_KEYS = {candidate_keys[int(i)] for i in chosen_idx}

    selected_latents: list[np.ndarray] = []
    selected_image_paths: list[str] = []
    for emb, image_path in valid_samples:
        if image_path in _FIXED_LATENT_IMAGE_KEYS:
            selected_latents.append(emb)
            selected_image_paths.append(image_path)

    if len(selected_latents) < 2:
        return None

    saved_path = save_latent_projection_with_images(
        latents=np.array(selected_latents),
        image_paths=selected_image_paths,
        output_path=output_path,
        title=title,
        method=method,
        thumbnail_zoom=thumbnail_zoom,
    )
    return saved_path
