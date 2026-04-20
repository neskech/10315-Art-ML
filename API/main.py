"""Modal app exposing VAE top-K pose retrieval as a web API.

This module builds a Modal `App` whose container:

  1. Bakes in the project source code and the (large, static) SAM3D + MHR
     checkpoints as image layers so cold starts only need to download the
     small, user-supplied artifacts.
  2. Mounts a Modal `Volume` containing the user-supplied artifacts:
        - ``processed_poses.parquet``
        - VAE checkpoint (``vae.pt``)
        - The pose images directory (``poses/``)
     The volume is populated from the local machine via ``API/serve.py``.
  3. On container start (`@modal.enter`), loads the SAM3D pose estimator and
     the trained VAE, then precomputes a (N, latent_dim) tensor of latent
     embeddings for every row of the parquet so retrieval is a single GPU
     matmul per request.
  4. Exposes a single POST endpoint that accepts a multipart image upload
     plus ``offset`` / ``limit`` query parameters and returns the matching
     pose images as base64-encoded JPEGs.

The path to the local parquet, VAE checkpoint and poses directory are passed
as command-line arguments to ``API/serve.py``; this module reads them from
environment variables that the wrapper sets before invoking ``modal``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local repo layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM3D_CHECKPOINT_LOCAL = REPO_ROOT / "checkpoints" / "sam3d" / "dinov3" / "model.ckpt"
MHR_MODEL_LOCAL = REPO_ROOT / "checkpoints" / "sam3d" / "dinov3" / "assets" / "mhr_model.pt"
SAM3D_MODEL_CONFIG_LOCAL = REPO_ROOT / "checkpoints" / "sam3d" / "dinov3" / "model_config.yaml"
PYPROJECT_LOCAL = REPO_ROOT / "pyproject.toml"
UV_LOCK_LOCAL = REPO_ROOT / "uv.lock"

# Source folders we ship into the container. We deliberately exclude `.venv`,
# `data/`, `vae_features/train/checkpoints/`, etc. to keep the layer small.
SOURCE_DIRS = ["pose_module", "vae_features", "topKRetrieval"]

# ---------------------------------------------------------------------------
# Remote container layout
# ---------------------------------------------------------------------------

REMOTE_REPO = "/root/repo"
REMOTE_VOLUME_MOUNT = "/vol"
REMOTE_PARQUET = f"{REMOTE_VOLUME_MOUNT}/processed_poses.parquet"
REMOTE_VAE_CKPT = f"{REMOTE_VOLUME_MOUNT}/vae.pt"
REMOTE_POSES_DIR = f"{REMOTE_VOLUME_MOUNT}/poses"

VOLUME_NAME = os.environ.get("VAE_API_VOLUME_NAME", "vae-retrieval-data")
APP_NAME = os.environ.get("VAE_API_APP_NAME", "vae-topk-retrieval")
GPU_TYPE = os.environ.get("VAE_API_GPU", "T4")
QUERY_CACHE_MAX_ENTRIES = int(os.environ.get("VAE_API_QUERY_CACHE_MAX", "1024"))

# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------

# Run-time deps for SAM3D pose estimation, the VAE and FastAPI. We mirror the
# project's `pyproject.toml` and additionally pin detectron2 (required by
# `pose_module.sam3d.tools.build_detector.HumanDetector` even when bbox
# detection is disabled at inference time).
_PIP_PACKAGES = [
    # Numerics / data
    "numpy>=1.26,<2.0",
    "pandas>=2.2",
    "pyarrow>=16",
    "opencv-python-headless>=4.10",
    "pillow>=10",
    "scipy",
    "scikit-image",
    # PyTorch ecosystem (CUDA wheels resolved by Modal's GPU base image)
    "torch==2.4.0",
    "torchvision==0.19.0",
    "pytorch-lightning>=2.2",
    "torchmetrics>=1.4",
    "timm>=1.0.7",
    "einops>=0.8",
    "fvcore>=0.1.5",
    "huggingface-hub>=0.24",
    # SAM3D-related (sam_3d_body/data/utils/io.py)
    "braceexpand>=0.1.7",
    "yacs>=0.1.8",
    "loguru>=0.7",
    "dill>=0.3",
    "cloudpickle>=3.0",  # detectron2 lazy config; omitted when d2 is pip-installed with --no-deps
    "appdirs>=1.4",
    "networkx==3.2.1",
    "roma>=1.5",
    "joblib>=1.3",
    "xtcocotools>=1.14",
    "pycocotools>=2.0.8",
    "optree>=0.11",
    "hydra-core>=1.3",
    "pyrootutils>=1.0",
    "mmengine>=0.10",
    "mmcv-lite>=2.1.0,<2.2.0",
    # Web layer
    "fastapi[standard]>=0.115",
    "python-multipart>=0.0.9",
    # Misc
    "tqdm>=4.66",
    "rich>=13.7",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
    )
    .pip_install(*_PIP_PACKAGES)
    # Hyperspherical VAE prior used by `vae_features/model/feedForwardVae.py`
    .pip_install(
        "git+https://github.com/nicola-decao/power_spherical.git",
    )
    # detectron2: HumanDetector eagerly imports it in __init__; install
    # without build isolation since it expects torch already on PYTHONPATH.
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        extra_options="--no-build-isolation --no-deps",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_REPO,
            # Force CPU init for any submodule that touches CUDA at import time.
            "TORCH_HOME": "/root/.cache/torch",
        }
    )
    # Bake source code into a layer
    .add_local_dir(
        str(REPO_ROOT / "pose_module"),
        f"{REMOTE_REPO}/pose_module",
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "vae_features"),
        f"{REMOTE_REPO}/vae_features",
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "topKRetrieval"),
        f"{REMOTE_REPO}/topKRetrieval",
        copy=True,
    )
    # Bake the static SAM3D + MHR checkpoints into the image as well so
    # cold starts don't have to fetch them from the volume.
    .add_local_file(
        str(SAM3D_CHECKPOINT_LOCAL),
        f"{REMOTE_REPO}/checkpoints/sam3d/dinov3/model.ckpt",
        copy=True,
    )
    .add_local_file(
        str(SAM3D_MODEL_CONFIG_LOCAL),
        f"{REMOTE_REPO}/checkpoints/sam3d/dinov3/model_config.yaml",
        copy=True,
    )
    .add_local_file(
        str(MHR_MODEL_LOCAL),
        f"{REMOTE_REPO}/checkpoints/sam3d/dinov3/assets/mhr_model.pt",
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# `fastapi` is only installed inside the Modal image, not locally. Using
# `image.imports()` makes these names importable at module scope when the
# code actually runs inside the container (where FastAPI resolves the
# `UploadFile` annotation via `typing.get_type_hints`), while skipping the
# import during the local app-definition pass Modal runs before deploying.
with image.imports():
    from fastapi import UploadFile


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    timeout=600,
    scaledown_window=3600,
    min_containers=0,
)
class VAERetrieval:
    """Loads VAE + SAM3D once per container and serves nearest-neighbour queries."""

    # ---- lifecycle -------------------------------------------------------

    @modal.enter()
    def _load(self) -> None:
        # Make local repo importable.
        if REMOTE_REPO not in sys.path:
            sys.path.insert(0, REMOTE_REPO)

        import torch

        # Imports deferred to runtime so the local image-build process does
        # not need to import torch / sam3d / etc.
        from pose_module.inference import SAM3DBodyInference  # type: ignore
        from vae_features.model.feedForwardVae import FeedForwardVAE  # type: ignore
        from vae_features.train.mhr_pose_dataset import (  # type: ignore
            MHRPoseDataset,
        )
        from vae_features.utils.feedForward import Norm  # type: ignore
        from vae_features.utils.skeletonFormat import SkeletonFormat  # type: ignore

        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger("vae_retrieval")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log.info("Using device: %s", self._device)

        # Skeleton + dataset (drives MHR -> joint-angle conversion).
        train_dir = Path(REMOTE_REPO) / "vae_features" / "train"
        skeleton_format = SkeletonFormat.from_json_file(
            train_dir / "mhr_skeleton_format.json"
        )

        if not Path(REMOTE_PARQUET).exists():
            raise FileNotFoundError(
                f"Parquet not found at {REMOTE_PARQUET}. Did you run "
                f"`python API/serve.py upload --parquet-path ...`?"
            )
        if not Path(REMOTE_VAE_CKPT).exists():
            raise FileNotFoundError(
                f"VAE checkpoint not found at {REMOTE_VAE_CKPT}. Did you run "
                f"`python API/serve.py upload --vae-checkpoint ...`?"
            )

        # Load checkpoint config early so the dataset and VAE agree on rotation format.
        self._log.info("Loading VAE checkpoint from %s", REMOTE_VAE_CKPT)
        ckpt = torch.load(REMOTE_VAE_CKPT, map_location=self._device, weights_only=False)
        config = ckpt.get("config", {})
        use_6d_rotations = bool(config.get("USE_6D_ROTATIONS", False))
        per_joint = 6 if use_6d_rotations else 3

        self._dataset = MHRPoseDataset(
            parquet_path=REMOTE_PARQUET,
            skeleton_format=skeleton_format,
            joint_names_path=train_dir / "joint_names.json",
            data_root=REMOTE_VOLUME_MOUNT,  # so {data_root}/poses/<rel_path> resolves
            max_samples=None,
            device=self._device,
            use_6d_rotations=use_6d_rotations,
        )
        self._post_processor = self._dataset.post_processor
        self._reorder_indices = self._dataset.reorder_indices.to(self._device)
        self._use_6d_rotations = use_6d_rotations

        # Pose estimator (heavy: DiNOv3 + MHR head).
        self._log.info("Loading SAM3D body inference model...")
        self._estimator = SAM3DBodyInference(
            device=self._device, use_torch_compile=False
        )

        encoder_sizes = config.get(
            "FF_ENCODER_SIZES",
            [skeleton_format.get_joint_count() * per_joint, 512, 256, 128],
        )
        normalization: Norm = (
            "layerNorm"
            if str(config.get("FF_NORMALIZATION", "layer")).lower() == "layer"
            else "batchNorm"
        )
        self._vae = FeedForwardVAE(
            skeletonFormat=skeleton_format,
            encoderSizes=encoder_sizes,
            dropout=float(config.get("FF_DROPOUT", 0.1)),
            use_residuals=bool(config.get("FF_USE_RESIDUALS", False)),
            activation=torch.nn.GELU(),
            normalization=normalization,
            device=self._device,
            use_6d_rotation_format=use_6d_rotations,
            initial_concentration=float(config.get("INITIAL_CONCENTRATION", 1.0)),
        ).to(self._device)
        self._vae.load_state_dict(ckpt["model_state_dict"])
        self._vae.eval()

        # Precompute parquet latents so each request is one matmul.
        self._log.info("Precomputing VAE latents for %d parquet rows...", len(self._dataset))
        self._dataset_latents, self._dataset_paths = self._compute_dataset_latents()
        self._log.info(
            "Cached %d latents of dim %d",
            self._dataset_latents.shape[0],
            self._dataset_latents.shape[1],
        )

        # In-process LRU cache for query-image latents keyed by SHA-256 of bytes.
        # SAM3D + VAE encode is the dominant per-request cost; caching pagination
        # of the same image avoids re-running the full pipeline.
        self._query_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._query_cache_max = QUERY_CACHE_MAX_ENTRIES
        self._query_cache_hits = 0
        self._query_cache_misses = 0

    # ---- helpers ---------------------------------------------------------

    def _compute_dataset_latents(self):
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            self._dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=self._dataset.collate_fn,
        )

        latents: list = []
        paths: list[str] = []
        with torch.no_grad():
            for batch in loader:
                joint_angles = batch.joint_angles.to(self._device)
                if torch.isnan(joint_angles).any():
                    # Skip degenerate rows but keep alignment by still adding paths.
                    flat = joint_angles.reshape(joint_angles.shape[0], -1)
                    latent = torch.zeros(
                        (flat.shape[0], self._vae.latent_size), device=self._device
                    )
                else:
                    flat = joint_angles.reshape(joint_angles.shape[0], -1)
                    _, latent, _ = self._vae.encode(flat)
                latents.append(latent.detach())
                paths.extend(meta.image_path for meta in batch.metadata)

        latent_tensor = torch.cat(latents, dim=0)
        latent_tensor = latent_tensor / (
            latent_tensor.norm(dim=-1, keepdim=True) + 1e-8
        )
        return latent_tensor, paths

    def _embed_query_image(self, image_bytes: bytes):
        """Run SAM3D + VAE on a single uploaded image, return a unit latent (1, D)."""
        import cv2
        import numpy as np
        import torch

        from vae_features.train.mhr_pose_dataset import (  # type: ignore
            _joint_angles_from_raw,
        )
        from vae_features.utils.angleFormat import euler_to_6d  # type: ignore

        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes (unsupported format).")

        # Match the existing topKRetrieval pipeline: use the full image as the bbox.
        outputs = self._estimator.predict([bgr], use_bbox_detector=False)
        if not outputs:
            raise ValueError("Pose estimator returned no outputs for the input image.")

        mhr_parameters = outputs[0]["mhr_parameters"]
        mhr_tensor = torch.tensor(
            [mhr_parameters], dtype=torch.float32, device=self._device
        )

        joint_data = self._post_processor.mhr_to_joint_data(
            mhr_tensor, zero_out_root_rotation=False
        )
        full_joint_angles = _joint_angles_from_raw(joint_data.raw)  # (1, J, 3)
        reordered = full_joint_angles[:, self._reorder_indices, :]
        if self._use_6d_rotations:
            reordered = euler_to_6d(reordered)
        flat = reordered.reshape(1, -1)

        with torch.no_grad():
            _, latent, _ = self._vae.encode(flat)
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        return latent  # (1, D)

    def _query_cache_get(self, key: str):
        cached = self._query_cache.get(key)
        if cached is None:
            return None
        self._query_cache.move_to_end(key)
        return cached

    def _query_cache_put(self, key: str, latent) -> None:
        if self._query_cache_max <= 0:
            return
        self._query_cache[key] = latent
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)

    def _get_or_compute_query_latent(
        self, image_bytes: bytes, ignore_query_cache: bool
    ):
        """Return (latent, cache_hit). Skips/refreshes cache if ignore_query_cache."""
        cache_key = hashlib.sha256(image_bytes).hexdigest()

        if not ignore_query_cache:
            cached = self._query_cache_get(cache_key)
            if cached is not None:
                self._query_cache_hits += 1
                return cached, True

        latent = self._embed_query_image(image_bytes)
        self._query_cache_put(cache_key, latent)
        self._query_cache_misses += 1
        return latent, False

    def _retrieve(self, query_latent, offset: int, limit: int):
        """Top-(offset + limit) retrieval; return rows offset:offset+limit."""
        import torch

        # Cosine similarity since both sides are unit vectors on the hypersphere.
        scores = (self._dataset_latents @ query_latent.T).squeeze(-1)  # (N,)
        k = max(0, offset + limit)
        if k == 0:
            return []
        k = min(k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k=k, largest=True, sorted=True)
        # Slice the requested window.
        window_indices = top_indices[offset:offset + limit].tolist()
        window_scores = top_scores[offset:offset + limit].tolist()
        return [
            {
                "rank": offset + i,
                "image_path": self._dataset_paths[idx],
                "cosine_similarity": float(score),
                "distance": float(1.0 - score),
            }
            for i, (idx, score) in enumerate(zip(window_indices, window_scores))
        ]

    @staticmethod
    def _read_image_b64(rel_image_path: str) -> str | None:
        """Load a pose image from the volume and return base64 contents."""
        clean = rel_image_path.lstrip("/")
        abs_path = Path(REMOTE_POSES_DIR) / clean
        if not abs_path.exists():
            return None
        with abs_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    # ---- HTTP endpoint ---------------------------------------------------

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def search(
        self,
        file: UploadFile,
        offset: int = 0,
        limit: int = 10,
        include_images: bool = True,
        ignore_query_cache: bool = False,
    ):
        """Top-K nearest pose retrieval against the precomputed VAE latents.

        Args:
            file:           multipart image upload (any format OpenCV can decode).
            offset:         number of top results to skip.
            limit:          number of results to return after the offset.
            include_images: if True, embed base64-encoded JPEGs for each result.
            ignore_query_cache:
                If True, recompute the query embedding from scratch and refresh
                the cache entry instead of reusing a previously-cached latent.
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        if offset < 0 or limit <= 0:
            raise HTTPException(
                status_code=400, detail="offset must be >= 0 and limit must be > 0."
            )
        if limit > 200:
            raise HTTPException(status_code=400, detail="limit capped at 200.")

        try:
            image_bytes = await file.read()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")

        try:
            query_latent, cache_hit = self._get_or_compute_query_latent(
                image_bytes, ignore_query_cache=ignore_query_cache
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        results = self._retrieve(query_latent, offset=offset, limit=limit)
        if include_images:
            for r in results:
                r["image_base64"] = self._read_image_b64(r["image_path"])

        return JSONResponse(
            {
                "offset": offset,
                "limit": limit,
                "total": int(self._dataset_latents.shape[0]),
                "latent_dim": int(self._dataset_latents.shape[1]),
                "results": results,
                "query_cache": {
                    "hit": cache_hit,
                    "size": len(self._query_cache),
                    "max": self._query_cache_max,
                },
            }
        )

    # ---- callable from Python clients ------------------------------------

    @modal.method()
    def search_bytes(
        self,
        image_bytes: bytes,
        offset: int = 0,
        limit: int = 10,
        include_images: bool = True,
        ignore_query_cache: bool = False,
    ) -> dict:
        """Same as the HTTP endpoint, but invokable via ``.remote()`` from
        any Python client using the Modal SDK (no HTTP round-trip required)."""
        query_latent, cache_hit = self._get_or_compute_query_latent(
            image_bytes, ignore_query_cache=ignore_query_cache
        )
        results = self._retrieve(query_latent, offset=offset, limit=limit)
        if include_images:
            for r in results:
                r["image_base64"] = self._read_image_b64(r["image_path"])
        return {
            "offset": offset,
            "limit": limit,
            "total": int(self._dataset_latents.shape[0]),
            "latent_dim": int(self._dataset_latents.shape[1]),
            "results": results,
            "query_cache": {
                "hit": cache_hit,
                "size": len(self._query_cache),
                "max": self._query_cache_max,
            },
        }


# ---------------------------------------------------------------------------
# Local entrypoint: smoke-test from a local file with `modal run API/main.py`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_test(
    image_path: str,
    offset: int = 0,
    limit: int = 5,
    include_images: bool = False,
    ignore_query_cache: bool = False,
):
    """Run a single retrieval request against a deployed/serving container.

    Usage:
        modal run API/main.py --image-path data/query/sit.jpg --limit 5
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    retrieval = VAERetrieval()
    result = retrieval.search_bytes.remote(
        image_bytes,
        offset=offset,
        limit=limit,
        include_images=include_images,
        ignore_query_cache=ignore_query_cache,
    )
    # Drop big base64 blobs from the printout
    for r in result.get("results", []):
        if "image_base64" in r and r["image_base64"]:
            r["image_base64"] = f"<{len(r['image_base64'])} bytes base64>"
    print(json.dumps(result, indent=2))
