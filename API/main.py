"""Modal app exposing top-K pose retrieval (VAE *or* squared keypoints) as a web API.

This module builds a Modal `App` whose container:

  1. Bakes in the project source code and the (large, static) SAM3D + MHR
     checkpoints as image layers so cold starts only need to download the
     small, user-supplied artifacts.
  2. Mounts a Modal `Volume` containing the user-supplied artifacts:
        - ``vae_features.parquet`` (columns: ``image_path``, ``vae_features``;
          produced by ``data_generation/write_vae_features.py``; this is the
          entire pose database represented purely as precomputed latents).
        - ``processed_poses.parquet`` (columns include ``image_path`` and
          ``mhr_parameters``; produced by ``data_generation/write_poses.py``).
          Required only when you want to serve the "squared-distance"
          metric — we run the scripted MHR model once at container start
          over every row to cache their 3D keypoints.
        - VAE checkpoint (``vae.pt``) -- only used for encoding *query* images.
        - The pose images directory (``poses/``) -- used for returning
          base64 payloads *and* for image-space deduplication of results.
     The volume is populated from the local machine via ``API/serve.py``.
  3. On container start (`@modal.enter`), loads the SAM3D pose estimator and
     the trained VAE, reads the precomputed (N, latent_dim) dataset latents
     straight out of the VAE-features parquet, and (if present) batch-runs
     the scripted MHR model on ``processed_poses.parquet`` to cache a
     ``(N, J, 3)`` tensor of 3D keypoints for squared-distance retrieval.
  4. Exposes a single POST endpoint that accepts a multipart image upload
     plus ``offset``, ``limit`` and ``metric`` query parameters and returns
     the matching pose images as base64-encoded JPEGs, optionally with
     image-space deduplication applied against the query and against
     previously-returned hits (mirrors
     ``visualization/visualize_retrieval.py``).

Query-time pipeline (both metrics): SAM3D body inference -> MHR parameters.
For ``metric=vae`` the MHR -> joint-angle -> VAE encoder path gives a unit
latent; retrieval is ``(N, D) @ (D, 1)`` cosine similarity against the
precomputed dataset latents. For ``metric=squared`` the scripted MHR model
turns the query's MHR params into ``(J, 3)`` keypoints and retrieval is
``sum_j ||kp_query - kp_dataset||^2`` against the cached dataset keypoints.

The path to the local parquets, VAE checkpoint and poses directory are
passed as command-line arguments to ``API/serve.py``; this module reads
them from environment variables that the wrapper sets before invoking
``modal``.
"""

from __future__ import annotations

import base64
import hashlib
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
REMOTE_VAE_FEATURES = f"{REMOTE_VOLUME_MOUNT}/vae_features.parquet"
REMOTE_PROCESSED_POSES = f"{REMOTE_VOLUME_MOUNT}/processed_poses.parquet"
REMOTE_VAE_CKPT = f"{REMOTE_VOLUME_MOUNT}/vae.pt"
REMOTE_POSES_DIR = f"{REMOTE_VOLUME_MOUNT}/poses"
REMOTE_MHR_MODEL = f"{REMOTE_REPO}/checkpoints/sam3d/dinov3/assets/mhr_model.pt"

# Dedup config (mirrors visualization/visualize_retrieval.py):
# letterbox each image to these square sizes, take the min RMS across scales.
DEDUPE_SIZES: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256)
DEDUPE_NCC_MAX_SIDE = 640  # downscale larger side to this before template match
# Batch size when running the scripted MHR model over processed_poses.parquet
# to precompute 3D keypoints for squared-distance retrieval.
MHR_PRECOMPUTE_BATCH = int(os.environ.get("VAE_API_MHR_BATCH", "256"))

VOLUME_NAME = os.environ.get("VAE_API_VOLUME_NAME", "vae-retrieval-data")
APP_NAME = os.environ.get("VAE_API_APP_NAME", "vae-topk-retrieval")
GPU_TYPE = os.environ.get("VAE_API_GPU", "T4")
QUERY_CACHE_MAX_ENTRIES = int(os.environ.get("VAE_API_QUERY_CACHE_MAX", "1024"))
# Keep ≥1 warm replica by default so SAM3D/ViTDet load once; use 0 for min idle cost.
MIN_CONTAINERS = max(0, int(os.environ.get("VAE_API_MIN_CONTAINERS", "1")))
# Seconds before Modal scales an idle container to zero (longer = fewer cold starts).
# Modal caps this at 3600s (1h).
_SCALEDOWN_RAW = int(os.environ.get("VAE_API_SCALEDOWN_WINDOW_SEC", "3600"))
SCALEDOWN_WINDOW_SEC = min(3600, max(1, _SCALEDOWN_RAW))

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
    scaledown_window=SCALEDOWN_WINDOW_SEC,
    min_containers=MIN_CONTAINERS,
)
class VAERetrieval:
    """Loads VAE + SAM3D once per container and serves nearest-neighbour queries."""

    # ---- lifecycle -------------------------------------------------------

    @modal.enter()
    def _load(self) -> None:
        # Make local repo importable.
        if REMOTE_REPO not in sys.path:
            sys.path.insert(0, REMOTE_REPO)

        import json as _json

        import numpy as np
        import pandas as pd
        import torch

        # Imports deferred to runtime so the local image-build process does
        # not need to import torch / sam3d / etc.
        from pose_module.inference import SAM3DBodyInference  # type: ignore
        from vae_features.model.feedForwardVae import FeedForwardVAE  # type: ignore
        from vae_features.utils.feedForward import Norm  # type: ignore
        from vae_features.utils.mhrToJointData import PostProcessor  # type: ignore
        from vae_features.utils.skeletonFormat import SkeletonFormat  # type: ignore

        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger("vae_retrieval")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log.info("Using device: %s", self._device)

        # ---- Config shared between query pipeline and dataset parquet ----
        train_dir = Path(REMOTE_REPO) / "vae_features" / "train"
        skeleton_format = SkeletonFormat.from_json_file(
            train_dir / "mhr_skeleton_format.json"
        )

        if not Path(REMOTE_VAE_FEATURES).exists():
            raise FileNotFoundError(
                f"VAE-features parquet not found at {REMOTE_VAE_FEATURES}. "
                f"Run `python API/serve.py upload-artifacts --parquet-path "
                f"data/vae_features_<checkpoint>.parquet ...` first. The "
                f"parquet must be the output of "
                f"`data_generation/write_vae_features.py` (columns: "
                f"'image_path', 'vae_features')."
            )
        if not Path(REMOTE_VAE_CKPT).exists():
            raise FileNotFoundError(
                f"VAE checkpoint not found at {REMOTE_VAE_CKPT}. Did you run "
                f"`python API/serve.py upload --vae-checkpoint ...`?"
            )

        # Load checkpoint config so the query-time encoder matches how the
        # dataset parquet was produced (same rotation format, same encoder).
        self._log.info("Loading VAE checkpoint from %s", REMOTE_VAE_CKPT)
        ckpt = torch.load(REMOTE_VAE_CKPT, map_location=self._device, weights_only=False)
        config = ckpt.get("config", {})
        use_6d_rotations = bool(config.get("USE_6D_ROTATIONS", False))
        per_joint = 6 if use_6d_rotations else 3

        # Query-time MHR -> joint-angle plumbing. Previously delegated to
        # MHRPoseDataset; reproduced here directly so we don't need the
        # pose-parameter columns in the parquet (we only have latents now).
        self._post_processor = PostProcessor(self._device)
        with (train_dir / "joint_names.json").open("r", encoding="utf-8") as f:
            source_joint_names = _json.load(f)["joint_names"]
        target_joint_names = skeleton_format.get_joint_names()
        source_idx_by_name = {name: idx for idx, name in enumerate(source_joint_names)}
        self._reorder_indices = torch.tensor(
            [source_idx_by_name[name] for name in target_joint_names],
            dtype=torch.long,
            device=self._device,
        )
        self._use_6d_rotations = use_6d_rotations

        # ---- Pose estimator (heavy: DiNOv3 + MHR head) ----
        self._log.info("Loading SAM3D body inference model...")
        self._estimator = SAM3DBodyInference(
            device=self._device, use_torch_compile=False
        )

        # ---- Scripted MHR (used for squared-distance keypoint retrieval) ----
        # ``mhr_parameters`` -> (vertices, skeleton_state); we only care about
        # the skeleton's first-3 coords, in the same normalization used by
        # ``pose_module.interpret_mhr_params.PoseDataInterpreter`` (divide by
        # 100 and flip Y/Z) so query vs dataset keypoints live in the same
        # frame.
        self._log.info("Loading scripted MHR model from %s", REMOTE_MHR_MODEL)
        self._mhr_model = torch.jit.load(REMOTE_MHR_MODEL, map_location=self._device)
        self._mhr_model.eval()

        # ---- VAE encoder (only used on query images) ----
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

        # ---- Dataset latents come straight from the parquet ----
        # We deliberately do NOT re-run SAM3D or VAE encode on the dataset:
        # that work was already done offline by
        # data_generation/write_vae_features.py and serialized into the
        # 'vae_features' column.
        self._log.info("Reading precomputed latents from %s", REMOTE_VAE_FEATURES)
        df = pd.read_parquet(REMOTE_VAE_FEATURES)
        required = {"image_path", "vae_features"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{REMOTE_VAE_FEATURES} is missing required columns. Need "
                f"{sorted(required)}, got {list(df.columns)}. Did you upload "
                f"the right parquet (output of "
                f"data_generation/write_vae_features.py)?"
            )

        latents_np = np.stack(
            [np.asarray(v, dtype=np.float32) for v in df["vae_features"].tolist()],
            axis=0,
        )
        latents = torch.from_numpy(latents_np).to(self._device)
        # Ensure unit vectors so cosine similarity reduces to a single matmul.
        latents = latents / (latents.norm(dim=-1, keepdim=True) + 1e-8)
        self._dataset_latents = latents
        self._dataset_paths: list[str] = [str(p) for p in df["image_path"].tolist()]
        self._log.info(
            "Loaded %d precomputed latents of dim %d from parquet",
            self._dataset_latents.shape[0],
            self._dataset_latents.shape[1],
        )

        # ---- Optional: precompute 3D keypoints for squared-distance retrieval ----
        # Only needed when the client asks for ``metric=squared``. If the
        # parquet is absent we keep the API up for VAE retrieval and error
        # clearly on squared requests.
        self._kp_3d = None  # type: ignore[assignment]
        self._kp_paths: list[str] = []
        self._kp_dim: tuple[int, int] = (0, 0)
        if Path(REMOTE_PROCESSED_POSES).exists():
            self._log.info(
                "Reading processed poses from %s for squared-distance retrieval",
                REMOTE_PROCESSED_POSES,
            )
            poses_df = pd.read_parquet(REMOTE_PROCESSED_POSES)
            if "image_path" not in poses_df.columns or "mhr_parameters" not in poses_df.columns:
                self._log.warning(
                    "%s missing 'image_path'/'mhr_parameters' columns (got %s); "
                    "squared-distance retrieval will be unavailable.",
                    REMOTE_PROCESSED_POSES,
                    list(poses_df.columns),
                )
            else:
                mhr_np = np.stack(
                    [
                        np.asarray(v, dtype=np.float32)
                        for v in poses_df["mhr_parameters"].tolist()
                    ],
                    axis=0,
                )
                self._log.info(
                    "Batch-running scripted MHR on %d rows (batch=%d) to cache "
                    "keypoints_3d for squared retrieval",
                    mhr_np.shape[0],
                    MHR_PRECOMPUTE_BATCH,
                )
                kp_chunks: list[torch.Tensor] = []
                for start in range(0, mhr_np.shape[0], MHR_PRECOMPUTE_BATCH):
                    batch = torch.from_numpy(
                        mhr_np[start : start + MHR_PRECOMPUTE_BATCH]
                    ).to(self._device)
                    kp_chunks.append(self._mhr_params_to_keypoints_3d(batch).cpu())
                self._kp_3d = torch.cat(kp_chunks, dim=0).to(self._device)
                self._kp_paths = [str(p) for p in poses_df["image_path"].tolist()]
                self._kp_dim = (
                    int(self._kp_3d.shape[1]),
                    int(self._kp_3d.shape[2]),
                )
                self._log.info(
                    "Cached keypoints tensor %s for squared-distance retrieval",
                    tuple(self._kp_3d.shape),
                )
        else:
            self._log.info(
                "No processed_poses parquet at %s; squared-distance retrieval "
                "will not be available until you upload one.",
                REMOTE_PROCESSED_POSES,
            )

        # In-process LRU cache for query-image latents keyed by SHA-256 of bytes.
        # SAM3D + VAE encode is the dominant per-request cost; caching pagination
        # of the same image avoids re-running the full pipeline.
        self._query_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._query_cache_max = QUERY_CACHE_MAX_ENTRIES
        self._query_cache_hits = 0
        self._query_cache_misses = 0

    # ---- helpers ---------------------------------------------------------

    def _mhr_params_to_keypoints_3d(self, mhr_params):
        """Batch-run the scripted MHR model, return (B, J, 3) 3D keypoints.

        Reproduces ``pose_module.interpret_mhr_params.PoseDataInterpreter``:
        zeros for id/expression coeffs, divide skeleton_state by 100, flip
        Y/Z. Keeping the same frame on both sides is what makes
        ``squaredDistanceMetric`` meaningful.
        """
        import torch

        b = mhr_params.shape[0]
        id_coeffs = torch.zeros(b, 45, device=mhr_params.device, dtype=mhr_params.dtype)
        expr_coeffs = torch.zeros(b, 72, device=mhr_params.device, dtype=mhr_params.dtype)
        with torch.no_grad():
            _, skeleton_state = self._mhr_model(id_coeffs, mhr_params, expr_coeffs)
        kp = skeleton_state[..., :3] / 100.0
        kp = kp.clone()
        kp[..., [1, 2]] *= -1.0
        return kp

    def _embed_query(self, image_bytes: bytes) -> dict:
        """Run SAM3D once on a query image and return *everything* downstream.

        SAM3D body inference is the dominant per-query cost (10–30x more
        than the MHR/VAE forward passes), so we eagerly derive *both* the
        VAE latent and the scripted-MHR 3D keypoints from a single SAM3D
        pass and stash the raw RGB too. The caller caches this entire dict
        keyed by the bytes hash (see ``_get_or_compute_query``), which
        means a client that later switches ``metric`` from ``"vae"`` to
        ``"squared"`` (or vice versa) on the same query gets a cache hit
        and SAM3D does **not** re-run -- we just read the already-computed
        ``keypoints_3d`` or ``latent`` out of the cache. Same goes for
        pagination (``offset`` / ``limit`` changes) and dedup-parameter
        tweaks.

        Returns a dict ``{"latent": (1, D), "keypoints_3d": (1, J, 3),
        "rgb": np.ndarray[H, W, 3] uint8}``.
        """
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
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

        kp_3d = self._mhr_params_to_keypoints_3d(mhr_tensor)  # (1, J, 3)

        return {"latent": latent, "keypoints_3d": kp_3d, "rgb": rgb}

    def _query_cache_get(self, key: str):
        cached = self._query_cache.get(key)
        if cached is None:
            return None
        self._query_cache.move_to_end(key)
        return cached

    def _query_cache_put(self, key: str, value) -> None:
        if self._query_cache_max <= 0:
            return
        self._query_cache[key] = value
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)

    def _get_or_compute_query(
        self, image_bytes: bytes, ignore_query_cache: bool
    ):
        """Return (cached_dict, cache_hit). See ``_embed_query`` for shape."""
        cache_key = hashlib.sha256(image_bytes).hexdigest()

        if not ignore_query_cache:
            cached = self._query_cache_get(cache_key)
            if cached is not None:
                self._query_cache_hits += 1
                return cached, True

        cached = self._embed_query(image_bytes)
        self._query_cache_put(cache_key, cached)
        self._query_cache_misses += 1
        return cached, False

    # ---- ranked retrieval (no dedup) -------------------------------------

    def _rank_vae(self, query_latent, k: int):
        """Top-k cosine-similarity ranking against the VAE latents."""
        import torch

        if k <= 0:
            return [], [], []
        scores = (self._dataset_latents @ query_latent.T).squeeze(-1)  # (N,)
        k = min(k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k=k, largest=True, sorted=True)
        indices = top_indices.tolist()
        sims = top_scores.tolist()
        return (
            [self._dataset_paths[i] for i in indices],
            sims,
            [float(1.0 - s) for s in sims],
        )

    def _rank_squared(self, query_keypoints_3d, k: int):
        """Top-k squared-distance ranking against the dataset keypoints."""
        import torch

        if self._kp_3d is None:
            raise ValueError(
                "Squared-distance retrieval is not available: processed_poses.parquet "
                "was not uploaded to the Modal volume. Re-run "
                "`python API/serve.py upload --processed-poses-parquet "
                "data/processed_poses.parquet ...`."
            )
        if k <= 0:
            return [], [], []
        diff = self._kp_3d - query_keypoints_3d  # (N, J, 3) - (1, J, 3)
        dists = (diff * diff).sum(dim=(1, 2))  # (N,)
        k = min(k, dists.shape[0])
        top_dists, top_indices = torch.topk(dists, k=k, largest=False, sorted=True)
        indices = top_indices.tolist()
        distances = [float(d) for d in top_dists.tolist()]
        return (
            [self._kp_paths[i] for i in indices],
            [None] * len(indices),
            distances,
        )

    # ---- image deduplication (mirrors visualize_retrieval.py) ------------

    @staticmethod
    def _letterbox_square_rgb(img, size: int):
        """Fit ``img`` in a ``size``x``size`` square, preserve aspect, black-pad."""
        import cv2
        import numpy as np

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

    @classmethod
    def _letterbox_pyramid(cls, img) -> list:
        """Precompute the letterbox at every dedupe size as float32/255."""
        import numpy as np

        return [
            cls._letterbox_square_rgb(img, s).astype(np.float32) / 255.0
            for s in DEDUPE_SIZES
        ]

    @staticmethod
    def _rms_from_pyramids(pyr_a: list, pyr_b: list) -> float:
        """Minimum RMS L2 across the scale pyramid."""
        import numpy as np

        best = float("inf")
        for a, b in zip(pyr_a, pyr_b):
            rms = float(np.sqrt(np.mean((a - b) ** 2)))
            if rms < best:
                best = rms
        return best

    @staticmethod
    def _prepare_ncc_gray(img):
        """Grayscale + optional downscale (max side ``DEDUPE_NCC_MAX_SIDE``)."""
        import cv2

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        m = max(h, w)
        if m > DEDUPE_NCC_MAX_SIDE:
            scale = DEDUPE_NCC_MAX_SIDE / m
            gray = cv2.resize(
                gray,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    @staticmethod
    def _max_ncc_if_crop(gray_a, gray_b) -> float:
        """TM_CCOEFF_NORMED with the smaller image as template. 0 if not applicable."""
        import cv2

        if gray_a is None or gray_b is None:
            return 0.0
        ha, wa = gray_a.shape[:2]
        hb, wb = gray_b.shape[:2]
        if ha * wa == 0 or hb * wb == 0:
            return 0.0
        if ha * wa > hb * wb:
            gray_a, gray_b = gray_b, gray_a
            ha, wa, hb, wb = hb, wb, ha, wa
        if ha < 3 or wa < 3:
            return 0.0
        if ha > hb or wa > wb:
            return 0.0
        res = cv2.matchTemplate(gray_b, gray_a, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    @classmethod
    def _is_duplicate(
        cls,
        pyr_a: list,
        gray_a,
        pyr_b: list,
        gray_b,
        *,
        epsilon: float,
        ncc_threshold: float,
    ) -> bool:
        if epsilon > 0.0 and cls._rms_from_pyramids(pyr_a, pyr_b) < epsilon:
            return True
        if ncc_threshold > 0.0 and cls._max_ncc_if_crop(gray_a, gray_b) >= ncc_threshold:
            return True
        return False

    @staticmethod
    def _read_pose_raw(rel_image_path: str) -> bytes | None:
        """Read raw bytes of a pose image from the mounted volume."""
        clean = rel_image_path.lstrip("/")
        abs_path = Path(REMOTE_POSES_DIR) / clean
        if not abs_path.exists():
            return None
        with abs_path.open("rb") as f:
            return f.read()

    @staticmethod
    def _decode_rgb(raw: bytes):
        """Decode JPEG/PNG bytes to H,W,3 RGB uint8."""
        import cv2
        import numpy as np

        if not raw:
            return None
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---- unified retrieval + dedup pipeline ------------------------------

    def _retrieve_with_dedup(
        self,
        query: dict,
        *,
        metric: str,
        offset: int,
        limit: int,
        dedupe_epsilon: float,
        dedupe_ncc_threshold: float,
        overfetch_factor: int,
    ) -> tuple[list[dict], dict]:
        """Rank the dataset, apply image-space dedup (incl. vs. query), return results.

        Returns ``(window, dedup_info)``. Each window entry contains
        ``rank``, ``image_path``, ``distance``, ``cosine_similarity`` (or
        None for squared), and an internal ``_raw_bytes`` key holding the
        image bytes we already read for dedup -- callers can use that to
        skip a second disk read when base64-encoding results.
        """
        end = offset + limit
        do_dedup = dedupe_epsilon > 0.0 or dedupe_ncc_threshold > 0.0
        overfetch = max(1, int(overfetch_factor))
        fetch_k = end * overfetch if do_dedup else end

        if metric == "vae":
            paths, sims, dists = self._rank_vae(query["latent"], fetch_k)
            total = int(self._dataset_latents.shape[0])
        elif metric == "squared":
            paths, sims, dists = self._rank_squared(query["keypoints_3d"], fetch_k)
            total = int(self._kp_3d.shape[0]) if self._kp_3d is not None else 0
        else:
            raise ValueError(
                f"Unknown metric {metric!r}; expected 'vae' or 'squared'."
            )

        # Short-circuit when dedup is disabled: slice window directly.
        if not do_dedup:
            window = []
            for i, (p, s, d) in enumerate(zip(paths[offset:end], sims[offset:end], dists[offset:end])):
                window.append(
                    {
                        "rank": offset + i,
                        "image_path": p,
                        "cosine_similarity": s,
                        "distance": d,
                        "_raw_bytes": None,
                    }
                )
            return window, {
                "enabled": False,
                "epsilon": dedupe_epsilon,
                "ncc_threshold": dedupe_ncc_threshold,
                "overfetch_factor": overfetch,
                "candidates_considered": len(paths),
                "unique_before_window": len(window),
                "total": total,
            }

        # ---- Dedup path: load query pyramid + grays once -----------------
        query_rgb = query["rgb"]
        query_pyr = self._letterbox_pyramid(query_rgb)
        query_gray = (
            self._prepare_ncc_gray(query_rgb) if dedupe_ncc_threshold > 0.0 else None
        )

        unique: list[dict] = []
        unique_pyrs: list[list] = []
        unique_grays: list = []
        seen_paths: set[str] = set()
        missing_during_dedup = 0

        for p, s, d in zip(paths, sims, dists):
            if p in seen_paths:
                continue
            raw = self._read_pose_raw(p)
            img = self._decode_rgb(raw) if raw is not None else None
            if img is None:
                # Still keep the candidate; we just can't dedup it. This
                # keeps the endpoint useful when a file is temporarily
                # missing from the volume.
                missing_during_dedup += 1
                unique.append(
                    {
                        "image_path": p,
                        "cosine_similarity": s,
                        "distance": d,
                        "_raw_bytes": raw,
                    }
                )
                seen_paths.add(p)
                if len(unique) >= end:
                    break
                continue

            cand_pyr = self._letterbox_pyramid(img)
            cand_gray = (
                self._prepare_ncc_gray(img) if dedupe_ncc_threshold > 0.0 else None
            )

            if self._is_duplicate(
                query_pyr,
                query_gray,
                cand_pyr,
                cand_gray,
                epsilon=dedupe_epsilon,
                ncc_threshold=dedupe_ncc_threshold,
            ):
                continue
            if any(
                self._is_duplicate(
                    kept_pyr,
                    kept_gray,
                    cand_pyr,
                    cand_gray,
                    epsilon=dedupe_epsilon,
                    ncc_threshold=dedupe_ncc_threshold,
                )
                for kept_pyr, kept_gray in zip(unique_pyrs, unique_grays)
            ):
                continue

            unique.append(
                {
                    "image_path": p,
                    "cosine_similarity": s,
                    "distance": d,
                    "_raw_bytes": raw,
                }
            )
            unique_pyrs.append(cand_pyr)
            unique_grays.append(cand_gray)
            seen_paths.add(p)
            if len(unique) >= end:
                break

        window = []
        for i, entry in enumerate(unique[offset:end]):
            entry_out = dict(entry)
            entry_out["rank"] = offset + i
            window.append(entry_out)

        return window, {
            "enabled": True,
            "epsilon": dedupe_epsilon,
            "ncc_threshold": dedupe_ncc_threshold,
            "overfetch_factor": overfetch,
            "candidates_considered": len(paths),
            "unique_before_window": len(unique),
            "missing_during_dedup": missing_during_dedup,
            "total": total,
        }

    @staticmethod
    def _read_image_b64(rel_image_path: str) -> str | None:
        """Load a pose image from the volume and return base64 contents."""
        clean = rel_image_path.lstrip("/")
        abs_path = Path(REMOTE_POSES_DIR) / clean
        if not abs_path.exists():
            return None
        with abs_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _attach_result_images(self, results: list[dict]) -> dict:
        """Attach base64 ``image_base64`` to each result; reload volume once if needed.

        If ``_raw_bytes`` was already populated during dedup we reuse those
        bytes instead of re-reading the file. Modal Volumes mounted into a
        container reflect volume state as of container start; if a file is
        missing we call ``volume.reload()`` once and retry.
        """
        missing_paths: list[str] = []
        for r in results:
            raw = r.pop("_raw_bytes", None)
            if raw is None:
                raw = self._read_pose_raw(r["image_path"])
            if raw is None:
                r["image_base64"] = None
                missing_paths.append(r["image_path"])
            else:
                r["image_base64"] = base64.b64encode(raw).decode("ascii")

        reloaded = False
        if missing_paths:
            try:
                volume.reload()
                reloaded = True
                still_missing: list[str] = []
                for r in results:
                    if r.get("image_base64") is not None:
                        continue
                    raw = self._read_pose_raw(r["image_path"])
                    if raw is None:
                        still_missing.append(r["image_path"])
                        r["image_base64"] = None
                    else:
                        r["image_base64"] = base64.b64encode(raw).decode("ascii")
                missing_paths = still_missing
            except Exception as exc:  # pragma: no cover - best-effort
                self._log.warning("volume.reload() failed: %s", exc)

        if missing_paths:
            self._log.warning(
                "Could not load %d result image(s) from the volume. "
                "Example missing path: %r",
                len(missing_paths),
                missing_paths[0],
            )

        return {
            "missing_image_count": len(missing_paths),
            "missing_image_examples": missing_paths[:5],
            "volume_reloaded": reloaded,
        }

    # ---- shared response assembly ----------------------------------------

    def _run_search(
        self,
        image_bytes: bytes,
        *,
        offset: int,
        limit: int,
        metric: str,
        include_images: bool,
        ignore_query_cache: bool,
        dedupe_epsilon: float,
        dedupe_ncc_threshold: float,
        overfetch_factor: int,
    ) -> dict:
        metric_normalized = (metric or "vae").lower()
        if metric_normalized not in {"vae", "squared"}:
            raise ValueError(
                f"Unknown metric {metric!r}; expected 'vae' or 'squared'."
            )

        query, cache_hit = self._get_or_compute_query(
            image_bytes, ignore_query_cache=ignore_query_cache
        )

        results, dedup_info = self._retrieve_with_dedup(
            query,
            metric=metric_normalized,
            offset=offset,
            limit=limit,
            dedupe_epsilon=dedupe_epsilon,
            dedupe_ncc_threshold=dedupe_ncc_threshold,
            overfetch_factor=overfetch_factor,
        )

        image_status: dict = {}
        if include_images:
            image_status = self._attach_result_images(results)
        else:
            # Dedup left _raw_bytes dangling; strip it so it isn't serialized.
            for r in results:
                r.pop("_raw_bytes", None)

        return {
            "metric": metric_normalized,
            "offset": offset,
            "limit": limit,
            "total": dedup_info["total"],
            "latent_dim": int(self._dataset_latents.shape[1]),
            "keypoints_shape": list(self._kp_dim),
            "results": results,
            "query_cache": {
                "hit": cache_hit,
                "size": len(self._query_cache),
                "max": self._query_cache_max,
            },
            "dedup": dedup_info,
            "image_status": image_status,
        }

    # ---- HTTP endpoint ---------------------------------------------------

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def search(
        self,
        file: UploadFile,
        offset: int = 0,
        limit: int = 10,
        metric: str = "vae",
        include_images: bool = True,
        ignore_query_cache: bool = False,
        dedupe_epsilon: float = 0.05,
        dedupe_ncc_threshold: float = 0.88,
        overfetch_factor: int = 5,
    ):
        """Top-K nearest pose retrieval with image-space deduplication.

        Args:
            file:           multipart image upload (any format OpenCV can decode).
            offset:         number of deduped results to skip.
            limit:          number of deduped results to return after the offset.
            metric:         ``"vae"`` (cosine similarity over VAE latents) or
                            ``"squared"`` (squared L2 over MHR 3D keypoints).
            include_images: if True, embed base64-encoded JPEGs for each result.
            ignore_query_cache:
                If True, recompute SAM3D + VAE + keypoints from scratch and
                refresh the cache entry instead of reusing a cached embedding.
            dedupe_epsilon:
                Minimum multiscale RMS L2 (letterboxed RGB, [0,1]) below which
                two images are considered duplicates. Set to 0 to disable RMS
                deduplication.
            dedupe_ncc_threshold:
                If > 0, also treat a pair as duplicate when OpenCV
                ``matchTemplate`` NCC (smaller image vs larger) reaches this
                score — catches crop vs full-image pairs. Set to 0 to disable.
            overfetch_factor:
                Multiplier on ``offset + limit`` used when deduplication is
                enabled; the server retrieves that many candidates before
                filtering. Ignored when dedup is disabled.
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        if offset < 0 or limit <= 0:
            raise HTTPException(
                status_code=400, detail="offset must be >= 0 and limit must be > 0."
            )
        if limit > 200:
            raise HTTPException(status_code=400, detail="limit capped at 200.")
        if dedupe_epsilon < 0.0:
            raise HTTPException(status_code=400, detail="dedupe_epsilon must be >= 0.")
        if not (0.0 <= dedupe_ncc_threshold <= 1.0):
            raise HTTPException(
                status_code=400, detail="dedupe_ncc_threshold must be in [0, 1]."
            )
        if overfetch_factor < 1:
            raise HTTPException(
                status_code=400, detail="overfetch_factor must be >= 1."
            )

        try:
            image_bytes = await file.read()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")

        try:
            payload = self._run_search(
                image_bytes,
                offset=offset,
                limit=limit,
                metric=metric,
                include_images=include_images,
                ignore_query_cache=ignore_query_cache,
                dedupe_epsilon=dedupe_epsilon,
                dedupe_ncc_threshold=dedupe_ncc_threshold,
                overfetch_factor=overfetch_factor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return JSONResponse(payload)

    # ---- callable from Python clients ------------------------------------

    @modal.method()
    def search_bytes(
        self,
        image_bytes: bytes,
        offset: int = 0,
        limit: int = 10,
        metric: str = "vae",
        include_images: bool = True,
        ignore_query_cache: bool = False,
        dedupe_epsilon: float = 0.05,
        dedupe_ncc_threshold: float = 0.88,
        overfetch_factor: int = 5,
    ) -> dict:
        """Same as the HTTP endpoint, but invokable via ``.remote()`` from
        any Python client using the Modal SDK (no HTTP round-trip required)."""
        return self._run_search(
            image_bytes,
            offset=offset,
            limit=limit,
            metric=metric,
            include_images=include_images,
            ignore_query_cache=ignore_query_cache,
            dedupe_epsilon=dedupe_epsilon,
            dedupe_ncc_threshold=dedupe_ncc_threshold,
            overfetch_factor=overfetch_factor,
        )


# ---------------------------------------------------------------------------
# Local entrypoint: smoke-test from a local file with `modal run API/main.py`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_test(
    image_path: str,
    offset: int = 0,
    limit: int = 5,
    metric: str = "vae",
    include_images: bool = False,
    ignore_query_cache: bool = False,
    dedupe_epsilon: float = 0.05,
    dedupe_ncc_threshold: float = 0.88,
    overfetch_factor: int = 5,
):
    """Run a single retrieval request against a deployed/serving container.

    Usage:
        modal run API/main.py --image-path data/query/sit.jpg --limit 5 --metric vae
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    retrieval = VAERetrieval()
    result = retrieval.search_bytes.remote(
        image_bytes,
        offset=offset,
        limit=limit,
        metric=metric,
        include_images=include_images,
        ignore_query_cache=ignore_query_cache,
        dedupe_epsilon=dedupe_epsilon,
        dedupe_ncc_threshold=dedupe_ncc_threshold,
        overfetch_factor=overfetch_factor,
    )
    for r in result.get("results", []):
        if "image_base64" in r and r["image_base64"]:
            r["image_base64"] = f"<{len(r['image_base64'])} bytes base64>"
    print(json.dumps(result, indent=2))
