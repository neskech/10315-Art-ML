"""Command-line wrapper for the retrieval Modal app.

The server stores the entire pose database as *precomputed* VAE latents in a
parquet file (output of ``data_generation/write_vae_features.py``). To also
support squared-distance retrieval over 3D keypoints, optionally upload the
``processed_poses.parquet`` (output of ``data_generation/write_poses.py``);
the server runs the scripted MHR model over it once at container start to
cache keypoints. SAM3D + VAE + MHR encoding only runs on the *query* image
at request time.

Usage examples
--------------

# 1) Upload the precomputed VAE-features parquet, the optional processed-poses
#    parquet (for squared-distance retrieval), the trained VAE checkpoint
#    (for encoding query images) and the poses image directory (for returning
#    base64 payloads + image-space dedup) to the Modal Volume backing the API.
python API/serve.py upload \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --processed-poses-parquet data/processed_poses.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --poses-dir data/poses

# 2) Develop locally (live-reload, ephemeral URL)
python API/serve.py serve \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt

# 3) Deploy to Modal (persistent URL)
python API/serve.py deploy \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt

# 4) Smoke-test against a deployed/serving container
python API/serve.py run \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --image data/query/sit.jpg --limit 5 --metric vae

The ``--parquet-path``, ``--processed-poses-parquet``, ``--vae-checkpoint`` and
``--poses-dir`` flags exist on every subcommand so the same invocation works
for upload + serve + deploy. They are exported as environment variables that
``API/main.py`` reads at module load time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_FILE = Path(__file__).resolve().parent / "main.py"

DEFAULT_PARQUET = REPO_ROOT / "data" / "vae_features_mhr_vae_latest.parquet"
DEFAULT_PROCESSED_POSES = REPO_ROOT / "data" / "processed_poses.parquet"
DEFAULT_VAE_CKPT = REPO_ROOT / "checkpoints" / "mhr_vae_latest.pt"
DEFAULT_POSES_DIR = REPO_ROOT / "data" / "poses"
DEFAULT_VOLUME_NAME = "vae-retrieval-data"
DEFAULT_APP_NAME = "vae-topk-retrieval"

# Modal caps scaledown_window at 3600 seconds.
_MAX_SCALEDOWN_WINDOW = 3600


def _scaledown_window_default() -> int:
    raw = int(os.environ.get("VAE_API_SCALEDOWN_WINDOW_SEC", str(_MAX_SCALEDOWN_WINDOW)))
    return min(_MAX_SCALEDOWN_WINDOW, max(1, raw))


def _parse_scaledown_window(s: str) -> int:
    v = int(s)
    return min(_MAX_SCALEDOWN_WINDOW, max(1, v))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=DEFAULT_PARQUET,
        help=(
            "Path to the local VAE-features parquet "
            "(output of data_generation/write_vae_features.py; columns: "
            "'image_path', 'vae_features'). (Default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--processed-poses-parquet",
        type=Path,
        default=DEFAULT_PROCESSED_POSES,
        help=(
            "Path to the local processed-poses parquet (output of "
            "data_generation/write_poses.py; columns include 'image_path' and "
            "'mhr_parameters'). Uploaded to /vol/processed_poses.parquet and "
            "used for squared-distance retrieval. Optional -- VAE retrieval "
            "still works without it. (Default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=DEFAULT_VAE_CKPT,
        help="Path to the trained VAE checkpoint (.pt). (Default: %(default)s)",
    )
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=DEFAULT_POSES_DIR,
        help=(
            "Path to the directory whose images are referenced by the parquet's "
            "`image_path` column. (Default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--volume-name",
        default=DEFAULT_VOLUME_NAME,
        help="Modal Volume name. (Default: %(default)s)",
    )
    parser.add_argument(
        "--app-name",
        default=DEFAULT_APP_NAME,
        help="Modal App name. (Default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("VAE_API_GPU", "T4"),
        help="GPU type (e.g. T4, L4, A10G). (Default: %(default)s)",
    )
    parser.add_argument(
        "--min-containers",
        type=int,
        default=int(os.environ.get("VAE_API_MIN_CONTAINERS", "1")),
        help=(
            "Modal min_containers: keep this many GPU workers warm (0 = scale "
            "to zero when idle; 1 avoids cold starts on most requests). "
            "(Default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--scaledown-window",
        type=_parse_scaledown_window,
        default=_scaledown_window_default(),
        help=(
            "Seconds Modal waits after the last request before scaling an idle "
            f"container to zero (max {_MAX_SCALEDOWN_WINDOW}). Larger = fewer "
            "cold starts. (Default: %(default)s)"
        ),
    )


def _validate_paths(
    args: argparse.Namespace,
    *,
    require_poses: bool,
    require_processed_poses: bool = False,
) -> None:
    missing = []
    if not args.parquet_path.exists():
        missing.append(f"parquet: {args.parquet_path}")
    if not args.vae_checkpoint.exists():
        missing.append(f"vae checkpoint: {args.vae_checkpoint}")
    if require_poses and not args.poses_dir.exists():
        missing.append(f"poses dir: {args.poses_dir}")
    if require_processed_poses and not args.processed_poses_parquet.exists():
        missing.append(f"processed poses parquet: {args.processed_poses_parquet}")
    if missing:
        sys.exit("Could not find:\n  - " + "\n  - ".join(missing))


def _env_for_modal(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["VAE_API_VOLUME_NAME"] = args.volume_name
    env["VAE_API_APP_NAME"] = args.app_name
    env["VAE_API_GPU"] = args.gpu
    env["VAE_API_MIN_CONTAINERS"] = str(args.min_containers)
    env["VAE_API_SCALEDOWN_WINDOW_SEC"] = str(args.scaledown_window)
    return env


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload local artifacts to the Modal Volume backing the API."""
    _validate_paths(args, require_poses=True)

    import modal  # local import so other subcommands work without modal installed in PATH only

    include_processed_poses = args.processed_poses_parquet.exists()
    if not include_processed_poses:
        print(
            f"Note: {args.processed_poses_parquet} not found; skipping upload of "
            "processed_poses.parquet (squared-distance metric will be unavailable)."
        )

    print(f"Connecting to Modal Volume '{args.volume_name}'...")
    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)

    print("Uploading artifacts (this may take a while for the poses dir)...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(args.parquet_path), "/vae_features.parquet")
        if include_processed_poses:
            batch.put_file(
                str(args.processed_poses_parquet), "/processed_poses.parquet"
            )
        batch.put_file(str(args.vae_checkpoint), "/vae.pt")
        batch.put_directory(str(args.poses_dir), "/poses")

    print(f"Done. Contents of '{args.volume_name}':")
    for entry in vol.listdir("/"):
        print(f"  {entry.path}  ({entry.type.name})")


def cmd_upload_artifacts_only(args: argparse.Namespace) -> None:
    """Upload only the parquet(s) + VAE checkpoint (skip the large poses dir)."""
    _validate_paths(args, require_poses=False)

    import modal

    include_processed_poses = args.processed_poses_parquet.exists()
    if not include_processed_poses:
        print(
            f"Note: {args.processed_poses_parquet} not found; skipping upload of "
            "processed_poses.parquet (squared-distance metric will be unavailable)."
        )

    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)
    print(f"Uploading parquet(s) + VAE checkpoint to '{args.volume_name}'...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(args.parquet_path), "/vae_features.parquet")
        if include_processed_poses:
            batch.put_file(
                str(args.processed_poses_parquet), "/processed_poses.parquet"
            )
        batch.put_file(str(args.vae_checkpoint), "/vae.pt")
    print("Done.")


def cmd_upload_processed_poses(args: argparse.Namespace) -> None:
    """Upload only the processed-poses parquet (for squared-distance retrieval)."""
    _validate_paths(args, require_poses=False, require_processed_poses=True)

    import modal

    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)
    print(
        f"Uploading processed_poses parquet to '{args.volume_name}' "
        f"from {args.processed_poses_parquet} ..."
    )
    with vol.batch_upload(force=True) as batch:
        batch.put_file(
            str(args.processed_poses_parquet), "/processed_poses.parquet"
        )
    print("Done.")


def _run_modal(subcommand: str, args: argparse.Namespace, extra: list[str]) -> int:
    cmd = ["modal", subcommand, str(MAIN_FILE), *extra]
    env = _env_for_modal(args)
    print("$", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def cmd_serve(args: argparse.Namespace) -> int:
    """`modal serve API/main.py` (live-reloading ephemeral deployment)."""
    return _run_modal("serve", args, [])


def cmd_deploy(args: argparse.Namespace) -> int:
    """`modal deploy API/main.py` (persistent deployment)."""
    return _run_modal("deploy", args, [])


def cmd_run(args: argparse.Namespace) -> int:
    """`modal run API/main.py --image-path ...` smoke test."""
    extra = [
        "--image-path",
        str(args.image),
        "--offset",
        str(args.offset),
        "--limit",
        str(args.limit),
        "--metric",
        str(args.metric),
        "--dedupe-epsilon",
        str(args.dedupe_epsilon),
        "--dedupe-ncc-threshold",
        str(args.dedupe_ncc_threshold),
        "--overfetch-factor",
        str(args.overfetch_factor),
    ]
    if args.include_images:
        extra.append("--include-images")
    if args.ignore_query_cache:
        extra.append("--ignore-query-cache")
    return _run_modal("run", args, extra)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for the VAE top-K pose retrieval Modal API."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser(
        "upload",
        help="Upload parquet + VAE checkpoint + poses dir to the Modal Volume.",
    )
    _shared_args(p_upload)
    p_upload.set_defaults(func=cmd_upload)

    p_upload_only = sub.add_parser(
        "upload-artifacts",
        help="Upload only parquet(s) + VAE checkpoint (skip the large poses dir).",
    )
    _shared_args(p_upload_only)
    p_upload_only.set_defaults(func=cmd_upload_artifacts_only)

    p_upload_poses = sub.add_parser(
        "upload-processed-poses",
        help="Upload only the processed-poses parquet (enables squared-distance metric).",
    )
    _shared_args(p_upload_poses)
    p_upload_poses.set_defaults(func=cmd_upload_processed_poses)

    p_serve = sub.add_parser(
        "serve",
        help="modal serve API/main.py (ephemeral, live-reloading deployment).",
    )
    _shared_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_deploy = sub.add_parser(
        "deploy",
        help="modal deploy API/main.py (persistent deployment).",
    )
    _shared_args(p_deploy)
    p_deploy.set_defaults(func=cmd_deploy)

    p_run = sub.add_parser(
        "run",
        help="modal run API/main.py with a smoke-test image.",
    )
    _shared_args(p_run)
    p_run.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Local query image to send through the deployed pipeline.",
    )
    p_run.add_argument("--offset", type=int, default=0)
    p_run.add_argument("--limit", type=int, default=5)
    p_run.add_argument(
        "--metric",
        choices=["vae", "squared"],
        default="vae",
        help=(
            "Retrieval metric: 'vae' for cosine similarity over VAE latents, "
            "'squared' for squared L2 over MHR 3D keypoints (requires "
            "processed_poses.parquet uploaded to the volume). (Default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--dedupe-epsilon",
        type=float,
        default=0.05,
        help=(
            "Multiscale RMS L2 threshold for image-space dedup. 0 disables. "
            "(Default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--dedupe-ncc-threshold",
        type=float,
        default=0.88,
        help=(
            "Crop-aware NCC threshold; if >= this, treat the pair as dup. "
            "0 disables. (Default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--overfetch-factor",
        type=int,
        default=5,
        help=(
            "When dedup is on, fetch (offset+limit)*factor candidates before "
            "filtering. (Default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--include-images",
        action="store_true",
        help="Include base64 image payloads in the response.",
    )
    p_run.add_argument(
        "--ignore-query-cache",
        action="store_true",
        help="Bypass the query-image latent cache and recompute SAM3D + VAE.",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
