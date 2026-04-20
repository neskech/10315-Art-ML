"""Command-line wrapper for the VAE retrieval Modal app.

The server stores the entire pose database as *precomputed* VAE latents in a
single parquet file (output of ``data_generation/write_vae_features.py``).
SAM3D + VAE encoding only runs on the *query* image at request time.

Usage examples
--------------

# 1) Upload the precomputed VAE-features parquet, the trained VAE checkpoint
#    (for encoding query images) and the poses image directory (for returning
#    base64 payloads) to the Modal Volume backing the API.
python API/serve.py upload \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
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
    --image data/query/sit.jpg --limit 5

The ``--parquet-path``, ``--vae-checkpoint`` and ``--poses-dir`` flags exist on
every subcommand so the same invocation works for upload + serve + deploy. They
are exported as environment variables that ``API/main.py`` reads at module
load time.
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
DEFAULT_VAE_CKPT = REPO_ROOT / "checkpoints" / "mhr_vae_latest.pt"
DEFAULT_POSES_DIR = REPO_ROOT / "data" / "poses"
DEFAULT_VOLUME_NAME = "vae-retrieval-data"
DEFAULT_APP_NAME = "vae-topk-retrieval"


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


def _validate_paths(args: argparse.Namespace, *, require_poses: bool) -> None:
    missing = []
    if not args.parquet_path.exists():
        missing.append(f"parquet: {args.parquet_path}")
    if not args.vae_checkpoint.exists():
        missing.append(f"vae checkpoint: {args.vae_checkpoint}")
    if require_poses and not args.poses_dir.exists():
        missing.append(f"poses dir: {args.poses_dir}")
    if missing:
        sys.exit("Could not find:\n  - " + "\n  - ".join(missing))


def _env_for_modal(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["VAE_API_VOLUME_NAME"] = args.volume_name
    env["VAE_API_APP_NAME"] = args.app_name
    env["VAE_API_GPU"] = args.gpu
    return env


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload local artifacts to the Modal Volume backing the API."""
    _validate_paths(args, require_poses=True)

    import modal  # local import so other subcommands work without modal installed in PATH only

    print(f"Connecting to Modal Volume '{args.volume_name}'...")
    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)

    print("Uploading artifacts (this may take a while for the poses dir)...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(args.parquet_path), "/vae_features.parquet")
        batch.put_file(str(args.vae_checkpoint), "/vae.pt")
        batch.put_directory(str(args.poses_dir), "/poses")

    print(f"Done. Contents of '{args.volume_name}':")
    for entry in vol.listdir("/"):
        print(f"  {entry.path}  ({entry.type.name})")


def cmd_upload_artifacts_only(args: argparse.Namespace) -> None:
    """Upload only the parquet + VAE checkpoint (skip the large poses dir)."""
    _validate_paths(args, require_poses=False)

    import modal

    vol = modal.Volume.from_name(args.volume_name, create_if_missing=True)
    print(f"Uploading parquet + VAE checkpoint to '{args.volume_name}'...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(args.parquet_path), "/vae_features.parquet")
        batch.put_file(str(args.vae_checkpoint), "/vae.pt")
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
        help="Upload only parquet + VAE checkpoint (skip the large poses dir).",
    )
    _shared_args(p_upload_only)
    p_upload_only.set_defaults(func=cmd_upload_artifacts_only)

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
