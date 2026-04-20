#!/usr/bin/env python3
"""Call the VAE retrieval Modal API and show results with matplotlib (like
``visualization/visualize_retrieval.py``).

Example:

    uv run python API/test.py \\
        --url 'https://<workspace>--vae-topk-retrieval-vaeretrieval-search-dev.modal.run/' \\
        --image data/query/sit.jpg \\
        --offset 0 \\
        --limit 10

Optional: also write a figure with ``--save path/to/out.png``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import platform

import cv2
import matplotlib


def _pick_default_backend() -> str:
    """Pick an interactive backend that works out of the box on this platform.

    Homebrew Python on macOS often ships without tkinter, which makes the usual
    ``TkAgg`` default crash with ``ModuleNotFoundError: No module named
    '_tkinter'``. But macOS Python does ship the Cocoa-based ``macosx``
    backend, so prefer that on Darwin and fall back to ``TkAgg`` elsewhere.
    Agg is only used as a last resort (headless).
    """
    if os.environ.get("MPLBACKEND"):
        return os.environ["MPLBACKEND"]

    candidates: list[str] = []
    if platform.system() == "Darwin":
        candidates += ["macosx", "QtAgg", "TkAgg"]
    else:
        candidates += ["QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg"]
    candidates.append("Agg")

    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:
            continue
    return matplotlib.get_backend()


_SELECTED_BACKEND = _pick_default_backend()

import numpy as np

def _matplotlib_backend_is_interactive() -> bool:
    """True if ``plt.show()`` can open a window without warnings (not Agg/svg/...)."""
    import matplotlib.pyplot as plt

    try:
        from matplotlib.backends import backend_registry
        import matplotlib.backends

        current = plt.get_backend().lower()
        interactive = {
            b.lower()
            for b in backend_registry.list_builtin(
                matplotlib.backends.BackendFilter.INTERACTIVE
            )
        }
        return current in interactive
    except Exception:
        from matplotlib import rcsetup

        current = plt.get_backend().lower()
        return current in {b.lower() for b in rcsetup.interactive_bk}


_MATPLOTLIB_SAVE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".svg",
        ".eps",
        ".tif",
        ".tiff",
        ".webp",
        ".pgf",
        ".ps",
        ".raw",
        ".rgba",
        ".svgz",
    }
)


def _resolve_matplotlib_output_path(path: Path) -> Path:
    """Matplotlib cannot write HTML and other arbitrary extensions; use .png."""
    ext = path.suffix.lower()
    if ext in _MATPLOTLIB_SAVE_SUFFIXES:
        return path
    return path.with_suffix(".png")


def _load_rgb_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_base64_rgb(b64: str) -> np.ndarray | None:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _build_multipart_body(
    file_path: Path, boundary: str
) -> tuple[bytes, str]:
    mime, _ = mimetypes.guess_type(str(file_path))
    content_type = mime or "application/octet-stream"
    raw = file_path.read_bytes()
    filename = file_path.name
    b = boundary.encode()
    parts: list[bytes] = [
        b"--" + b + b"\r\n",
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode(),
        raw,
        b"\r\n--" + b + b"--\r\n",
    ]
    body = b"".join(parts)
    full_ct = f"multipart/form-data; boundary={boundary}"
    return body, full_ct


def _request_json(url: str, image_path: Path, params: dict[str, str | int | bool]) -> dict:
    parsed = urllib.parse.urlparse(url)
    q_values = {
        k: ("true" if v is True else "false" if v is False else v) for k, v in params.items()
    }
    q = urllib.parse.urlencode({k: str(v) for k, v in q_values.items()})
    full_url = urllib.parse.urlunparse(parsed._replace(query=q))
    boundary = f"boundary{uuid.uuid4().hex}"
    body, ct = _build_multipart_body(image_path, boundary)
    req = urllib.request.Request(
        full_url,
        data=body,
        method="POST",
        headers={"Content-Type": ct},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed: {e}") from e
    return json.loads(payload)


def render_results_matplotlib(
    data: dict,
    query_image_path: Path,
    *,
    save_path: Path | None = None,
) -> None:
    """Layout aligned with ``visualization/visualize_retrieval.render_results_table``."""
    import matplotlib.pyplot as plt

    results = data.get("results") or []
    k = len(results)
    cols = 3
    rows = (k // cols) + (1 if k % cols != 0 else 0)

    fig = plt.figure(figsize=(15, 5 * (rows + 1)))
    gs = fig.add_gridspec(rows + 1, cols)

    cache_note = ""
    qc = data.get("query_cache")
    if isinstance(qc, dict):
        cache_note = f"  |  query_cache hit={qc.get('hit')}"

    fig.suptitle(
        f"Top-{k} Pose Retrieval (API)\n"
        f"offset={data.get('offset')} limit={data.get('limit')} "
        f"total={data.get('total')}{cache_note}",
        fontsize=20,
        fontweight="bold",
    )

    query_img = _load_rgb_image(query_image_path)
    if query_img is None:
        raise FileNotFoundError(f"Could not load query image at {query_image_path}")
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_img)
    ax_query.set_title("QUERY", fontsize=20, fontweight="bold", pad=15)
    ax_query.axis("off")

    for i, r in enumerate(results):
        r_grid = (i // cols) + 1
        c = i % cols
        ax = fig.add_subplot(gs[r_grid, c])

        b64 = r.get("image_base64")
        res_img = _decode_base64_rgb(b64) if b64 else None
        dist = r.get("distance")
        rank = r.get("rank", i)

        if res_img is not None:
            ax.imshow(res_img)
            title = f"Rank {rank + 1}\nDist: {float(dist):.4f}" if dist is not None else f"Rank {rank + 1}"
            ax.set_title(title, fontsize=12)
        else:
            ax.text(
                0.5,
                0.5,
                "No image\n(use --include-images)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Rank {rank + 1}", fontsize=12)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        resolved = _resolve_matplotlib_output_path(save_path)
        if resolved != save_path:
            print(
                f"Note: matplotlib cannot save {save_path.suffix!r}; "
                f"writing {resolved} instead.",
                file=sys.stderr,
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(resolved, bbox_inches="tight", dpi=150)
        print(f"Also saved figure to: {resolved.resolve()}")

    if _matplotlib_backend_is_interactive():
        plt.show()
    elif save_path is None:
        print(
            "Headless matplotlib (e.g. Agg): no display. "
            "Use --save out.png to write the figure, or MPLBACKEND=TkAgg if you have tkinter.",
            file=sys.stderr,
        )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POST a query image to the retrieval API and display results with matplotlib."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of the Modal fastapi_endpoint (POST to root), e.g. "
        "https://user--app-class-method-dev.modal.run/",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the query image file (multipart field name: file).",
    )
    parser.add_argument("--offset", type=int, default=0, help="Result offset (default 0).")
    parser.add_argument("--limit", type=int, default=10, help="Number of results (default 10).")
    parser.add_argument(
        "--include-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request base64 result images (default: true; required for result thumbnails).",
    )
    parser.add_argument(
        "--ignore-query-cache",
        action="store_true",
        help="Ask the API to bypass the query latent cache.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: also save the figure (png/pdf/svg/...). Unsupported extensions "
        "are replaced with .png.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    url = args.url.rstrip("/") + "/"
    params: dict[str, str | int | bool] = {
        "offset": args.offset,
        "limit": args.limit,
        "include_images": args.include_images,
    }
    if args.ignore_query_cache:
        params["ignore_query_cache"] = True

    data = _request_json(url, args.image, params)

    status = data.get("image_status") or {}
    missing_count = int(status.get("missing_image_count") or 0)
    if missing_count:
        examples = status.get("missing_image_examples") or []
        print(
            f"Warning: {missing_count} result image(s) could not be loaded from "
            f"the Modal volume (volume_reloaded={status.get('volume_reloaded')}).",
            file=sys.stderr,
        )
        for p in examples:
            print(f"  missing: {p}", file=sys.stderr)
        print(
            "These files are referenced by processed_poses.parquet but not present "
            "at /poses on the vae-retrieval-data volume. Upload the corresponding "
            "subdirectory with `modal volume put vae-retrieval-data <local> /poses/<remote>`.",
            file=sys.stderr,
        )

    render_results_matplotlib(data, args.image, save_path=args.save)


if __name__ == "__main__":
    main()
