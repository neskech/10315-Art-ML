# Top-K Pose Retrieval API

A [Modal](https://modal.com/docs) deployment that exposes nearest-neighbour
pose retrieval as a single HTTP endpoint. Supports two metrics (mirrors
`visualization/visualize_retrieval.py`):

- `metric=vae` — cosine similarity over VAE latents (default).
- `metric=squared` — squared L2 over the 3D keypoints produced by the
  scripted MHR model.

Both metrics apply the same image-space deduplication that
`visualization/visualize_retrieval.py` uses (letterboxed multiscale RMS L2
+ optional `matchTemplate` NCC for crop detection) — including against the
query image itself, in case the query is a row in the dataset.

## How it works

The pose database is represented server-side as two parquet files on the
Modal Volume:

- `vae_features.parquet` — precomputed VAE latents. Columns:
  `image_path`, `vae_features`. Produced offline by
  `data_generation/write_vae_features.py`.
- `processed_poses.parquet` — MHR parameters per image. Columns include
  `image_path`, `mhr_parameters`. Produced offline by
  `data_generation/write_poses.py`. **Optional**; upload it if you want
  to serve `metric=squared`. At container start the server runs the
  scripted MHR model over every row once and caches the `(N, J, 3)` 3D
  keypoint tensor.

The server never re-runs SAM3D or the VAE encoder on dataset rows; it only
runs them on the query image.

Pipeline per request:

1. The user POSTs an image (multipart upload) with `metric=vae|squared`.
2. The container runs SAM 3D Body to extract MHR parameters from the image.
3. Depending on the metric:
   - `vae`: MHR parameters → joint angles → `FeedForwardVAE` encoder → unit
     latent on the hypersphere → cosine similarity against every row of the
     precomputed `(N, D)` latent matrix.
   - `squared`: MHR parameters → scripted MHR model → `(J, 3)` 3D keypoints
     → sum-of-squares distance against every row of the cached
     `(N, J, 3)` keypoint tensor.
4. The top `(offset + limit) * overfetch_factor` candidates are pulled back,
   each candidate image is loaded from the volume, and the pair
   `(query, candidate)` + `(kept_i, candidate)` is checked for visual
   duplication using letterboxed multiscale RMS L2 and (optionally) NCC
   template matching. Duplicates are dropped.
5. The deduped list is sliced to `[offset:offset+limit]` and the
   corresponding pose images are returned as base64-encoded JPEGs.

Pagination still works the same way: `offset=0, limit=10` is the top 10
**deduped** poses; `offset=10, limit=10` is ranks 10..19 of the deduped
list.

## Layout

```
API/
├── main.py     # Modal app: image, Volume mount, VAERetrieval class, /search endpoint
├── serve.py    # CLI wrapper: upload artifacts, modal serve / deploy / run
├── test.py     # Local client that plots the results
└── README.md
```

## Prerequisites

- The Modal client installed in your local environment:

  ```bash
  uv pip install modal
  # or: pip install modal
  ```

- A Modal account and `modal token new` already run on this machine.
- The standard project artifacts present locally:
    - `data/vae_features_<checkpoint>.parquet`  (produced by
      `data_generation/write_vae_features.py`; columns: `image_path`,
      `vae_features`)
    - `data/processed_poses.parquet`            (produced by
      `data_generation/write_poses.py`; required only if you want the
      `metric=squared` retrieval path)
    - `data/poses/`                             (image folder whose relative
      paths match the parquet's `image_path` column; needed for returning
      base64 images **and** for image-space deduplication)
    - `checkpoints/<vae>.pt`                    (trained `FeedForwardVAE`
      checkpoint — used to encode query images; must be the same VAE that
      produced the parquet)
    - `checkpoints/sam3d/dinov3/model.ckpt`     (baked into the image)
    - `checkpoints/sam3d/dinov3/assets/mhr_model.pt` (baked into the image)

## 1. Upload data to a Modal Volume

The VAE-features parquet, (optional) processed-poses parquet, VAE checkpoint
and poses directory are pushed once to a Modal `Volume` (default name:
`vae-retrieval-data`). The container mounts the volume at `/vol`. Remote
paths:

- `/vol/vae_features.parquet`      — precomputed VAE latents (required)
- `/vol/processed_poses.parquet`   — MHR parameters per image (required for
                                     `metric=squared`, otherwise optional)
- `/vol/vae.pt`                    — VAE checkpoint (for encoding queries)
- `/vol/poses/...`                 — pose images (for base64 responses and
                                     image-space deduplication)

```bash
python API/serve.py upload \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --processed-poses-parquet data/processed_poses.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --poses-dir data/poses
```

`--processed-poses-parquet` is optional — if you don't pass it (or the file
doesn't exist) the server starts fine for VAE retrieval but the
`metric=squared` path will return HTTP 422 until you upload it.

If you only changed the parquet(s)/checkpoint and want to skip re-uploading
the 1+ GB poses directory:

```bash
python API/serve.py upload-artifacts \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --processed-poses-parquet data/processed_poses.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt
```

To push just the processed-poses parquet (e.g. after regenerating it):

```bash
python API/serve.py upload-processed-poses \
    --processed-poses-parquet data/processed_poses.parquet
```

## 2. Run the API

Live-reload, ephemeral URL (great for development):

```bash
python API/serve.py serve \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt
```

Persistent deployment:

```bash
python API/serve.py deploy \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt
```

The `--parquet-path`, `--processed-poses-parquet`, `--vae-checkpoint` and
`--poses-dir` flags are accepted by every subcommand so the same incantation
works for `upload`, `serve`, `deploy` and `run`. Note that
`serve` / `deploy` / `run` do **not** upload anything — they only configure
env vars for the Modal app. To push a new parquet or checkpoint you must
run `upload` / `upload-artifacts` / `upload-processed-poses` first.

## 3. Hit the endpoint

`modal serve` / `modal deploy` will print a URL ending in `/search`. POST a
multipart form with the image file:

```bash
curl -X POST \
    -F "file=@data/query/sit.jpg" \
    "https://<your-workspace>--vae-topk-retrieval-vaeretrieval-search.modal.run?offset=0&limit=10&metric=vae"
```

Supported query params:

| Param                   | Default | Description                                                         |
| ----------------------- | ------- | ------------------------------------------------------------------- |
| `offset`                | 0       | Number of deduped results to skip.                                  |
| `limit`                 | 10      | Number of deduped results to return (capped at 200).                |
| `metric`                | `vae`   | `vae` (cosine on latents) or `squared` (3D keypoints).              |
| `include_images`        | true    | Base64-encode each returned pose image.                             |
| `ignore_query_cache`    | false   | Bypass the query-embedding LRU cache.                               |
| `dedupe_epsilon`        | 0.05    | Multiscale RMS L2 threshold; 0 disables RMS dedup.                  |
| `dedupe_ncc_threshold`  | 0.88    | `matchTemplate` NCC threshold; 0 disables crop-aware dedup.         |
| `overfetch_factor`      | 5       | When dedup is on, fetch `(offset+limit)*factor` before filtering.   |

Response shape:

```json
{
  "metric": "vae",
  "offset": 0,
  "limit": 10,
  "total": 29625,
  "latent_dim": 128,
  "keypoints_shape": [127, 3],
  "results": [
    {
      "rank": 0,
      "image_path": "downloaded_pins/gesture/676665912791710178.png",
      "cosine_similarity": 0.987,
      "distance": 0.013,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "query_cache": { "hit": false, "size": 1, "max": 1024 },
  "dedup": {
    "enabled": true,
    "epsilon": 0.05,
    "ncc_threshold": 0.88,
    "overfetch_factor": 5,
    "candidates_considered": 50,
    "unique_before_window": 10,
    "missing_during_dedup": 0,
    "total": 29625
  },
  "image_status": {
    "missing_image_count": 0,
    "missing_image_examples": [],
    "volume_reloaded": false
  }
}
```

`squared` responses put the squared-L2 distance in `distance` and set
`cosine_similarity` to `null`.

Pass `include_images=false` to skip the base64 payloads (much smaller
responses) and just get the ranked image paths + scores.

For pagination beyond top-K, just shift the offset:

```bash
# top 10 deduped
curl ... "?offset=0&limit=10"

# ranks 10..19 deduped
curl ... "?offset=10&limit=10"
```

The dedup step runs before the `[offset:offset+limit]` slice, so
consecutive pages are consistent with each other as long as the
`dedupe_*` / `overfetch_factor` params stay the same.

## 4. Smoke test from the command line

```bash
python API/serve.py run \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --image data/query/sit.jpg --limit 5 \
    --metric vae --dedupe-epsilon 0.05 --dedupe-ncc-threshold 0.88
```

This calls `VAERetrieval.search_bytes` via `.remote()` and prints the JSON
result (with base64 payloads stripped for readability). Swap
`--metric vae` for `--metric squared` to exercise the keypoint path.

## Notes / gotchas

- **Container startup.** VAE retrieval costs `read_parquet` +
  `torch.from_numpy` — well under a second once model weights are loaded.
  If you upload `processed_poses.parquet`, the `@modal.enter` lifecycle
  additionally batch-runs the scripted MHR model over all ~30k rows to
  cache 3D keypoints; this is a few seconds on a T4 and one-off per
  container.
- The Modal image bakes the SAM3D + MHR weights as a layer (~2.6 GB) so cold
  starts only need to download the small VAE checkpoint and parquet from the
  volume.
- `detectron2` is installed because `pose_module.sam3d.tools.build_detector.HumanDetector`
  imports it eagerly in `__init__`. We still call `predict(..., use_bbox_detector=False)`
  so no detection actually runs — just like `topKRetrieval/topKRetrieval.py`.
- The container default GPU is `T4`. Override with `--gpu L4` (or `A10G`,
  `A100`) if you want faster cold starts / inference.
- **Warm replicas:** defaults are `min_containers=1` and `scaledown_window=3600`
  (1 hour, Modal’s maximum), so one GPU worker stays up and idle workers are
  not torn down immediately — this avoids re-running SAM3D + ViTDet init on
  every cold start.
  For dev cost savings use `--min-containers 0` (and optionally a shorter
  `--scaledown-window`). You can also set `VAE_API_MIN_CONTAINERS` and
  `VAE_API_SCALEDOWN_WINDOW_SEC` in the environment.
- The VAE checkpoint you upload **must be the same checkpoint** that produced
  the parquet — otherwise query latents and dataset latents live in different
  spaces and similarities are meaningless. The parquet filename convention
  (`vae_features_<checkpoint-stem>.parquet`) makes this explicit.
