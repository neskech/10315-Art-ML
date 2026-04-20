# VAE Top-K Pose Retrieval API

A [Modal](https://modal.com/docs) deployment that exposes nearest-neighbour
pose retrieval over the VAE latent space as a single HTTP endpoint.

## How it works

The entire pose database is represented server-side as a single parquet of
**precomputed** VAE latents (`image_path`, `vae_features` columns, produced
offline by `data_generation/write_vae_features.py`). The server never
re-runs SAM3D or the VAE encoder on dataset rows; it only runs them on the
query image.

Pipeline per request:

1. The user POSTs an image (multipart upload).
2. The container runs SAM 3D Body to extract MHR parameters from the image.
3. The MHR parameters are converted to joint angles and encoded by the trained
   `FeedForwardVAE` (see `vae_features/model/feedForwardVae.py`) to produce a
   unit latent vector on the hypersphere.
4. The query latent is compared (cosine similarity) against the precomputed
   (N, D) latent matrix loaded at container start from
   `vae_features_<checkpoint>.parquet`.
5. The top `offset + limit` rows are selected, sliced to `[offset:offset+limit]`,
   and the corresponding pose images are returned as base64-encoded JPEGs.

Because both the dataset latents and the query latent live on the unit
hypersphere, the precomputed matmul gives true top-K retrieval. With
`offset=0, limit=10` the response is the closest 10 poses; with
`offset=10, limit=10` the response is the next 10 (i.e. ranks 10..19).

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
    - `data/poses/`                             (image folder whose relative
      paths match the parquet's `image_path` column; only needed if you want
      the API to embed base64 images in the response)
    - `checkpoints/<vae>.pt`                    (trained `FeedForwardVAE`
      checkpoint — used to encode query images; must be the same VAE that
      produced the parquet)
    - `checkpoints/sam3d/dinov3/model.ckpt`     (baked into the image)
    - `checkpoints/sam3d/dinov3/assets/mhr_model.pt` (baked into the image)

## 1. Upload data to a Modal Volume

The VAE-features parquet, VAE checkpoint and poses directory are pushed once
to a Modal `Volume` (default name: `vae-retrieval-data`). The container
mounts the volume at `/vol`. Remote paths:

- `/vol/vae_features.parquet` — precomputed latents
- `/vol/vae.pt`               — VAE checkpoint (for encoding queries)
- `/vol/poses/...`            — pose images (for base64 responses)

```bash
python API/serve.py upload \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --poses-dir data/poses
```

If you only changed the parquet/checkpoint and want to skip re-uploading the
1+ GB poses directory:

```bash
python API/serve.py upload-artifacts \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt
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

The `--parquet-path`, `--vae-checkpoint` and `--poses-dir` flags are accepted
by every subcommand so the same incantation works for `upload`, `serve`,
`deploy` and `run`. Note that `serve` / `deploy` / `run` do **not** upload
anything — they only configure env vars for the Modal app. To push a new
parquet or checkpoint you must run `upload` / `upload-artifacts` first.

## 3. Hit the endpoint

`modal serve` / `modal deploy` will print a URL ending in `/search`. POST a
multipart form with the image file:

```bash
curl -X POST \
    -F "file=@data/query/sit.jpg" \
    "https://<your-workspace>--vae-topk-retrieval-vaeretrieval-search.modal.run?offset=0&limit=10"
```

Response shape:

```json
{
  "offset": 0,
  "limit": 10,
  "total": 29625,
  "latent_dim": 128,
  "results": [
    {
      "rank": 0,
      "image_path": "downloaded_pins/gesture/676665912791710178.png",
      "cosine_similarity": 0.987,
      "distance": 0.013,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "image_status": {
    "missing_image_count": 0,
    "missing_image_examples": [],
    "volume_reloaded": false
  }
}
```

Pass `include_images=false` to skip the base64 payloads (much smaller
responses) and just get the ranked image paths + scores.

For pagination beyond top-K, just shift the offset:

```bash
# top 10
curl ... "?offset=0&limit=10"

# ranks 10..19
curl ... "?offset=10&limit=10"
```

The endpoint internally computes `top(offset + limit)` and slices the
`[offset:offset+limit]` window, matching the spec exactly.

## 4. Smoke test from the command line

```bash
python API/serve.py run \
    --parquet-path data/vae_features_mhr_vae_latest.parquet \
    --vae-checkpoint checkpoints/mhr_vae_latest.pt \
    --image data/query/sit.jpg --limit 5
```

This calls `VAERetrieval.search_bytes` via `.remote()` and prints the JSON
result (with base64 payloads stripped for readability).

## Notes / gotchas

- **Container startup is fast now.** Since the dataset latents are read
  straight from the parquet, the `@modal.enter` lifecycle is
  `read_parquet` + `torch.from_numpy` — no dataset iteration, no batched
  VAE encode. Expect well under a second after the model weights are loaded.
- The Modal image bakes the SAM3D + MHR weights as a layer (~2.6 GB) so cold
  starts only need to download the small VAE checkpoint and parquet from the
  volume.
- `detectron2` is installed because `pose_module.sam3d.tools.build_detector.HumanDetector`
  imports it eagerly in `__init__`. We still call `predict(..., use_bbox_detector=False)`
  so no detection actually runs — just like `topKRetrieval/topKRetrieval.py`.
- The container default GPU is `T4`. Override with `--gpu L4` (or `A10G`,
  `A100`) if you want faster cold starts / inference.
- **Warm replicas:** defaults are `min_containers=1` and `scaledown_window=7200`
  (2 hours), so one GPU worker stays up and idle workers are not torn down
  immediately — this avoids re-running SAM3D + ViTDet init on every cold start.
  For dev cost savings use `--min-containers 0` (and optionally a shorter
  `--scaledown-window`). You can also set `VAE_API_MIN_CONTAINERS` and
  `VAE_API_SCALEDOWN_WINDOW_SEC` in the environment.
- The VAE checkpoint you upload **must be the same checkpoint** that produced
  the parquet — otherwise query latents and dataset latents live in different
  spaces and similarities are meaningless. The parquet filename convention
  (`vae_features_<checkpoint-stem>.parquet`) makes this explicit.
