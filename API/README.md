# VAE Top-K Pose Retrieval API

A [Modal](https://modal.com/docs) deployment that exposes nearest-neighbour
pose retrieval over the VAE latent space as a single HTTP endpoint.

Pipeline per request:

1. The user POSTs an image (multipart upload).
2. The container runs SAM 3D Body to extract MHR parameters from the image.
3. The MHR parameters are converted to joint angles and encoded by the trained
   `FeedForwardVAE` (see `vae_features/model/feedForwardVae.py`) to produce a
   unit latent vector on the hypersphere.
4. The query latent is compared (cosine similarity) against a precomputed
   matrix of latents for every row in `data/processed_poses.parquet`.
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
    - `data/processed_poses.parquet`            (path is the required CLI arg)
    - `data/poses/`                             (image folder referenced by the parquet)
    - `vae_features/train/checkpoints/<vae>.pt` (trained `FeedForwardVAE` checkpoint)
    - `checkpoints/sam3d/dinov3/model.ckpt`     (baked into the image)
    - `checkpoints/sam3d/dinov3/assets/mhr_model.pt` (baked into the image)

## 1. Upload data to a Modal Volume

The parquet, VAE checkpoint and poses directory are pushed once to a Modal
`Volume` (default name: `vae-retrieval-data`). The container mounts the volume
at `/vol`.

```bash
python API/serve.py upload \
    --parquet-path data/processed_poses.parquet \
    --vae-checkpoint vae_features/train/checkpoints/mhr_vae_best.pt \
    --poses-dir data/poses
```

If you only changed the parquet/checkpoint and want to skip re-uploading the
1+ GB poses directory:

```bash
python API/serve.py upload-artifacts \
    --parquet-path /some/other/processed_poses.parquet \
    --vae-checkpoint /some/other/vae.pt
```

## 2. Run the API

Live-reload, ephemeral URL (great for development):

```bash
python API/serve.py serve \
    --parquet-path data/processed_poses.parquet \
    --vae-checkpoint vae_features/train/checkpoints/mhr_vae_best.pt
```

Persistent deployment:

```bash
python API/serve.py deploy \
    --parquet-path data/processed_poses.parquet \
    --vae-checkpoint vae_features/train/checkpoints/mhr_vae_best.pt
```

The `--parquet-path`, `--vae-checkpoint` and `--poses-dir` flags are accepted
by every subcommand so the same incantation works for `upload`, `serve`,
`deploy` and `run`. They're forwarded to `API/main.py` via environment
variables (`VAE_API_VOLUME_NAME`, `VAE_API_APP_NAME`, `VAE_API_GPU`).

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
  "total": 3472,
  "latent_dim": 128,
  "results": [
    {
      "rank": 0,
      "image_path": "downloaded_pins/gesture/676665912791710178.png",
      "cosine_similarity": 0.987,
      "distance": 0.013,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    ...
  ]
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
    --parquet-path data/processed_poses.parquet \
    --vae-checkpoint vae_features/train/checkpoints/mhr_vae_best.pt \
    --image data/query/sit.jpg --limit 5
```

This calls `VAERetrieval.search_bytes` via `.remote()` and prints the JSON
result (with base64 payloads stripped for readability).

## Notes / gotchas

- The Modal image bakes the SAM3D + MHR weights as a layer (~2.6 GB) so cold
  starts only need to download the small VAE checkpoint and parquet from the
  volume.
- `detectron2` is installed because `pose_module.sam3d.tools.build_detector.HumanDetector`
  imports it eagerly in `__init__`. We still call `predict(..., use_bbox_detector=False)`
  so no detection actually runs — just like `topKRetrieval/topKRetrieval.py`.
- The container default GPU is `T4`. Override with `--gpu L4` (or `A10G`,
  `A100`) if you want faster cold starts / inference.
- `min_containers=0` keeps idle cost at zero. Increase it (e.g. `min_containers=1`
  in `main.py`) for lower latency at the cost of a hot replica.
