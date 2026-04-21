import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import "./App.css";
import type { ApiSearchResultRow, CanvasItem, ImageAsset } from "./types";
import { DND_MIME } from "./types";

const PAGE_SIZE = 8;
const API_RESULT_LIMIT = 40;
const DEFAULT_PLACE_WIDTH = 200;
const DEFAULT_PLACE_HEIGHT = 150;

const PREVIEW_PAD = 14;
const PREVIEW_MAX_W = 640;
const PREVIEW_MAX_H = 480;

type HoverPreviewState = { src: string; name: string; left: number; top: number };

function computePreviewPosition(clientX: number, clientY: number) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const boxW = Math.min(PREVIEW_MAX_W, vw - PREVIEW_PAD * 2);
  const boxH = Math.min(PREVIEW_MAX_H, vh - PREVIEW_PAD * 2);
  let left = clientX + PREVIEW_PAD;
  let top = clientY + PREVIEW_PAD;
  if (left + boxW > vw - PREVIEW_PAD) left = clientX - boxW - PREVIEW_PAD;
  if (top + boxH > vh - PREVIEW_PAD) top = clientY - boxH - PREVIEW_PAD;
  left = Math.max(PREVIEW_PAD, Math.min(left, vw - boxW - PREVIEW_PAD));
  top = Math.max(PREVIEW_PAD, Math.min(top, vh - boxH - PREVIEW_PAD));
  return { left, top };
}

function parseAssetPayload(raw: string): ImageAsset | null {
  if (!raw) return null;
  try {
    const data = JSON.parse(raw) as Partial<ImageAsset>;
    if (
      typeof data.id === "string" &&
      typeof data.src === "string" &&
      typeof data.name === "string"
    ) {
      return { id: data.id, src: data.src, name: data.name };
    }
  } catch {
    /* ignore */
  }
  return null;
}

function collectProtectedSrcs(saved: ImageAsset[], canvas: CanvasItem[]): Set<string> {
  const out = new Set<string>();
  saved.forEach((s) => out.add(s.src));
  canvas.forEach((c) => out.add(c.src));
  return out;
}

function revokeUnprotectedBatch(prev: ImageAsset[], saved: ImageAsset[], canvas: CanvasItem[]) {
  const keep = collectProtectedSrcs(saved, canvas);
  prev.forEach((a) => {
    if (!keep.has(a.src) && a.src.startsWith("blob:")) URL.revokeObjectURL(a.src);
  });
}

type Metric = "vae" | "squared";

function buildSearchUrl(metric: Metric): string {
  const base = import.meta.env.VITE_API_SEARCH_URL?.trim();
  if (!base) return "";
  const q = new URLSearchParams({
    offset: "0",
    limit: String(API_RESULT_LIMIT),
    include_images: "true",
    metric,
  });
  return base.includes("?") ? `${base}&${q}` : `${base}?${q}`;
}

function rowToImageAsset(row: ApiSearchResultRow): ImageAsset {
  let src = "";
  if (row.image_base64) src = `data:image/jpeg;base64,${row.image_base64}`;
  else {
    const origin = import.meta.env.VITE_RESULT_IMAGE_BASE?.trim();
    if (origin) src = `${origin.replace(/\/$/, "")}/${row.image_path.replace(/^\//, "")}`;
  }
  return { id: crypto.randomUUID(), name: row.image_path, src };
}

function httpDetail(data: unknown, fallback: string): string {
  if (!data || typeof data !== "object" || !("detail" in data)) return fallback.slice(0, 400);
  const d = (data as { detail: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join("; ");
  return String(d);
}

export default function App() {
  const [tab, setTab] = useState<"batch" | "mood">("batch");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [batch, setBatch] = useState<ImageAsset[]>([]);
  const [saved, setSaved] = useState<ImageAsset[]>([]);
  const [canvasItems, setCanvasItems] = useState<CanvasItem[]>([]);
  const [page, setPage] = useState(0);
  const [sidebarDragOver, setSidebarDragOver] = useState(false);
  const [canvasDragOver, setCanvasDragOver] = useState(false);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [queryPreviewUrl, setQueryPreviewUrl] = useState<string | null>(null);
  const [drawingDirty, setDrawingDirty] = useState(false);
  const [metric, setMetric] = useState<Metric>("vae");

  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const drawCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const drawStrokeRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false,
    lastX: 0,
    lastY: 0,
  });
  const stateRef = useRef({ batch, saved, canvasItems });
  stateRef.current = { batch, saved, canvasItems };
  const moveRef = useRef<{
    id: string;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  const pageCount = Math.max(1, Math.ceil(batch.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageSlice = useMemo(() => {
    const start = safePage * PAGE_SIZE;
    return batch.slice(start, start + PAGE_SIZE);
  }, [batch, safePage]);

  const apiSearchUrl = useMemo(() => buildSearchUrl(metric), [metric]);

  useEffect(() => {
    setPage((p) => Math.min(p, Math.max(0, pageCount - 1)));
  }, [pageCount]);

  useEffect(() => {
    setHoverPreview(null);
  }, [tab]);

  useEffect(() => {
    if (!queryFile) {
      setQueryPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(queryFile);
    setQueryPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [queryFile]);

  const primeDrawingCanvas = useCallback(() => {
    const c = drawCanvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    drawCtxRef.current = ctx;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  useEffect(() => {
    if (tab === "batch") primeDrawingCanvas();
  }, [tab, primeDrawingCanvas]);

  const canvasPointToPixel = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = drawCanvasRef.current;
    if (!c) return { x: 0, y: 0 };
    const rect = c.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * c.width;
    const y = ((e.clientY - rect.top) / rect.height) * c.height;
    return { x, y };
  };

  const onDrawPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const ctx = drawCtxRef.current;
    if (!ctx) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const { x, y } = canvasPointToPixel(e);
    drawStrokeRef.current = { active: true, lastX: x, lastY: y };
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    if (!drawingDirty) setDrawingDirty(true);
  };

  const onDrawPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const s = drawStrokeRef.current;
    if (!s.active) return;
    const ctx = drawCtxRef.current;
    if (!ctx) return;
    const { x, y } = canvasPointToPixel(e);
    ctx.beginPath();
    ctx.moveTo(s.lastX, s.lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    drawStrokeRef.current = { active: true, lastX: x, lastY: y };
  };

  const onDrawPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drawStrokeRef.current.active = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const clearDrawing = useCallback(() => {
    primeDrawingCanvas();
    setDrawingDirty(false);
  }, [primeDrawingCanvas]);

  const showHoverPreview = useCallback((src: string, name: string, e: React.MouseEvent) => {
    const { left, top } = computePreviewPosition(e.clientX, e.clientY);
    setHoverPreview({ src, name, left, top });
  }, []);

  const moveHoverPreview = useCallback((e: React.MouseEvent) => {
    if (moveRef.current) return;
    setHoverPreview((prev) => {
      if (!prev) return prev;
      const { left, top } = computePreviewPosition(e.clientX, e.clientY);
      return { ...prev, left, top };
    });
  }, []);

  const hideHoverPreview = useCallback(() => setHoverPreview(null), []);

  const onPickQueryFile = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file || !file.type.startsWith("image/")) return;
      setQueryFile(file);
      setApiError(null);
      setBatch((prev) => {
        revokeUnprotectedBatch(prev, saved, canvasItems);
        return [];
      });
      setPage(0);
    },
    [saved, canvasItems],
  );

  const searchWithFile = useCallback(
    async (file: File) => {
      if (!apiSearchUrl) return;
      setSearching(true);
      setApiError(null);
      try {
        const body = new FormData();
        body.append("file", file);
        const res = await fetch(apiSearchUrl, { method: "POST", body });
        const text = await res.text();
        let data: unknown = null;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          /* ignore */
        }
        if (!res.ok) throw new Error(httpDetail(data, text) || `HTTP ${res.status}`);
        const parsed = data as { results?: ApiSearchResultRow[] };
        const rows = Array.isArray(parsed.results) ? parsed.results : [];
        setBatch((prev) => {
          revokeUnprotectedBatch(prev, saved, canvasItems);
          return rows.map(rowToImageAsset);
        });
        setPage(0);
      } catch (e) {
        setApiError(e instanceof Error ? e.message : "Search failed");
      } finally {
        setSearching(false);
      }
    },
    [apiSearchUrl, saved, canvasItems],
  );

  const runSearch = useCallback(() => {
    if (queryFile) void searchWithFile(queryFile);
  }, [queryFile, searchWithFile]);

  const searchFromDrawing = useCallback(async () => {
    const c = drawCanvasRef.current;
    if (!c || !apiSearchUrl) return;
    const blob = await new Promise<Blob | null>((resolve) =>
      c.toBlob((b) => resolve(b), "image/png"),
    );
    if (!blob) {
      setApiError("Could not capture drawing.");
      return;
    }
    const file = new File([blob], "drawing.png", { type: "image/png" });
    await searchWithFile(file);
  }, [apiSearchUrl, searchWithFile]);

  useEffect(() => {
    return () => {
      const { batch: b, saved: s, canvasItems: c } = stateRef.current;
      const urls = new Set<string>();
      b.forEach((x) => urls.add(x.src));
      s.forEach((x) => urls.add(x.src));
      c.forEach((x) => urls.add(x.src));
      urls.forEach((url) => {
        if (url.startsWith("blob:")) URL.revokeObjectURL(url);
      });
    };
  }, []);

  const addToSaved = useCallback((asset: ImageAsset) => {
    setSaved((prev) => {
      if (prev.some((p) => p.id === asset.id)) return prev;
      return [...prev, asset];
    });
  }, []);

  const removeFromSaved = useCallback(
    (asset: ImageAsset) => {
      setSaved((prev) => prev.filter((p) => p.id !== asset.id));
      const stillUsed =
        canvasItems.some((c) => c.src === asset.src) ||
        batch.some((b) => b.src === asset.src);
      if (!stillUsed && asset.src.startsWith("blob:")) {
        URL.revokeObjectURL(asset.src);
      }
    },
    [batch, canvasItems],
  );

  const handleSidebarDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    setSidebarDragOver(true);
  };

  const handleSidebarDragLeave = () => setSidebarDragOver(false);

  const handleSidebarDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setSidebarDragOver(false);
    const asset = parseAssetPayload(e.dataTransfer.getData(DND_MIME));
    if (asset) addToSaved(asset);
  };

  const handleSavedDragStart = (asset: ImageAsset, e: React.DragEvent) => {
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setData(DND_MIME, JSON.stringify(asset));
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    dragOffsetRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const placeOnCanvas = useCallback((asset: ImageAsset, clientX: number, clientY: number) => {
    const el = canvasRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const { x: ox, y: oy } = dragOffsetRef.current;
    let x = clientX - rect.left - ox;
    let y = clientY - rect.top - oy;
    x = Math.max(0, Math.min(x, rect.width - DEFAULT_PLACE_WIDTH));
    y = Math.max(0, Math.min(y, rect.height - DEFAULT_PLACE_HEIGHT));
    setCanvasItems((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        src: asset.src,
        name: asset.name,
        x,
        y,
        width: DEFAULT_PLACE_WIDTH,
        height: DEFAULT_PLACE_HEIGHT,
      },
    ]);
  }, []);

  const handleCanvasDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    setCanvasDragOver(true);
  };

  const handleCanvasDragLeave = (e: React.DragEvent) => {
    if (e.currentTarget === e.target) setCanvasDragOver(false);
  };

  const handleCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setCanvasDragOver(false);
    const asset = parseAssetPayload(e.dataTransfer.getData(DND_MIME));
    if (asset) placeOnCanvas(asset, e.clientX, e.clientY);
  };

  const onCanvasItemPointerDown = (item: CanvasItem, e: React.PointerEvent) => {
    if (e.button !== 0) return;
    setHoverPreview(null);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    moveRef.current = {
      id: item.id,
      startX: e.clientX,
      startY: e.clientY,
      originX: item.x,
      originY: item.y,
    };
    setCanvasItems((prev) => {
      const idx = prev.findIndex((p) => p.id === item.id);
      if (idx === -1) return prev;
      const next = [...prev];
      const [picked] = next.splice(idx, 1);
      next.push(picked);
      return next;
    });
  };

  const onCanvasItemPointerMove = (e: React.PointerEvent) => {
    const ctx = moveRef.current;
    if (!ctx) return;
    const el = canvasRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setCanvasItems((prev) => {
      const item = prev.find((c) => c.id === ctx.id);
      if (!item) return prev;
      const dx = e.clientX - ctx.startX;
      const dy = e.clientY - ctx.startY;
      let nx = ctx.originX + dx;
      let ny = ctx.originY + dy;
      nx = Math.max(0, Math.min(nx, rect.width - item.width));
      ny = Math.max(0, Math.min(ny, rect.height - item.height));
      return prev.map((c) => (c.id === ctx.id ? { ...c, x: nx, y: ny } : c));
    });
  };

  const onCanvasItemPointerUp = (e: React.PointerEvent) => {
    if ((e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    }
    moveRef.current = null;
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="app-title">Poseboard</h1>
        <nav className="tab-row" aria-label="Primary">
          <button
            type="button"
            className={`tab ${tab === "batch" ? "tab-active" : ""}`}
            onClick={() => setTab("batch")}
          >
            Results
          </button>
          <button
            type="button"
            className={`tab ${tab === "mood" ? "tab-active" : ""}`}
            onClick={() => setTab("mood")}
          >
            Moodboard
          </button>
        </nav>
      </header>

      <main className="main-panel">
        {tab === "batch" ? (
          <>
            {!apiSearchUrl && (
              <p className="api-hint">
                Set <code>VITE_API_SEARCH_URL</code> to your deployed <code>API</code> search URL
                (POST multipart field <code>file</code>). For local dev use <code>/poseboard-api</code>{" "}
                plus <code>POSEBOARD_PROXY_TARGET</code> in <code>.env.development</code> (see{" "}
                <code>vite.config.ts</code>).
              </p>
            )}
            {apiError && (
              <div className="error-banner" role="alert">
                {apiError}
              </div>
            )}
            <div className="toolbar">
              <label className="file-btn">
                <span>Choose query image</span>
                <input
                  className="visually-hidden"
                  type="file"
                  accept="image/*"
                  onChange={(e) => onPickQueryFile(e.target.files)}
                />
              </label>
              <button
                type="button"
                className="search-btn"
                disabled={!queryFile || !apiSearchUrl || searching}
                onClick={() => void runSearch()}
              >
                {searching ? "Searching…" : "Search"}
              </button>
              <label className="metric-select" title="Similarity metric used by the API">
                <span className="metric-select-label">Metric</span>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as Metric)}
                  disabled={searching}
                >
                  <option value="vae">VAE</option>
                  <option value="squared">Squared</option>
                </select>
              </label>
              {queryFile && (
                <span className="meta">
                  {queryFile.name} ·{" "}
                  {batch.length ? `${batch.length} matches` : "not searched yet"}
                </span>
              )}
            </div>

            <div className="query-row">
              {queryPreviewUrl && (
                <figure className="query-preview">
                  <img src={queryPreviewUrl} alt="" />
                  <figcaption className="query-preview-caption" title={queryFile?.name}>
                    Query · {queryFile?.name}
                  </figcaption>
                </figure>
              )}
              <figure className="draw-pad">
                <canvas
                  ref={drawCanvasRef}
                  className="draw-canvas"
                  width={480}
                  height={360}
                  onPointerDown={onDrawPointerDown}
                  onPointerMove={onDrawPointerMove}
                  onPointerUp={onDrawPointerUp}
                  onPointerCancel={onDrawPointerUp}
                />
                <figcaption className="draw-caption">
                  <span>Draw a pose sketch</span>
                  <span className="draw-actions">
                    <button
                      type="button"
                      className="draw-btn"
                      disabled={!drawingDirty || searching}
                      onClick={clearDrawing}
                    >
                      Clear
                    </button>
                    <button
                      type="button"
                      className="search-btn"
                      disabled={!drawingDirty || !apiSearchUrl || searching}
                      onClick={() => void searchFromDrawing()}
                    >
                      {searching ? "Searching…" : "Search drawing"}
                    </button>
                  </span>
                </figcaption>
              </figure>
            </div>

            {batch.length === 0 ? (
              <div className="empty-panel">
                After Search, drag any result thumbnail into <strong>Saved Images</strong> on the
                right.
              </div>
            ) : (
              <>
                <div className="pagination pagination-above-grid">
                  <button
                    type="button"
                    disabled={safePage <= 0}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                  >
                    Prev
                  </button>
                  <span className="meta">
                    Page {safePage + 1} / {pageCount}
                  </span>
                  <button
                    type="button"
                    disabled={safePage >= pageCount - 1}
                    onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  >
                    Next
                  </button>
                </div>
                <div className="grid">
                {pageSlice.map((asset) => (
                  <article key={asset.id} className="card">
                    <div
                      className="card-drag"
                      draggable
                      onMouseEnter={(e) => showHoverPreview(asset.src, asset.name, e)}
                      onMouseMove={moveHoverPreview}
                      onMouseLeave={hideHoverPreview}
                      onDragStart={(e) => {
                        hideHoverPreview();
                        e.dataTransfer.effectAllowed = "copy";
                        e.dataTransfer.setData(DND_MIME, JSON.stringify(asset));
                        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                        dragOffsetRef.current = {
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top,
                        };
                      }}
                    >
                      <img src={asset.src} alt="" draggable={false} />
                    </div>
                    <div className="card-caption" title={asset.name}>
                      {asset.name}
                    </div>
                  </article>
                ))}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="mood-wrap">
            <div
              ref={canvasRef}
              className={`mood-canvas ${canvasDragOver ? "mood-drop-target" : ""}`}
              onDragOver={handleCanvasDragOver}
              onDragLeave={handleCanvasDragLeave}
              onDrop={handleCanvasDrop}
            >
              <span className="mood-label" aria-hidden>
                Moodboard
              </span>
              {canvasItems.map((item, index) => (
                <div
                  key={item.id}
                  className="canvas-item"
                  style={{
                    left: item.x,
                    top: item.y,
                    width: item.width,
                    height: item.height,
                    zIndex: 10 + index,
                  }}
                  onMouseEnter={(e) => showHoverPreview(item.src, item.name, e)}
                  onMouseMove={moveHoverPreview}
                  onMouseLeave={hideHoverPreview}
                  onPointerDown={(e) => onCanvasItemPointerDown(item, e)}
                  onPointerMove={onCanvasItemPointerMove}
                  onPointerUp={onCanvasItemPointerUp}
                  onPointerCancel={onCanvasItemPointerUp}
                >
                  <img src={item.src} alt={item.name} draggable={false} />
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <aside className="sidebar" aria-label="Saved images">
        <h2 className="sidebar-title">Saved Images</h2>
        <p className="sidebar-hint">
          Drag from the results grid here. On the Moodboard tab, drag these onto the canvas.
        </p>
        <div className="sidebar-scroll">
          <div
            className={`sidebar-drop ${sidebarDragOver ? "sidebar-drop-active" : ""}`}
            onDragOver={handleSidebarDragOver}
            onDragLeave={handleSidebarDragLeave}
            onDrop={handleSidebarDrop}
          >
            {saved.length === 0 ? (
              <div className="sidebar-empty">Drop images here to save them.</div>
            ) : (
              saved.map((asset) => (
                <div key={asset.id} className="sidebar-thumb">
                  <button
                    type="button"
                    className="sidebar-remove"
                    aria-label={`Remove ${asset.name}`}
                    onClick={() => removeFromSaved(asset)}
                  >
                    ×
                  </button>
                  <div
                    draggable
                    onMouseEnter={(e) => showHoverPreview(asset.src, asset.name, e)}
                    onMouseMove={moveHoverPreview}
                    onMouseLeave={hideHoverPreview}
                    onDragStart={(e) => {
                      hideHoverPreview();
                      handleSavedDragStart(asset, e);
                    }}
                  >
                    <img src={asset.src} alt="" draggable={false} />
                  </div>
                  <div className="sidebar-thumb-label" title={asset.name}>
                    {asset.name}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </aside>

      {hoverPreview &&
        createPortal(
          <div
            className="hover-preview"
            style={{ left: hoverPreview.left, top: hoverPreview.top }}
            role="img"
            aria-label={hoverPreview.name}
          >
            <div className="hover-preview-frame">
              <img src={hoverPreview.src} alt="" draggable={false} />
            </div>
            <div className="hover-preview-caption">{hoverPreview.name}</div>
          </div>,
          document.body,
        )}
    </div>
  );
}
