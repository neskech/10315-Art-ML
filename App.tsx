import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import "./App.css";
import type { CanvasItem, ImageAsset } from "./types";
import { DND_MIME } from "./types";

const PAGE_SIZE = 8;
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

export default function App() {
  const [tab, setTab] = useState<"batch" | "mood">("batch");
  const [batch, setBatch] = useState<ImageAsset[]>([]);
  const [saved, setSaved] = useState<ImageAsset[]>([]);
  const [canvasItems, setCanvasItems] = useState<CanvasItem[]>([]);
  const [page, setPage] = useState(0);
  const [sidebarDragOver, setSidebarDragOver] = useState(false);
  const [canvasDragOver, setCanvasDragOver] = useState(false);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);

  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLDivElement | null>(null);
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

  useEffect(() => {
    setPage((p) => Math.min(p, Math.max(0, pageCount - 1)));
  }, [pageCount]);

  useEffect(() => {
    setHoverPreview(null);
  }, [tab]);

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

  const replaceBatch = useCallback(
    (files: FileList | null) => {
      if (!files?.length) return;
      const protectedSrcs = collectProtectedSrcs(saved, canvasItems);
      setBatch((prev) => {
        prev.forEach((asset) => {
          if (!protectedSrcs.has(asset.src)) {
            URL.revokeObjectURL(asset.src);
          }
        });
        return Array.from(files).map((file) => ({
          id: crypto.randomUUID(),
          src: URL.createObjectURL(file),
          name: file.name,
        }));
      });
      setPage(0);
    },
    [saved, canvasItems],
  );

  useEffect(() => {
    return () => {
      const { batch: b, saved: s, canvasItems: c } = stateRef.current;
      const urls = new Set<string>();
      b.forEach((x) => urls.add(x.src));
      s.forEach((x) => urls.add(x.src));
      c.forEach((x) => urls.add(x.src));
      urls.forEach((url) => URL.revokeObjectURL(url));
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
      if (!stillUsed) {
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
            Image batch
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
            <div className="toolbar">
              <label className="file-btn">
                <span>Choose images</span>
                <input
                  className="visually-hidden"
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => replaceBatch(e.target.files)}
                />
              </label>
              <span className="meta">
                {batch.length
                  ? `${batch.length} image${batch.length === 1 ? "" : "s"} loaded`
                  : "No images yet — pick a batch to begin."}
              </span>
              <div className="pagination" hidden={batch.length === 0}>
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
            </div>

            {batch.length === 0 ? (
              <div className="empty-panel">
                Load a batch of images, then drag any thumbnail into{" "}
                <strong>Saved Images</strong> on the right.
              </div>
            ) : (
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
          Drag from the batch grid here. On the Moodboard tab, drag these onto the canvas.
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
