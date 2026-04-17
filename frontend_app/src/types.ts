export type ImageAsset = {
  id: string;
  src: string;
  name: string;
};

export type CanvasItem = {
  id: string;
  src: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export const DND_MIME = "application/x-poseboard-asset";
