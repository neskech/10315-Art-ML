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

/** One element of the `results` array from `API/main.py` `search`. */
export type ApiSearchResultRow = {
  rank: number;
  image_path: string;
  cosine_similarity: number;
  distance: number;
  image_base64?: string | null;
};

