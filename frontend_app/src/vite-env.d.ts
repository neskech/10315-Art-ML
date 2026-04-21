/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_SEARCH_URL?: string;
  /** If the API sets `include_images=false`, optional base URL for result `image_path`. */
  readonly VITE_RESULT_IMAGE_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
