import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev: set `POSEBOARD_PROXY_TARGET=https://…modal.run` and `VITE_API_SEARCH_URL=/poseboard-api` in `.env.development` (API has no CORS).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const proxyTarget = env.POSEBOARD_PROXY_TARGET?.trim();

  return {
    plugins: [react()],
    server:
      mode === "development" && proxyTarget
        ? {
            proxy: {
              "/poseboard-api": {
                target: proxyTarget,
                changeOrigin: true,
                secure: true,
                rewrite: (path) => path.replace(/^\/poseboard-api/, "") || "/",
              },
            },
          }
        : undefined,
  };
});
