import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on :8000 by default (python -m src.serving.app).
//
// Every API prefix the app uses is proxied.  `/api` is rewritten (the client
// prefixes its calls with it); the others are served verbatim, including
// `/media/video/*` — without that entry Vite answers video requests with the
// SPA fallback HTML and the browser reports a demuxer error.
const API_PREFIXES = [
  "/api",
  "/health",
  "/stats",
  "/system",
  "/models",
  "/evaluation",
  "/media",
  "/feed",
  "/items",
  "/users",
  "/cold",
  "/recommend",
];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((prefix) => [
        prefix,
        {
          target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
          changeOrigin: true,
          ...(prefix === "/api" ? { rewrite: (p: string) => p.replace(/^\/api/, "") } : {}),
        },
      ]),
    ),
  },
});
