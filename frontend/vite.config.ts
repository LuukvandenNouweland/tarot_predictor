import { defineConfig } from "vite";

// Dev-modus: `npm run dev` op :5173, video en herkenningen komen van de backend op :8000.
const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  // three.js is ~600 kB; voor een lokale app geen probleem.
  build: { chunkSizeWarningLimit: 800 },
  server: {
    proxy: {
      "/video": BACKEND,
      "/status": BACKEND,
      "/ws": { target: BACKEND, ws: true },
    },
  },
});
