import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// THE /api PROXY, AND WHY IT IS HERE RATHER THAN A BASE URL IN THE CLIENT
// ----------------------------------------------------------------------
// The API returns RELATIVE urls inside its payloads -- `image.url` on an
// analysis is "/api/tablets/{id}/image", not an absolute address. That is the
// honest shape: the server does not know what host a browser reached it on.
// Proxying /api here means those urls work verbatim, so the client never has to
// rewrite a url the server gave it. A rewriting client is a client that can
// rewrite one wrongly.
//
// CORS in `cuneiform/api/main.py` allows http://localhost:5173 and
// http://127.0.0.1:5173 anyway, so a direct cross-origin call also works. The
// proxy is kept because it is the configuration a single-origin deployment
// would have, and because it keeps VITE_API_BASE empty in the common case.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true, // fail loudly rather than silently moving to 5174
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        // Domain 3 can take ~53 s, and the first call after idle adds ~60 s of
        // VRAM load. A proxy that gives up before the server does would show
        // the user a network error for a request that is going to succeed.
        timeout: 300000,
        proxyTimeout: 300000,
      },
    },
  },
});
