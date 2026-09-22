import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

// Phones refuse getUserMedia outside a secure context, so the dev server is served over
// HTTPS and bound to every interface. The API is proxied through this same origin rather
// than called directly: a page on https:// cannot fetch http://, and going through one
// origin also means no CORS entry is needed for each new device.
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
