import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "map-vendor": ["mapbox-gl"],
          "graph-vendor": ["react-force-graph-2d"],
          "markdown-vendor": ["react-markdown"],
          "icons-vendor": ["lucide-react"],
        },
      },
    },
  },
});
