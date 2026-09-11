import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: Number(process.env.PORT) || 8080,
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(
    Boolean
  ),
  build: {
    rollupOptions: {
      // Strip console.log / debug / info in production; keep console.warn and
      // console.error for observability of real problems.
      //
      // This was `esbuild: { pure: [...] }` until Vite 8, which replaced esbuild
      // with rolldown/oxc — esbuild is no longer even installed, so the option
      // became dead config that quietly stopped stripping anything.
      // `manualPureFunctions` is rolldown's equivalent: marking these calls pure
      // lets tree-shaking drop them. There is no `compress.dropConsole`
      // alternative here — that one takes warn and error with it.
      treeshake: {
        manualPureFunctions:
          mode === "production"
            ? ["console.log", "console.debug", "console.info"]
            : [],
      },
    },
  },
  preview: {
    allowedHosts: ["lerobot-makerlab.hf.space"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
