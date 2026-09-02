import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    // Node >=25 enables its experimental Web Storage by default, and without
    // --localstorage-file the global is a method-less stub. Vitest never
    // shadows an existing global, so jsdom's real localStorage is skipped and
    // tests hit the stub. Disabling it here restores jsdom's Storage; the flag
    // is a no-op on Node 22 (CI), where webstorage is off by default.
    execArgv: ["--no-experimental-webstorage"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
