import { defineConfig, mergeConfig } from "vitest/config";
// The extension is explicit: Vite's native config loader warns without it, and
// that warning is going to become an error in a future major version.
import viteConfig from "./vite.config.ts";

// The component tests run in jsdom, which has no layout engine and no pixels.
// They prove behaviour -- thirty rows, a click selecting the right photograph,
// no forbidden wording -- and deliberately not appearance. Appearance is
// checked by a person opening the browser, and that check is listed separately
// rather than implied.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      globals: false,
      include: ["src/**/*.test.{ts,tsx}"],
      restoreMocks: true,
    },
  }),
);
