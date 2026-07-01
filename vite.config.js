import { createRequire } from 'node:module';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// `base` is the path the site is served from. GitHub Pages project sites live at
// https://<user>.github.io/<repo>/, so CI passes VITE_BASE=/<repo>/. Locally it
// defaults to '/'. import.meta.env.BASE_URL in the app resolves asset URLs.

// Cross-origin isolation (COOP + COEP) enables SharedArrayBuffer → onnxruntime-web
// wasm multi-threading (and a pose worker later). Requires same-origin wasm,
// which the serve-ort plugin provides.
const setIsolationHeaders = (res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
};

const crossOriginIsolation = {
  name: 'cross-origin-isolation',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => { setIsolationHeaders(res); next(); });
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => { setIsolationHeaders(res); next(); });
  }
};

// Serve onnxruntime-web's wasm + jsep .mjs glue from /ort/ — same-origin so COEP
// allows it. node_modules can't go through ?url (ort's exports field blocks deep
// imports) nor /public (Vite forbids importing .mjs from /public). So: serve
// straight from node_modules in dev (before Vite's transform), emit to dist/ort
// on build. ort.env.wasm.wasmPaths points here.
const require = createRequire(import.meta.url);
// resolve the package main (exported), then its dist dir holds the wasm/mjs.
const ortDistDir = dirname(require.resolve('onnxruntime-web'));
const ortFiles = () => readdirSync(ortDistDir).filter((f) => /^ort-wasm-simd-threaded.*\.(wasm|mjs)$/.test(f));
const ortMime = (f) => (f.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');

const serveOrt = {
  name: 'serve-ort-wasm',
  configureServer(server) {
    // Direct use() in configureServer runs BEFORE Vite's internal middlewares,
    // so /ort/*.mjs is served static instead of treated as a source module.
    server.middlewares.use((req, res, next) => {
      const path = req.url?.split('?')[0];
      if (!path || !path.includes('/ort/')) return next();
      const name = basename(path);
      if (!/^ort-wasm-simd-threaded.*\.(wasm|mjs)$/.test(name)) return next();
      try {
        const body = readFileSync(join(ortDistDir, name));
        res.setHeader('Content-Type', ortMime(name));
        res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
        res.end(body);
      } catch {
        next();
      }
    });
  },
  generateBundle() {
    for (const f of ortFiles()) {
      this.emitFile({ type: 'asset', fileName: `ort/${f}`, source: readFileSync(join(ortDistDir, f)) });
    }
  }
};

export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [crossOriginIsolation, serveOrt, react(), tailwindcss()]
});
