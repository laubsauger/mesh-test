// onnxruntime-web session factory. Creates an InferenceSession on EXACTLY the
// requested EP — a failed EP throws, never silently falls back (V23).
import * as ort from 'onnxruntime-web';
import { assertPoseEP } from './ep.js';
import { assetUrl } from '../asset-url.js';

let configured = false;

function configureOrt() {
  if (configured) return;
  ort.env.logLevel = 'error';
  // `/ort/` is served from node_modules by the serve-ort Vite plugin (dev) and
  // emitted to dist/ort on build — same-origin (COEP-safe), no /public transform
  // clash, no CDN. ort fetches its wasm + jsep .mjs glue from here.
  ort.env.wasm.wasmPaths = assetUrl('ort/');
  // wasm threads need SharedArrayBuffer → cross-origin isolation (COOP+COEP set
  // in vite.config). Request threads only when actually isolated.
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? (navigator.hardwareConcurrency || 4) : 1;
  configured = true;
}

export { ort };

export async function createPoseSession(modelData, ep, extra = {}) {
  assertPoseEP(ep);
  configureOrt();
  return ort.InferenceSession.create(modelData, {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    ...extra
  });
}
