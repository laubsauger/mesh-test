# Mesh VJ Layer — WebGPU crowd + real-time webcam pose driving

A WebGPU "VJ layer": GPU-skinned crowd of Meshy biped characters with live
arrangement / motion / lighting, **and** real-time RTMW3D webcam pose driving one
(or many) of the characters. Three.js r0.185 + TSL + Vite. Pure ESM. **WebGPU
only** — no WebGL fallback.

See `SPEC.md` for the full spec/invariants and `docs/pose-driver.md` for the pose
design.

## Prerequisites

- **Node 18+** and npm.
- A **WebGPU browser**: Chrome or Edge (Windows/macOS/Linux). Check
  `chrome://gpu` shows WebGPU enabled.
- A **webcam** for pose driving.
- The **pose ONNX models** (large, git-ignored — see below).

## Assets: two separate folders

Two kinds of model live under `public/`, kept in **distinct** folders so the
overloaded word "models" stops being ambiguous:

| Folder | Contents | In git? | Source |
| --- | --- | --- | --- |
| `public/models/` | Character **mesh** GLBs (`*_biped.glb`) | ✅ committed | `npm run optimize:bipeds` |
| `public/inference/` | Pose **inference** ONNX models | ❌ git-ignored | file transfer (see below) |

### Inference models (required for pose, git-ignored)

The pose ONNX models are **not committed** — `rtmw3d-x` is ~352 MB (> GitHub's
100 MB file limit) — and they are **not regenerable from this repo**: they're
exported by the separate **`object-detect`** project, which is not vendored
here. A fresh clone has none. Bring them over by **file transfer** and drop them
at:

```
public/inference/rtmw3d-x/inference_model.onnx      # RTMW3D-x 3D pose (≈352 MB)
public/inference/yolo26n/inference_model_384.onnx   # person detector (top-down crop)
```

`npm run check:models` verifies they're present (it also runs automatically
before `dev`/`build`). It's **warn-only**: the crowd renderer runs without them —
only webcam pose driving needs them, and would otherwise fail with an opaque
"model fetch 404" at runtime.

## Run

```bash
npm install
npm run dev        # http://127.0.0.1:5173  (port may shift if taken)
npm run build      # production build → dist/  (also emits dist/ort/)
npm run preview    # serve the built dist/
npm test           # vitest (pure pose-math units)
```

The dev server sets **COOP/COEP** headers (cross-origin isolation) so
onnxruntime-web can multi-thread; a small Vite plugin serves the ort wasm from
`node_modules` (and emits to `dist/ort/` on build). No CDN, no manual copy.

All runtime controls live in the three.js **Inspector** panels (`VJ Layer / *`),
not custom DOM — Pose, Pose Stats, Scene, Lighting, Meshes.

## Pose pipeline (short version)

Webcam → **Web Worker** (`src/pose/pose-worker.js`, own GPUDevice so inference
doesn't contend with the renderer) → yolo26n detect (letterboxed 384) → crop →
rtmw3d → SimCC decode → canonical → retarget onto the Meshy rig. Inference EP is
**webgpu** (the only viable one in-browser here; wasm OOMs on the 369 MB model
alongside the GLB scene — see `SPEC.md` §B5).

## Windows / performance

Cloning and running on **Windows with a discrete NVIDIA/AMD GPU + Chrome WebGPU**
is the easy perf win — the in-browser webgpu EP will be far faster than an
integrated/battery Mac (the 369 MB model's per-inference time drops a lot). Just
follow **Run** above; everything is cross-platform (the Vite plugin uses Node path
APIs, COOP/COEP works on Windows dev).

**TensorRT note:** onnxruntime-**web** (what runs in the browser) has **no
TensorRT EP** — TRT is a *native* provider. To use TensorRT / CUDA you'd run pose
inference **natively** (Python/C++ onnxruntime with the TRT EP) as a local server
and stream `RTMWPoseFrame`s to the browser over WebSocket — the same bridge
pattern `object-detect/bridge.py` uses for detection. That's a separate native
path (not yet built here); the browser side already speaks `RTMWPoseFrame`, so the
adapter/retarget/render stack would be reused unchanged. This is the route for
max performance on a Windows TRT box.
