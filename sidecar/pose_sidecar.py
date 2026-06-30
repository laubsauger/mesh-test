#!/usr/bin/env python3
"""Native pose inference sidecar — onnxruntime-gpu (TensorRT → CUDA → CPU).

Why this exists: the browser path runs onnxruntime-WEB (webgpu EP), which has no
TensorRT/CUDA. This sidecar runs the SAME yolo26n→crop→rtmw3d pipeline through
native ORT so the 4090 box can use the TensorRT EP (~50ms webgpu → ~5-15ms TRT).

Transport (see src/pose/sidecar-pose-provider.js): the browser owns the webcam,
downscales each frame, reads back RAW RGBA bytes, and ships them over a localhost
WebSocket — NO codec, NO pre-shaped tensor (the pose crop depends on the detector
result computed HERE). This sidecar does det→crop→pose→decode natively and returns
keypoints (tiny JSON). Pixel math is a faithful port of src/pose/{decode,detector}.js
and pose-worker.js — keep them in sync.

Run (uv-managed venv — no global pollution; deps split by platform in pyproject.toml):
      cd sidecar && uv run python pose_sidecar.py [--port 8787] [--ep auto|trt|cuda|cpu]
      (or from repo root: `npm run sidecar`)
"""
import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
import websockets

# --- Constants — MIRROR src/pose/rtmw-constants.js (single source of truth there) ---
REPO = Path(__file__).resolve().parent.parent

# Model selection is a LAUNCH flag (--rtmw-variant l|x, --yolo-res 320/384/512).
# Only l and x are real RTMW3D (3D) releases — there is no 3D m/s.
# Unlike the web path, the sidecar AUTO-READS the rtmw input res + output node names
# from the loaded session, so any variant/fp16 export works with no hardcoding.
def rtmw_path(variant):
    return REPO / f"public/inference/rtmw3d-{variant}/inference_model.onnx"

def yolo_path(res):
    return REPO / f"public/inference/yolo26n/inference_model_{res}.onnx"

RTMW_RES_W, RTMW_RES_H = 288, 384  # fallback only (used if input dims are symbolic)
POSE_MEAN = np.float32([123.675, 116.28, 103.53])
POSE_STD = np.float32([58.395, 57.12, 57.375])
POSE_PADDING = 1.25
Z_RANGE = 2.1744869

PERSON_CLASS = 0

# Proper per-platform GPU device:
#   nvidia (Linux/Win) → TensorRT or CUDA   |   Apple (Mac) → CoreML (ANE/GPU/MPS)
# NOTE: MLX and "MPS" are NOT ONNX Runtime EPs — CoreML IS the Apple-GPU path in
# ORT (it dispatches to ANE/GPU internally). torch is irrelevant to ORT inference.
EP_MAP = {
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def select_providers(ep):
    """auto → TRT→CUDA→CoreML→CPU (skips any not installed → picks the best GPU
    device present on THIS box). A named EP is STRICT: error if unavailable
    (no silent fallback — matches the project EP rule)."""
    available = ort.get_available_providers()
    if ep == "auto":
        order = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                 "CoreMLExecutionProvider", "CPUExecutionProvider"]
        chosen = [p for p in order if p in available]
        if not chosen:
            raise SystemExit(f"no usable EP in {available}")
        return chosen
    want = EP_MAP[ep]
    if want not in available:
        raise SystemExit(f"EP '{ep}' ({want}) not available. Installed: {available}. "
                         f"Use onnxruntime-gpu for trt/cuda; CoreML/CPU ship with onnxruntime on Mac.")
    return [want, "CPUExecutionProvider"] if want != "CPUExecutionProvider" else [want]


TRT_CACHE = REPO / "sidecar" / ".trt-cache"


def trt_options():
    """TensorRT EP options. Engine cache = the SLOW first-run engine build is saved
    to disk and reused (otherwise every launch rebuilds → looks 'stuck' for minutes).
    fp16 = ~2x faster on the 4090's tensor cores (rtmw3d tolerates it well)."""
    TRT_CACHE.mkdir(parents=True, exist_ok=True)
    return {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(TRT_CACHE),
        "trt_timing_cache_enable": True,
        "trt_fp16_enable": True,
    }


def providers_for(ep):
    """EP name → ORT providers list (with options). TRT gets engine caching + fp16,
    and falls back to CUDA (not CPU) for any ops TensorRT can't take."""
    if ep == "TensorrtExecutionProvider":
        opts = trt_options()
        n = len([f for f in TRT_CACHE.glob("*") if f.is_file()]) if TRT_CACHE.exists() else 0
        print(f"[sidecar] TRT fp16=ON, engine cache={TRT_CACHE} ({n} cached file(s) — "
              f"{'HIT, fast start' if n else 'EMPTY, first build will take minutes'})")
        return [(ep, opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if ep == "CPUExecutionProvider":
        return [ep]
    return [ep, "CPUExecutionProvider"]


def probe_ep(test_path, candidates):
    """Return the first GPU EP that ACTUALLY activates, probed on the small yolo
    model. Why not just pass the whole ordered list to InferenceSession? ORT has a
    nasty quirk: if TensorRT is in the list and its plugin registration throws
    (e.g. nvinfer_*.dll missing), ORT falls back STRAIGHT TO CPU — skipping CUDA,
    even though CUDA would work. Probing each EP alone sidesteps that → we get CUDA
    when only TRT is broken."""
    for ep in candidates:
        if ep == "CPUExecutionProvider":
            return "CPUExecutionProvider"
        try:
            if ep == "TensorrtExecutionProvider":
                print("[sidecar] probing TensorRT — building/loading engine (FIRST run can take "
                      "several MINUTES, not frozen; cached to .trt-cache after)…", flush=True)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            s = ort.InferenceSession(str(test_path), sess_options=so, providers=providers_for(ep))
            active = s.get_providers()[0]
            if active == ep:
                return ep
            print(f"[sidecar] EP probe: {ep} registered but inactive (libs missing) → trying next")
        except Exception as e:  # noqa: BLE001
            print(f"[sidecar] EP probe: {ep} failed ({str(e)[:80]}) → trying next")
    return "CPUExecutionProvider"


def make_session(path, providers):
    if not path.exists():
        raise SystemExit(f"model missing: {path}\n  Bring it over by file transfer (see README 'Inference models').")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Silence WARNING-level spam (e.g. CoreML's per-graph "GetCapability / N partitions"
    # node-count notes — informational, not problems). ERROR-level still prints, so the
    # Windows CUDA/TRT DLL-load failures stay visible. 3 = error, 2 = warning.
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


# --- Pixel math — faithful ports (keep in sync with the JS) ---

def letterbox(frame_rgb, R):
    """Aspect-preserved fit into R×R with gray (114) pad — what yolo expects.
    Returns (input_R×R RGB uint8, scale, offX, offY)."""
    h, w = frame_rgb.shape[:2]
    scale = R / max(w, h)
    dw, dh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((R, R, 3), 114, np.uint8)
    ox, oy = (R - dw) // 2, (R - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = resized
    return canvas, scale, ox, oy


def bbox_to_rect(box, res_w=RTMW_RES_W, res_h=RTMW_RES_H, padding=POSE_PADDING):
    """Pad a detector box to the model aspect + padding → crop rect (source px)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    aspect = res_w / res_h
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    w *= padding
    h *= padding
    return cx - w / 2, cy - h / 2, w, h  # sx, sy, sw, sh


def crop_affine(frame_rgb, rect, res_w=RTMW_RES_W, res_h=RTMW_RES_H):
    """Replicate canvas drawImage(src rect → res_w×res_h): out-of-frame source px
    map to 0 (borderValue=0), bilinear — same as the worker's clearRect+drawImage."""
    sx, sy, sw, sh = rect
    M = np.float32([[res_w / sw, 0, -sx * res_w / sw],
                    [0, res_h / sh, -sy * res_h / sh]])
    return cv2.warpAffine(frame_rgb, M, (res_w, res_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def decode3d(out_x, out_y, out_z, rect, res_w=RTMW_RES_W, res_h=RTMW_RES_H):
    """3-axis SimCC argmax decode (vectorized). Returns (k2d, k3d) lists.
    k2d = source-frame px + score (overlay); k3d = root-relative normalized (drive)."""
    tx, ty, tz = out_x[0], out_y[0], out_z[0]  # (K,Wx) (K,Hy) (K,Wz)
    K, Wx = tx.shape
    Hy = ty.shape[1]
    Wz = tz.shape[1]
    bx, vx = tx.argmax(1), tx.max(1)
    by, vy = ty.argmax(1), ty.max(1)
    bz, vz = tz.argmax(1), tz.max(1)
    mx = bx / (Wx / res_w)
    my = by / (Hy / res_h)
    mz = bz / (Wz / res_w)
    score = 0.5 * (vx + vy)  # z maxima unreliable → gate on x,y only
    sx, sy, sw, sh = rect
    k2d = [{"x": float(sx + (mx[k] / res_w) * sw), "y": float(sy + (my[k] / res_h) * sh),
            "confidence": float(score[k])} for k in range(K)]
    k3d = [{"x": float(mx[k] / (res_h / 2)), "y": float(my[k] / (res_h / 2)),
            "z": float((mz[k] / (res_w / 2) - 1) * Z_RANGE), "confidence": float(score[k])}
           for k in range(K)]
    return k2d, k3d


GPU_EPS = {"TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"}


def available_rtmw_variants():
    base = REPO / "public/inference"
    if not base.exists():
        return []
    return sorted(p.name[len("rtmw3d-"):] for p in base.glob("rtmw3d-*")
                  if (p / "inference_model.onnx").exists())


class PosePipeline:
    def __init__(self, providers, variant="l", yolo_res=320, rtmw_out=None, allow_cpu=False):
        t = time.perf_counter()
        rp = rtmw_path(variant)
        if not rp.exists():
            have = available_rtmw_variants()
            hint = (f"on disk now: {have} → run `npm run sidecar -- --rtmw-variant {have[0]}`"
                    if have else "no rtmw3d variant on disk — file-transfer one (see README 'Inference models').")
            raise SystemExit(f"[sidecar] rtmw3d-{variant} missing at {rp}\n  {hint}")
        self.yolo_res = yolo_res
        self.det = make_session(yolo_path(yolo_res), providers)
        self.pose = make_session(rp, providers)

        # Auto-read rtmw I/O from the loaded model → variant/fp16-agnostic.
        pin = self.pose.get_inputs()[0]
        self.in_name = pin.name
        self.res_h = pin.shape[2] if isinstance(pin.shape[2], int) else RTMW_RES_H
        self.res_w = pin.shape[3] if isinstance(pin.shape[3], int) else RTMW_RES_W
        outs = [o.name for o in self.pose.get_outputs()]
        self.out_x, self.out_y, self.out_z = rtmw_out or outs[:3]  # graph order = [X,Y,Z]
        self.det_in = self.det.get_inputs()[0].name
        self.det_out = self.det.get_outputs()[0].name

        self.active_ep = self.pose.get_providers()[0]
        print(f"[sidecar] sessions ready in {time.perf_counter()-t:.1f}s")
        print(f"[sidecar] ACTIVE EP: {self.active_ep}   (det={self.det.get_providers()[0]})")
        print(f"[sidecar] rtmw3d-{variant} {self.res_w}x{self.res_h} in={self.in_name} out={outs[:3]} | yolo {yolo_res}")

        # No SILENT CPU fallback. ORT registers a GPU EP but then quietly drops to
        # CPU when its runtime DLLs (cublas/cudnn/tensorrt) can't load — refuse and
        # surface the exact fix, unless the user opted into CPU.
        if self.active_ep == "CPUExecutionProvider" and not allow_cpu:
            registered_gpu = [e for e in ort.get_available_providers() if e in GPU_EPS]
            if registered_gpu:
                raise SystemExit(
                    f"[sidecar] GPU EP(s) {registered_gpu} are registered but FAILED to load → ORT fell back to "
                    "CPU. The CUDA/cuDNN runtime libs (e.g. cublas64_*.dll) aren't loaded.\n"
                    "  The pyproject now pins onnxruntime-gpu[cuda,cudnn] on the CUDA-12 line, which pulls the\n"
                    "  matching nvidia wheels INTO the venv. If you're seeing this, your venv is STALE (built\n"
                    "  before that change) — force a clean re-resolve:\n"
                    "    rmdir /s /q sidecar\\.venv   (Windows)   |   rm -rf sidecar/.venv   (Linux)\n"
                    "    npm run sidecar              # CUDA EP\n"
                    "    npm run sidecar:cuda         # + TensorRT EP\n"
                    "  Last resort — system CUDA 12 + cuDNN 9 on PATH, or run CPU: npm run sidecar -- --allow-cpu")
            raise SystemExit(
                "[sidecar] no GPU EP available on this box (only CPU). Pass --allow-cpu to run on CPU, "
                "or install onnxruntime-gpu (nvidia) / use Mac CoreML.")

    def detect(self, frame_rgb, thresh):
        h, w = frame_rgb.shape[:2]
        lb, scale, ox, oy = letterbox(frame_rgb, self.yolo_res)
        x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1,3,R,R RGB
        out = self.det.run([self.det_out], {self.det_in: x})[0]
        rows = out[0]  # (N, stride): x1,y1,x2,y2,score,cls,...
        boxes = []
        for r in rows:
            if r[4] < thresh:
                break  # rows score-sorted desc
            if round(float(r[5])) != PERSON_CLASS:
                continue
            x1 = (r[0] - ox) / scale
            y1 = (r[1] - oy) / scale
            x2 = (r[2] - ox) / scale
            y2 = (r[3] - oy) / scale
            boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(r[4])))
        boxes.sort(key=lambda b: -b[4])
        return boxes

    def infer(self, frame_rgb, thresh, boxes):
        t0 = time.perf_counter()
        # detection handled by caller (detectEveryN cache); boxes passed in.
        t1 = time.perf_counter()
        if not boxes:
            return None, {"detect": 0, "preprocess": 0, "inference": 0, "decode": 0, "total": (t1 - t0) * 1000}
        box = boxes[0]
        rect = bbox_to_rect(box[:4], self.res_w, self.res_h)
        crop = crop_affine(frame_rgb, rect, self.res_w, self.res_h)
        chw = ((crop.astype(np.float32) - POSE_MEAN) / POSE_STD).transpose(2, 0, 1)[None]
        t2 = time.perf_counter()
        ox, oy, oz = self.pose.run([self.out_x, self.out_y, self.out_z], {self.in_name: chw})
        t3 = time.perf_counter()
        k2d, k3d = decode3d(ox, oy, oz, rect, self.res_w, self.res_h)
        t4 = time.perf_counter()
        frame = {
            "keypoints2D": k2d,
            "keypoints3D": k3d,
            "boundingBox": {"x": box[0], "y": box[1], "width": box[2], "height": box[3], "confidence": box[4]},
        }
        timings = {"detect": 0, "preprocess": (t2 - t1) * 1000, "inference": (t3 - t2) * 1000,
                   "decode": (t4 - t3) * 1000, "total": (t4 - t0) * 1000}
        return frame, timings


ZERO_TIMINGS = {"detect": 0, "preprocess": 0, "inference": 0, "decode": 0, "total": 0}


async def native_capture_loop(ws, pipe, thresh, device, width, height, preview, preview_w):
    """BACKEND-OWNED capture: the sidecar opens the webcam ITSELF, runs det→crop→pose
    →decode natively, and PUSHES only keypoints to the browser. No camera frame ever
    crosses the wire (the browser sends nothing) → zero readback/transport, latency =
    capture + inference. Latest-frame-wins: BUFFERSIZE=1 + always read the newest.
    Optional: a LOW-RES JPEG preview (so the frontend's debug overlay isn't blind) is
    encoded + pushed as a separate binary message — toggleable live, instrumented."""
    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(None, lambda: cv2.VideoCapture(device))
    if not cap.isOpened():
        await ws.send(json.dumps({"type": "error",
                                  "message": f"sidecar can't open camera {device} — in use by the browser/another app? "
                                             "Native mode needs the camera free (close other camera tabs/apps)."}))
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # latest-frame-wins (no stale backlog)
    cap.set(cv2.CAP_PROP_FPS, 60)  # best-effort; falls back to the cam's max
    print(f"[sidecar] NATIVE capture: camera {device} @ {width}x{height} — keypoints + "
          f"{'low-res preview' if preview else 'NO preview'}")
    await ws.send(json.dumps({"type": "native", "ep": pipe.active_ep}))

    cfg = {"preview": preview, "thresh": thresh}

    async def reader():  # live config: toggle preview / kptThresh without a restart
        try:
            async for m in ws:
                if isinstance(m, str):
                    c = json.loads(m)
                    if "preview" in c:
                        cfg["preview"] = bool(c["preview"])
                    if "kptThresh" in c:
                        cfg["thresh"] = float(c["kptThresh"])
        except websockets.ConnectionClosed:
            pass

    reader_task = asyncio.create_task(reader())
    pushed = 0
    try:
        while True:
            ok, bgr = await loop.run_in_executor(None, cap.read)
            if not ok:
                await asyncio.sleep(0.005)
                continue
            t0 = time.perf_counter()
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR→RGB
            td = time.perf_counter()
            boxes = pipe.detect(rgb, cfg["thresh"])
            detect_ms = (time.perf_counter() - td) * 1000
            frame, timings = pipe.infer(rgb, cfg["thresh"], boxes)
            timings["detect"] = detect_ms
            timings["frameW"] = int(rgb.shape[1])
            timings["frameH"] = int(rgb.shape[0])

            jpg = None
            timings["previewEncode"] = 0.0
            timings["previewBytes"] = 0
            if cfg["preview"]:
                pe0 = time.perf_counter()
                ph = max(1, int(preview_w * bgr.shape[0] / bgr.shape[1]))
                small = cv2.resize(bgr, (preview_w, ph), interpolation=cv2.INTER_AREA)
                okj, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if okj:
                    jpg = buf.tobytes()
                    timings["previewEncode"] = (time.perf_counter() - pe0) * 1000
                    timings["previewBytes"] = len(jpg)

            timings["serverTotal"] = timings["total"] = (time.perf_counter() - t0) * 1000
            pushed += 1
            await ws.send(json.dumps({"type": "pose", "timestampMs": time.perf_counter() * 1000.0,
                                      "frame": frame, "timings": timings, "ep": pipe.active_ep}))
            if jpg is not None:
                await ws.send(jpg)  # binary preview, decoded by the browser as an ImageBitmap
    except websockets.ConnectionClosed:
        pass
    finally:
        reader_task.cancel()
        await loop.run_in_executor(None, cap.release)
        print(f"[sidecar] native capture stopped ({pushed} frames)")


async def handle(ws, pipe):
    print("[sidecar] client connected")
    thresh = 0.3
    frame_count = 0
    last_boxes = None
    frame_errs = 0
    try:
        async for msg in ws:
            if isinstance(msg, str):  # config (text)
                cfg = json.loads(msg)
                thresh = float(cfg.get("kptThresh", thresh))
                if cfg.get("mode") == "native":  # backend owns the camera, pushes results
                    await native_capture_loop(ws, pipe, thresh, int(cfg.get("device", 0)),
                                              int(cfg.get("width", 640)), int(cfg.get("height", 480)),
                                              bool(cfg.get("preview", True)), int(cfg.get("previewW", 320)))
                    return
                await ws.send(json.dumps({"type": "ready", "ep": pipe.active_ep}))
                continue
            # binary frame: header(20) = ts f64 | w u32 | h u32 | detectEveryN u32 | then RGBA.
            # The browser is strictly one-frame-in-flight, so each binary message maps 1:1
            # to the reply below — but ONE bad frame must NOT kill the stream (else the
            # browser's pending promise never resolves → permanent freeze / "freak out"
            # when the box is busy). Validate, process, and ALWAYS reply — on error we
            # reply frame=null (a clean drop) so the loop self-recovers and stays in sync.
            t_recv = time.perf_counter()  # instrumentation: handler entry for this frame
            ts = 0.0
            try:
                ts, w, h, every_n = struct.unpack_from("<dIII", msg, 0)
                expected = 20 + w * h * 4
                if len(msg) != expected:
                    raise ValueError(f"frame bytes {len(msg)} != header expects {expected} (w={w} h={h}) — truncated/garbled, dropping")
                rgba = np.frombuffer(msg, np.uint8, offset=20).reshape(h, w, 4)
                frame_rgb = np.ascontiguousarray(rgba[:, :, :3])  # drop alpha
                td = time.perf_counter()
                frame_count += 1
                if every_n <= 1 or last_boxes is None or frame_count % every_n == 0:
                    last_boxes = pipe.detect(frame_rgb, thresh)
                detect_ms = (time.perf_counter() - td) * 1000
                frame, timings = pipe.infer(frame_rgb, thresh, last_boxes)
                timings["detect"] = detect_ms
                timings["total"] += detect_ms
                # serverDecode = numpy frame decode; serverTotal = whole handler (recv→
                # reply-build) so the browser can split round-trip into server vs wire.
                timings["serverDecode"] = (td - t_recv) * 1000
                timings["serverTotal"] = (time.perf_counter() - t_recv) * 1000
                timings["bytes"] = len(msg)
                reply = {"type": "pose", "timestampMs": ts, "frame": frame, "timings": timings, "ep": pipe.active_ep}
            except Exception as e:  # noqa: BLE001 — one bad frame must not break the stream
                frame_errs += 1
                if frame_errs <= 5:
                    print(f"[sidecar] frame dropped ({frame_errs}): {e}")
                z = dict(ZERO_TIMINGS)
                z["serverTotal"] = (time.perf_counter() - t_recv) * 1000
                reply = {"type": "pose", "timestampMs": ts, "frame": None, "timings": z, "ep": pipe.active_ep}
            await ws.send(json.dumps(reply))
    except websockets.ConnectionClosed:
        print("[sidecar] client disconnected")


def preload_gpu_dlls():
    """Load CUDA/cuDNN/TensorRT from the nvidia-* pip wheels installed in THIS venv
    (the `sidecar:cuda` extra), so onnxruntime-gpu finds them without a global CUDA
    install. Skipped when those wheels aren't installed — otherwise preload_dlls
    spams "Failed to load cublas64_*.dll" for libs that legitimately aren't there
    (plain `npm run sidecar`). No-op on older ORT / Mac (function absent)."""
    if not hasattr(ort, "preload_dlls"):
        return
    import importlib.util
    import os
    have_wheels = (importlib.util.find_spec("nvidia") is not None
                   or importlib.util.find_spec("tensorrt") is not None
                   or importlib.util.find_spec("tensorrt_libs") is not None)
    if not have_wheels:
        return  # no nvidia runtime wheels → don't probe (avoids the DLL-load spam)

    # preload_dlls() reliably loads the nvidia-* CUDA/cuDNN wheels but NOT the TensorRT
    # libs (the `tensorrt_libs` package, where nvinfer_10.dll lives) — so ORT's TRT
    # provider DLL can't find nvinfer_10.dll. Put that dir on the DLL search path
    # explicitly (Windows: add_dll_directory; also PATH for good measure).
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec and spec.submodule_search_locations:
        trt_dir = list(spec.submodule_search_locations)[0]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(trt_dir)
            except OSError:
                pass
        os.environ["PATH"] = trt_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[sidecar] TensorRT libs on DLL path: {trt_dir}")

    try:
        ort.preload_dlls()  # cuda=True, cudnn=True, tensorrt=True by default
        print("[sidecar] ort.preload_dlls() loaded CUDA/cuDNN/TRT from venv wheels")
    except Exception as e:  # noqa: BLE001 — best-effort; refusal check below catches real failure
        print(f"[sidecar] preload_dlls skipped: {e}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--ep", choices=["auto", "trt", "cuda", "coreml", "cpu"], default="auto")
    ap.add_argument("--rtmw-variant", choices=["l", "x"], default="l", help="rtmw3d 3D size: l=faster(219MB), x=largest(352MB)")
    ap.add_argument("--yolo-res", type=int, choices=[320, 384, 512], default=320)
    ap.add_argument("--allow-cpu", action="store_true", help="permit CPU fallback instead of refusing")
    ap.add_argument("--rtmw-out", help="comma-sep X,Y,Z output names (else auto-read from the model)")
    args = ap.parse_args()

    ort.set_default_logger_severity(3)  # hide WARNING-level EP partition spam (errors still print)
    preload_gpu_dlls()
    providers = select_providers(args.ep)
    # AUTO: probe each candidate (on the small yolo model) and pin the one that
    # actually activates — so a broken/absent TensorRT can't drag us to CPU past a
    # working CUDA EP (ORT's TRT-fail→CPU quirk). Explicit --ep stays strict.
    winner = probe_ep(yolo_path(args.yolo_res), providers) if args.ep == "auto" else providers[0]
    if args.ep == "auto":
        print(f"[sidecar] auto-EP probe → {winner}")
    providers = providers_for(winner)  # TRT gets engine-cache + fp16 options
    if winner == "TensorrtExecutionProvider":
        print(f"[sidecar] building TensorRT engine for rtmw3d-{args.rtmw_variant} — FIRST run takes "
              f"MINUTES (cached to {TRT_CACHE.name}/, instant after). Not frozen…", flush=True)
    rtmw_out = tuple(args.rtmw_out.split(",")) if args.rtmw_out else None
    print(f"[sidecar] requested EP={args.ep} variant={args.rtmw_variant} yolo={args.yolo_res} → providers {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    pipe = PosePipeline(providers, variant=args.rtmw_variant, yolo_res=args.yolo_res,
                        rtmw_out=rtmw_out, allow_cpu=(args.allow_cpu or args.ep == "cpu"))

    async with websockets.serve(lambda ws: handle(ws, pipe), "127.0.0.1", args.port, max_size=16 * 1024 * 1024):
        print(f"[sidecar] listening ws://127.0.0.1:{args.port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
