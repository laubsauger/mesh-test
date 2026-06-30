// Pose inference Web Worker (T20). Runs yolo26n + rtmw3d ENTIRELY off the main
// thread, on its OWN GPUDevice (webgpu EP) — an independent command queue from
// the three renderer, so inference no longer contends with rendering. Receives an
// ImageBitmap per frame, returns an RTMWPoseFrame + stage timings.
import { ort, createPoseSession } from './ort-session.js';
import { RTMW3D_MODEL, YOLO_DET_MODEL, POSE_MEAN, POSE_STD } from './rtmw-constants.js';
import { bboxToRect, decode3d, letterboxRect } from './decode.js';
import { assetUrl } from '../asset-url.js';

const R = YOLO_DET_MODEL.res;
const resW = RTMW3D_MODEL.resW;
const resH = RTMW3D_MODEL.resH;

let kptThresh = 0.3;
let poseSession = null;
let detSession = null;
let detCtx;
let detBuf;
let poseCtx;
let poseBuf;
let frameCount = 0;
let lastBoxes = null;
let busy = false;

async function init(msg) {
  kptThresh = msg.kptThresh ?? 0.3;
  const detBytes = await (await fetch(assetUrl(YOLO_DET_MODEL.url))).arrayBuffer();
  detSession = await createPoseSession(detBytes, msg.ep);
  const poseBytes = await (await fetch(assetUrl(RTMW3D_MODEL.url))).arrayBuffer();
  poseSession = await createPoseSession(poseBytes, msg.ep);

  detCtx = new OffscreenCanvas(R, R).getContext('2d', { willReadFrequently: true });
  detBuf = new Float32Array(3 * R * R);
  poseCtx = new OffscreenCanvas(resW, resH).getContext('2d', { willReadFrequently: true });
  poseBuf = new Float32Array(3 * resW * resH);
  postMessage({ type: 'ready' });
}

async function detect(bitmap, vidW, vidH) {
  // Letterbox (aspect-preserved) with gray pad — what yolo expects.
  const lb = letterboxRect(vidW, vidH, R);
  detCtx.fillStyle = 'rgb(114,114,114)';
  detCtx.fillRect(0, 0, R, R);
  detCtx.drawImage(bitmap, 0, 0, vidW, vidH, lb.offsetX, lb.offsetY, lb.drawW, lb.drawH);
  const d = detCtx.getImageData(0, 0, R, R).data;
  const hw = R * R;
  for (let i = 0; i < hw; i += 1) {
    detBuf[i] = d[i * 4] / 255;
    detBuf[hw + i] = d[i * 4 + 1] / 255;
    detBuf[2 * hw + i] = d[i * 4 + 2] / 255;
  }
  const t = new ort.Tensor('float32', detBuf, [1, 3, R, R]);
  const out = await detSession.run({ [YOLO_DET_MODEL.inputName]: t });
  t.dispose();
  const o0 = out.output0;
  const data = o0.data;
  const stride = o0.dims[2];
  const boxes = [];
  const unX = (v) => (v - lb.offsetX) / lb.scale; // R-space px → source px
  const unY = (v) => (v - lb.offsetY) / lb.scale;
  for (let q = 0; q < data.length / stride; q += 1) {
    const o = q * stride;
    const score = data[o + 4];
    if (score < kptThresh) break;
    if (Math.round(data[o + 5]) !== YOLO_DET_MODEL.personClassId) continue;
    const x1 = unX(data[o]);
    const y1 = unY(data[o + 1]);
    boxes.push({ x: x1, y: y1, w: unX(data[o + 2]) - x1, h: unY(data[o + 3]) - y1, score });
  }
  for (const k in out) out[k].dispose();
  boxes.sort((a, b) => b.score - a.score);
  return boxes;
}

async function infer(msg) {
  const bitmap = msg.bitmap;
  const vidW = bitmap.width;
  const vidH = bitmap.height;
  const t0 = performance.now();

  frameCount += 1;
  let boxes;
  if (msg.detectEveryN <= 1 || !lastBoxes || frameCount % msg.detectEveryN === 0) {
    boxes = await detect(bitmap, vidW, vidH);
    lastBoxes = boxes;
  } else {
    boxes = lastBoxes;
  }
  const t1 = performance.now();

  if (!boxes.length) {
    bitmap.close();
    postMessage({ type: 'pose', frame: null, timings: { detect: t1 - t0, preprocess: 0, inference: 0, decode: 0, total: t1 - t0 } });
    return;
  }

  const box = boxes[0];
  const rect = bboxToRect(box, resW, resH);
  poseCtx.clearRect(0, 0, resW, resH);
  poseCtx.drawImage(bitmap, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, resW, resH);
  bitmap.close();
  const pd = poseCtx.getImageData(0, 0, resW, resH).data;
  const phw = resW * resH;
  for (let i = 0; i < phw; i += 1) {
    poseBuf[i] = (pd[i * 4] - POSE_MEAN[0]) / POSE_STD[0];
    poseBuf[phw + i] = (pd[i * 4 + 1] - POSE_MEAN[1]) / POSE_STD[1];
    poseBuf[2 * phw + i] = (pd[i * 4 + 2] - POSE_MEAN[2]) / POSE_STD[2];
  }
  const t2 = performance.now();

  const tensor = new ort.Tensor('float32', poseBuf, [1, 3, resH, resW]);
  const out = await poseSession.run({ [RTMW3D_MODEL.inputName]: tensor });
  tensor.dispose();
  const t3 = performance.now();
  const { k2d, k3d } = decode3d(out, rect, resW, resH);
  for (const k in out) out[k].dispose();
  const t4 = performance.now();

  postMessage({
    type: 'pose',
    frame: {
      timestampMs: msg.timestampMs,
      keypoints2D: k2d.map((p) => ({ x: p.x, y: p.y, confidence: p.score })),
      keypoints3D: k3d.map((p) => ({ x: p.x, y: p.y, z: p.z, confidence: p.score })),
      boundingBox: { x: box.x, y: box.y, width: box.w, height: box.h, confidence: box.score }
    },
    timings: { detect: t1 - t0, preprocess: t2 - t1, inference: t3 - t2, decode: t4 - t3, total: t4 - t0 }
  });
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      await init(msg);
    } else if (msg.type === 'frame') {
      if (busy || !poseSession) { msg.bitmap.close(); return; } // newest-only: drop while busy
      busy = true;
      await infer(msg);
      busy = false;
    }
  } catch (error) {
    busy = false;
    if (msg.bitmap) { try { msg.bitmap.close(); } catch { /* already closed */ } }
    postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};
