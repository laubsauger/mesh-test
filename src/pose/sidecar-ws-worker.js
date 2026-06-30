// Sidecar WS worker — owns the WebSocket to the native Python sidecar OFF the main
// thread. The renderer hogs the main thread, so handling the WS round-trip there
// added ~57ms of pure scheduling lag (the reply waiting behind a render frame).
// Here the round-trip + readback run on this worker thread, decoupled from render:
//   main: createImageBitmap(video) → transfer here
//   here: OffscreenCanvas readback → raw RGBA over WS → keypoints back
//   main: just consumes keypoints when they arrive
// This is what makes the sidecar worth it on a TRT/CUDA box (inference ~10ms → the
// transport, not inference, would otherwise dominate).

const HEADER_BYTES = 20; // ts/seq f64 | w u32 | h u32 | detectEveryN u32

let ws = null;
let canvas = null;
let ctx = null;
let kptThresh = 0.3;
let sendMaxSide = 1280;
let ep = 'sidecar';
let inflight = null; // { seq, sentAt, w, h, vidW, vidH, readbackMs }

function postPose(seq, frame, native, w, h, vidW, vidH, readbackMs, sentAt) {
  // Scale k2d (downscaled-frame px) → video px so it matches the worker backend's
  // overlay contract. k3d is normalized → untouched. Done here so main just consumes.
  let out = null;
  if (frame) {
    const fx = (vidW || 1) / w;
    const fy = (vidH || 1) / h;
    // timestampMs is REQUIRED downstream: the one-euro smoother computes dt from it.
    // Without it dt → NaN → NaN canonical → the retarget NaN-guard freezes the mesh.
    // Stamp with the frame's send time (worker-clock monotonic ms; only deltas matter).
    out = { ...frame, timestampMs: sentAt, keypoints2D: frame.keypoints2D.map((p) => ({ x: p.x * fx, y: p.y * fy, confidence: p.confidence })) };
  }
  const round = performance.now() - sentAt;             // send→reply, measured OFF main thread
  const serverTotal = native?.serverTotal ?? native?.total ?? 0;
  const transport = Math.max(0, round - serverTotal);   // true WS transport (no render lag)
  const timings = {
    detect: native?.detect ?? 0, preprocess: native?.preprocess ?? 0,
    inference: native?.inference ?? 0, decode: native?.decode ?? 0, total: native?.total ?? 0,
    readback: readbackMs, serverDecode: native?.serverDecode ?? 0,
    serverTotal, transport, round, wire: transport, bytes: native?.bytes ?? 0
  };
  postMessage({ type: 'pose', seq, frame: out, timings, ep });
}

function onWsMessage(data) {
  const msg = JSON.parse(data);
  if (msg.type === 'ready') { ep = msg.ep || ep; postMessage({ type: 'inited', ep }); return; }
  if (msg.type === 'pose') {
    if (!inflight || msg.timestampMs !== inflight.seq) return; // stale reply → ignore
    const f = inflight;
    inflight = null;
    postPose(f.seq, msg.frame, msg.timings, f.w, f.h, f.vidW, f.vidH, f.readbackMs, f.sentAt);
  }
}

function processFrame(bitmap, seq) {
  const vidW = bitmap.width;
  const vidH = bitmap.height;
  const scale = Math.min(1, sendMaxSide / Math.max(vidW, vidH));
  const w = Math.round(vidW * scale);
  const h = Math.round(vidH * scale);
  if (canvas.width !== w) { canvas.width = w; canvas.height = h; }

  const rb0 = performance.now();
  ctx.drawImage(bitmap, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data; // GPU→CPU readback (on THIS thread)
  bitmap.close();
  const readbackMs = performance.now() - rb0;

  const buf = new Uint8Array(HEADER_BYTES + rgba.length);
  const dv = new DataView(buf.buffer);
  const sentAt = performance.now();
  dv.setFloat64(0, seq, true);
  dv.setUint32(8, w, true);
  dv.setUint32(12, h, true);
  dv.setUint32(16, 1, true); // sidecar always detects every frame
  buf.set(rgba, HEADER_BYTES);

  inflight = { seq, sentAt, w, h, vidW, vidH, readbackMs };
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
  else { inflight = null; postMessage({ type: 'pose', seq, frame: null, timings: {}, ep }); }
}

onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') {
    kptThresh = m.kptThresh ?? 0.3;
    sendMaxSide = m.sendMaxSide ?? 1280;
    canvas = new OffscreenCanvas(1, 1);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    ws = new WebSocket(m.url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => ws.send(JSON.stringify({ type: 'config', kptThresh }));
    ws.onmessage = (ev) => onWsMessage(ev.data);
    ws.onerror = () => postMessage({ type: 'error', message: `sidecar ws ${m.url} failed — is \`npm run sidecar\` running?` });
    ws.onclose = () => { inflight = null; postMessage({ type: 'closed' }); };
    return;
  }
  if (m.type === 'frame') {
    if (inflight) { m.bitmap.close(); return; } // one in flight (newest-only)
    if (m.sendMaxSide) sendMaxSide = m.sendMaxSide;
    processFrame(m.bitmap, m.seq);
    return;
  }
};
