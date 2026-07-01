// Native-capture pose provider — the OPPOSITE of SidecarPoseProvider. The native
// sidecar owns the webcam: it captures, runs det→crop→pose→decode, and PUSHES only
// keypoints over the WebSocket. The browser sends NOTHING back — no getUserMedia, no
// readback, no frame transport. Latency = capture + inference (~10ms on the 4090),
// not the ~100ms of shuffling a frame across the process boundary every cycle.
//
// Same provider API (start/infer/stop/timings/latestFrame/stats) so the pose loop is
// unchanged. infer() is push-driven: it returns the next frame the sidecar pushes.
// Trade-off: the browser has no raw-video preview (the camera is the sidecar's) — the
// 2D skeleton overlay + the driven 3D mesh are the feedback.

export class NativeCapturePoseProvider {
  constructor({ kptThresh = 0.3, url = 'ws://127.0.0.1:8787', device = 0, width = 640, height = 480, preview = true } = {}) {
    this.kptThresh = kptThresh;
    this.url = url;
    this.device = device;
    this.width = width;
    this.height = height;
    this.previewOn = preview;
    this.preview = null; // latest decoded preview ImageBitmap (overlay backdrop), or null
    this.detectEveryN = 1; // unused (sidecar owns cadence); API parity
    this.ws = null;
    this.running = false;
    this.latestFrame = null;
    this.ep = 'sidecar-native';
    // No browser camera → expose a video STUB so the pose loop's overlay-sizing
    // (reads video.videoWidth) works; filled from the first pushed frame's dims.
    this.video = { videoWidth: 0, videoHeight: 0 };
    this.stream = null;
    this.timings = {
      capture: 0, detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0,
      readback: 0, serverDecode: 0, serverTotal: 0, transport: 0, round: 0, wire: 0, bytes: 0
    };
    this.stats = { sent: 0, recv: 0, dropped: 0, stale: 0, timedOut: 0, sendHz: 0 };
    this.debug = false;
    this._waiter = null; // resolve for an infer() awaiting the requested frame
    this._waitTimer = null;
    this._lastPushAt = 0;
  }

  async start() {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      const fail = (e) => reject(new Error(`sidecar ws ${this.url}: ${e?.message || 'connection failed — is `npm run sidecar` running?'}`));
      this.ws.onerror = fail;
      this.ws.onclose = () => { if (!this.running) fail(); else this._wake(null); };
      this.ws.onopen = () => {
        this._readyResolve = resolve;
        this._readyReject = reject;
        // ask the sidecar to OWN the camera and push results (+ optional preview)
        this.ws.send(JSON.stringify({ type: 'config', mode: 'native', kptThresh: this.kptThresh,
          device: this.device, width: this.width, height: this.height, preview: this.previewOn }));
      };
      this.ws.onmessage = (e) => this._onMessage(e.data);
    });
    this.running = true;
  }

  _onMessage(data) {
    if (data instanceof ArrayBuffer) { this._onPreview(data); return; } // binary = preview JPEG
    const msg = JSON.parse(data);
    if (msg.type === 'native') { // sidecar confirmed it owns the camera
      this.ep = msg.ep || 'sidecar-native';
      this._readyResolve?.(); this._readyResolve = null; this._readyReject = null;
      return;
    }
    if (msg.type === 'error') {
      this._readyReject?.(new Error(msg.message)); this._readyReject = null; this._readyResolve = null;
      console.error('[sidecar-native]', msg.message);
      return;
    }
    if (msg.type === 'pose') {
      const t = msg.timings || {};
      if (t.frameW) { this.video.videoWidth = t.frameW; this.video.videoHeight = t.frameH; }
      const now = performance.now();
      if (this._lastPushAt) this.stats.sendHz = 1000 / Math.max(1, now - this._lastPushAt);
      this._lastPushAt = now;
      this.timings = { ...this.timings, ...t };
      // Stamp the frame with receive-time so RECORDINGS carry timing (the sidecar's raw
      // `frame` has none → analyzer couldn't compute rates). Monotonic; deltas are valid.
      if (msg.frame && msg.frame.timestampMs == null) msg.frame.timestampMs = now;
      this.latestFrame = msg.frame;
      if (msg.frame) this.stats.recv += 1; else this.stats.dropped += 1;
      if (this.debug) {
        console.log(`[native] ${msg.frame ? 'pose' : 'NULL'} | conf ${t.poseConf ?? '-'} box ${JSON.stringify(t.box ?? null)} | det ${(t.detect ?? 0).toFixed(1)} inf ${(t.inference ?? 0).toFixed(1)} server ${(t.serverTotal ?? 0).toFixed(1)}ms @ ${(this.stats.sendHz || 0).toFixed(0)}Hz | recv ${this.stats.recv}`);
      }
      this._wake(msg.frame);
    }
  }

  // Decode the low-res preview JPEG → ImageBitmap (overlay backdrop). Measured so the
  // Pose Stats panel shows what the preview costs (browser-side decode).
  async _onPreview(buf) {
    const d0 = performance.now();
    try {
      const bmp = await createImageBitmap(new Blob([buf], { type: 'image/jpeg' }));
      this.preview?.close?.();
      this.preview = bmp;
      this.timings.previewDecode = performance.now() - d0;
    } catch { /* drop a bad preview frame */ }
  }

  // Live toggle (no restart): tell the sidecar to start/stop encoding the preview.
  setPreview(on) {
    this.previewOn = on;
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'config', preview: on }));
    if (!on) { this.preview?.close?.(); this.preview = null; }
  }

  // Resolve the infer() waiting for the requested frame.
  _wake(frame) {
    if (this._waitTimer) { clearTimeout(this._waitTimer); this._waitTimer = null; }
    const w = this._waiter; this._waiter = null; w?.(frame);
  }

  // Browser-PACED pull: request one frame (the sidecar reads the freshest camera frame
  // and pushes it), then await it. Strictly one-in-flight → the sidecar never floods
  // the WS (the Windows disconnect bug). Timeout so a dead sidecar can't hang the loop.
  async infer() {
    if (!this.running || this.ws?.readyState !== WebSocket.OPEN) return null;
    this.ws.send(JSON.stringify({ type: 'ready' }));
    return new Promise((resolve) => {
      this._waiter = resolve;
      this._waitTimer = setTimeout(() => { if (this._waiter === resolve) { this.stats.timedOut += 1; this._waiter = null; resolve(null); } }, 2000);
    });
  }

  stop() {
    this.running = false;
    this._wake(null); // release any awaiting infer()
    this.preview?.close?.(); this.preview = null;
    if (this.ws) { try { this.ws.close(); } catch { /* already closed */ } this.ws = null; }
  }
}
