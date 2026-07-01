// MediaPipe pose provider — runs entirely IN-BROWSER (WASM + GPU) via @mediapipe/
// tasks-vision: PoseLandmarker (+ FaceLandmarker for blendshapes, + optional
// HandLandmarker). No sidecar, no WebSocket, no model download prompt. Same provider
// interface as the others (start/infer/stop/timings/latestFrame). The browser owns the
// webcam (getUserMedia), exactly like the worker path.
//
// Output frame is MediaPipe-native (poseWorld + blendshapes + hands) tagged source:
// 'mediapipe' — the adapter (mediapipe-adapter.js) turns it into the canonical + face
// scalars. keypoints2D (COCO) is filled for the overlay/recorder.
//
// SETUP: needs `npm i @mediapipe/tasks-vision`. Model .task files load from Google's CDN
// by default (below) — self-host for offline/deploy.
import { startWebcam, stopWebcam } from './webcam.js';
import { poseNormToKeypoints2D, fillHands2D } from './mediapipe-adapter.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODELS = {
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
};

export class MediaPipeProvider {
  constructor({ kptThresh = 0.3, face = true, hands = false, delegate = 'GPU' } = {}) {
    this.kptThresh = kptThresh;
    this.wantFace = face;
    this.wantHands = hands;
    this.delegate = delegate;
    this.detectEveryN = 1; // MediaPipe tracks internally; kept for API parity
    this.video = null;
    this.stream = null;
    this.running = false;
    this.latestFrame = null;
    this.ep = 'mediapipe';
    this.timings = { detect: 0, preprocess: 0, inference: 0, decode: 0, total: 0 };
    this._pose = null;
    this._face = null;
    this._hand = null;
  }

  async start() {
    // Dynamic import so the rest of the app doesn't pull MediaPipe unless this backend runs.
    const { FilesetResolver, PoseLandmarker, FaceLandmarker, HandLandmarker } = await import('@mediapipe/tasks-vision');
    const cam = await startWebcam();
    this.video = cam.video;
    this.stream = cam.stream;
    const vision = await FilesetResolver.forVisionTasks(WASM);
    this._pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELS.pose, delegate: this.delegate },
      runningMode: 'VIDEO', numPoses: 1
    });
    if (this.wantFace) {
      this._face = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODELS.face, delegate: this.delegate },
        runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true
      });
      // Static tessellation connectivity (~2600 edges) for the face-mesh overlay — the
      // proper mesh look (three-mediapipe-rig / DrawingUtils), not just dots.
      this._faceEdges = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
    }
    if (this.wantHands) {
      this._hand = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODELS.hand, delegate: this.delegate },
        runningMode: 'VIDEO', numHands: 2
      });
    }
    this.running = true;
  }

  async infer() {
    if (!this.running || !this.video) return null;
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) return null;
    const t = performance.now();

    const poseRes = this._pose.detectForVideo(this.video, t);
    const world = poseRes.worldLandmarks?.[0];
    if (!world) { this.timings.total = performance.now() - t; return null; } // no person this frame

    const k2d = poseNormToKeypoints2D(poseRes.landmarks?.[0], w, h); // COCO body 2D
    const frame = {
      timestampMs: t,
      source: 'mediapipe',
      poseWorld: world, // 33 world landmarks {x,y,z,visibility}
      keypoints2D: k2d, // body (+ hands below) for overlay
      leftHand: null, // 21 WORLD landmarks (wrist-origin) → 3D re-root
      rightHand: null,
      faceBlendshapes: null,
      faceLandmarks2D: null, // 478 face points in px → 2D overlay dot cloud
      boundingBox: null
    };

    if (this._face) {
      const fr = this._face.detectForVideo(this.video, t);
      const cats = fr.faceBlendshapes?.[0]?.categories;
      if (cats) {
        const bs = {};
        for (const c of cats) bs[c.categoryName] = c.score;
        frame.faceBlendshapes = bs;
      }
      const fl = fr.faceLandmarks?.[0];
      if (fl) { frame.faceLandmarks2D = fl.map((p) => ({ x: p.x * w, y: p.y * h })); frame.faceEdges = this._faceEdges; }
    }
    let handCount = 0;
    if (this._hand) {
      const hr = this._hand.detectForVideo(this.video, t);
      const lm = hr.landmarks || [];
      const wl = hr.worldLandmarks || [];
      const hd = hr.handedness || [];
      handCount = lm.length;
      // Assign by handedness NAME (image-space; selfie may swap — downstream mirror
      // handles it), falling back to index so a missing/odd label can't drop the hand.
      for (let i = 0; i < lm.length; i += 1) {
        const name = (hd[i]?.[0]?.categoryName || '').toLowerCase();
        const isLeft = name ? name.startsWith('l') : i === 0;
        if (isLeft) { frame.leftHand = wl[i]; fillHands2D(k2d, lm[i], 91, w, h); }
        else { frame.rightHand = wl[i]; fillHands2D(k2d, lm[i], 112, w, h); }
      }
    }

    this._logN = (this._logN || 0) + 1;
    if (this._logN % 30 === 0) {
      console.log(`[mediapipe] pose:${!!world} face-bs:${frame.faceBlendshapes ? Object.keys(frame.faceBlendshapes).length : 0} hands:${handCount} (landmarker:${!!this._hand})`);
    }

    this.timings.inference = performance.now() - t;
    this.timings.total = this.timings.inference;
    this.latestFrame = frame;
    return frame;
  }

  stop() {
    this.running = false;
    this._pose?.close?.();
    this._face?.close?.();
    this._hand?.close?.();
    this._pose = this._face = this._hand = null;
    stopWebcam(this.stream);
    this.video = null;
    this.stream = null;
  }
}
