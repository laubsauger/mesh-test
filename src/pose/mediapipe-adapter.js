// MediaPipe → canonical adapter. Maps BlazePose-33 + Face blendshapes (+ optional
// Hands-21) into the SAME structures the retargeter + face system already consume, so
// nothing downstream changes. This is the ONLY MediaPipe-specific handling — the "adapter
// boundary" from the design: topology remap + axis/z convention + hand re-root.
//
// Conventions:
//   MediaPipe pose WORLD landmarks: meters, origin = hip midpoint, x-right, y-DOWN,
//     z ≈ toward-camera (smaller = closer). Canonical wants x-right, y-UP, z-FORWARD,
//     pelvis-centered → flip y and z (AX). Signs are the one thing to verify live.
//   Hands come WRIST-relative (own origin) → re-rooted onto the pose wrist (your point).
import { KPT, NUM_KPTS } from './rtmw-constants.js';

// COCO-WholeBody body index → BlazePose-33 index. BlazePose has every joint the rig reads
// (nose/eyes/ears/shoulders/elbows/wrists/hips/knees/ankles/feet); it lacks a distinct
// small-toe so we reuse the foot-index point there.
export const COCO_FROM_BLAZE = {
  [KPT.nose]: 0,
  [KPT.leftEye]: 2, [KPT.rightEye]: 5,
  [KPT.leftEar]: 7, [KPT.rightEar]: 8,
  [KPT.leftShoulder]: 11, [KPT.rightShoulder]: 12,
  [KPT.leftElbow]: 13, [KPT.rightElbow]: 14,
  [KPT.leftWrist]: 15, [KPT.rightWrist]: 16,
  [KPT.leftHip]: 23, [KPT.rightHip]: 24,
  [KPT.leftKnee]: 25, [KPT.rightKnee]: 26,
  [KPT.leftAnkle]: 27, [KPT.rightAnkle]: 28,
  [KPT.leftBigToe]: 31, [KPT.leftSmallToe]: 31, [KPT.leftHeel]: 29,
  [KPT.rightBigToe]: 32, [KPT.rightSmallToe]: 32, [KPT.rightHeel]: 30
};

// Flip MediaPipe axes into the canonical frame. If a limb points the wrong way or depth
// is inverted at runtime, flip one of these — the single place axis bugs live.
export const AX = { x: 1, y: -1, z: -1 };

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2, confidence: Math.min(a.confidence, b.confidence) });

// Re-root a wrist-origin hand (21 world landmarks) onto the canonical wrist joint.
function rerootHand(joints, hand, wrist, base) {
  if (!hand || !wrist) return;
  for (let k = 0; k < 21 && k < hand.length; k += 1) {
    const p = hand[k];
    joints[base + k] = {
      x: wrist.x + p.x * AX.x,
      y: wrist.y + p.y * AX.y,
      z: wrist.z + p.z * AX.z,
      // HandLandmarker leaves visibility=0 (no visibility head) — a real 0, not
      // undefined, so `??` wouldn't fall back. `||` treats absent (0) as confident.
      confidence: p.visibility || 0.9
    };
  }
}

// frame = { timestampMs, poseWorld:[33 {x,y,z,visibility}], leftHand?, rightHand?, boundingBox? }
// → CanonicalPoseObservation (pelvis-centered, +y up) — identical shape to toCanonical().
export function toCanonicalFromMediaPipe(frame) {
  const w = frame.poseWorld;
  if (!w || w.length < 33) throw new Error('mediapipe: missing pose world landmarks');
  const lh = w[23];
  const rh = w[24];
  const px = (lh.x + rh.x) / 2;
  const py = (lh.y + rh.y) / 2;
  const pz = (lh.z + rh.z) / 2;

  const joints = new Array(NUM_KPTS);
  for (let i = 0; i < NUM_KPTS; i += 1) joints[i] = { x: 0, y: 0, z: 0, confidence: 0 };
  for (const key of Object.keys(COCO_FROM_BLAZE)) {
    const coco = Number(key);
    const p = w[COCO_FROM_BLAZE[key]];
    joints[coco] = {
      x: (p.x - px) * AX.x,
      y: (p.y - py) * AX.y,
      z: (p.z - pz) * AX.z,
      confidence: p.visibility ?? 1
    };
  }
  // Optional articulated hands (COCO-WholeBody: left 91-111, right 112-132).
  rerootHand(joints, frame.leftHand, joints[KPT.leftWrist], 91);
  rerootHand(joints, frame.rightHand, joints[KPT.rightWrist], 112);

  return {
    timestampMs: frame.timestampMs,
    joints,
    pelvisCenter: { x: 0, y: 0, z: 0 },
    shoulderCenter: mid(joints[KPT.leftShoulder], joints[KPT.rightShoulder]),
    boundingBox: frame.boundingBox ?? null
  };
}

// Normalized pose landmarks (image space, 0..1) → COCO keypoints2D in pixels, for the
// overlay + recorder (same contract as the RTMW path).
export function poseNormToKeypoints2D(norm, width, height) {
  const k2d = Array.from({ length: NUM_KPTS }, () => ({ x: 0, y: 0, confidence: 0 }));
  if (!norm) return k2d;
  for (const key of Object.keys(COCO_FROM_BLAZE)) {
    const coco = Number(key);
    const p = norm[COCO_FROM_BLAZE[key]];
    if (p) k2d[coco] = { x: p.x * width, y: p.y * height, confidence: p.visibility ?? 1 };
  }
  return k2d;
}

// Fill the 21 COCO-WholeBody hand keypoints2D (left base 91, right base 112) from a
// MediaPipe hand's 21 NORMALIZED landmarks — MP's layout matches COCO 1:1. For overlay.
export function fillHands2D(k2d, handNorm, base, width, height) {
  if (!handNorm) return;
  for (let k = 0; k < 21 && k < handNorm.length; k += 1) {
    const p = handNorm[k];
    k2d[base + k] = { x: p.x * width, y: p.y * height, confidence: p.visibility || 0.9 }; // hands: visibility=0, see rerootHand
  }
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

// ARKit-style face blendshapes (from FaceLandmarker) → the 7 scalars the face system
// already drives. Direct + richer than extracting from sparse landmarks. `bs` = name→score.
// NOTE: ARKit L/R are SUBJECT-relative; a selfie mirror may swap them — flip if inverted.
export function blendshapesToExpr(bs) {
  const g = (n) => bs[n] ?? 0;
  return {
    jawOpen: clamp01(g('jawOpen')),
    smile: clamp01((g('mouthSmileLeft') + g('mouthSmileRight')) / 2),
    pucker: clamp01(g('mouthPucker')),
    blinkL: clamp01(g('eyeBlinkLeft')),
    blinkR: clamp01(g('eyeBlinkRight')),
    browL: clamp01(g('browInnerUp') + g('browOuterUpLeft') - g('browDownLeft')),
    browR: clamp01(g('browInnerUp') + g('browOuterUpRight') - g('browDownRight'))
  };
}
