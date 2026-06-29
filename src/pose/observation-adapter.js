// PoseObservationAdapter (T7 / V21, V22): bridge an RTMWPoseFrame (decode3d
// normalized space, score field) into a CanonicalPoseObservation — pelvis-
// centered, +y up, score→confidence. Pure (no three/DOM), unit-testable.
//
// decode3d k3d is root-relative normalized with image-down y. We recenter on the
// hip midpoint and negate axes to the canonical basis used by the proven viewer:
//   +x performer-right, +y up, +z forward, origin = hip midpoint (§5).
import { KPT } from './rtmw-constants.js';

function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

export function toCanonical(frame) {
  const k = frame.keypoints3D;
  const lh = k[KPT.leftHip];
  const rh = k[KPT.rightHip];
  if (!lh || !rh) throw new Error('toCanonical: missing hip keypoints (11/12) — cannot establish pelvis origin');

  const pelvis = mid(lh, rh);

  // Canonical frame matched to the Meshy rig (measured): +x = image-right =
  // performer's anatomical LEFT = rig left (+x); +y = up (image y is down → flip).
  // Pelvis-centered. Puppet-correct with NO extra mirror (the old all-axes negate
  // was copied from the reference VIEWER → double-flip → crossed arms, see §B).
  // Depth (z) sign flipped (RTMW depth opposes rig +z). Depth SCALING is applied
  // per-segment in the retargeter — arms need depth; torso/head/legs de-emphasize
  // it to avoid droop / knee back-fold from noisy monocular depth.
  const joints = k.map((p) => ({
    x: p.x - pelvis.x,
    y: -(p.y - pelvis.y),
    z: -(p.z - pelvis.z),
    confidence: p.confidence
  }));

  const ls = joints[KPT.leftShoulder];
  const rs = joints[KPT.rightShoulder];

  return {
    timestampMs: frame.timestampMs,
    joints, // indexed by COCO-WholeBody id; semantic access via KPT
    pelvisCenter: { x: 0, y: 0, z: 0 }, // origin by construction
    shoulderCenter: mid(ls, rs),
    boundingBox: frame.boundingBox ?? null
  };
}
