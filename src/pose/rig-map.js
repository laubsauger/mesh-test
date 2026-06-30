// HumanoidRigDefinition + canonical→bone map (T11). All 7 Meshy bipeds share one
// 24-bone Mixamo-style rig (verified across every GLB), so one map is universal.
// Canonical (COCO-WholeBody) joints drive bones by segment direction: each
// directional bone is aimed from `from` KPT toward `to` KPT (T12 retargeter).
import { KPT } from './rtmw-constants.js';

export const MESHY_BONES = {
  hips: 'Hips',
  spine: 'Spine',
  spine01: 'Spine01',
  spine02: 'Spine02',
  neck: 'neck',
  head: 'Head',
  leftShoulder: 'LeftShoulder',
  leftArm: 'LeftArm',
  leftForeArm: 'LeftForeArm',
  leftHand: 'LeftHand',
  rightShoulder: 'RightShoulder',
  rightArm: 'RightArm',
  rightForeArm: 'RightForeArm',
  rightHand: 'RightHand',
  leftUpLeg: 'LeftUpLeg',
  leftLeg: 'LeftLeg',
  leftFoot: 'LeftFoot',
  leftToeBase: 'LeftToeBase',
  rightUpLeg: 'RightUpLeg',
  rightLeg: 'RightLeg',
  rightFoot: 'RightFoot',
  rightToeBase: 'RightToeBase'
};

// Endpoint resolvers (pure) → {x,y,z,confidence} from a canonical observation.
// KPT pairs for limbs; computed centers for torso/head so the spine bends and the
// head tracks (pelvis is the canonical origin).
const kpt = (i) => (c) => c.joints[i];
const mid = (a, b) => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
  z: (a.z + b.z) / 2,
  confidence: Math.min(a.confidence, b.confidence)
});
export const hipCenter = (c) => mid(c.joints[KPT.leftHip], c.joints[KPT.rightHip]);
export const shoulderCenter = (c) => mid(c.joints[KPT.leftShoulder], c.joints[KPT.rightShoulder]);

// Head target from the centroid of all face points (nose + eyes + ears), not one
// noisy nose point — averaging 5 keypoints is far steadier for head direction.
const HEAD_POINTS = [KPT.nose, KPT.leftEye, KPT.rightEye, KPT.leftEar, KPT.rightEar];
export const headCenter = (c) => {
  let x = 0;
  let y = 0;
  let z = 0;
  let conf = 1;
  let n = 0;
  for (const i of HEAD_POINTS) {
    const j = c.joints[i];
    if (!j) continue;
    x += j.x;
    y += j.y;
    z += j.z;
    conf = Math.min(conf, j.confidence);
    n += 1;
  }
  return n ? { x: x / n, y: y / n, z: z / n, confidence: conf } : { x: 0, y: 0, z: 0, confidence: 0 };
};

// Upper limb bones get full swing-TWIST (direction + bend-plane) so the arm/leg
// roll is determined, not left to bind roll (§22). root/mid/end = the 3 canonical
// joints; midBone/endBone give the bind bend-plane from the rig.
export const LIMBS = [
  { upper: 'leftArm', midBone: 'leftForeArm', endBone: 'leftHand', root: KPT.leftShoulder, mid: KPT.leftElbow, end: KPT.leftWrist, depth: 1 },
  { upper: 'rightArm', midBone: 'rightForeArm', endBone: 'rightHand', root: KPT.rightShoulder, mid: KPT.rightElbow, end: KPT.rightWrist, depth: 1 },
  { upper: 'leftUpLeg', midBone: 'leftLeg', endBone: 'leftFoot', root: KPT.leftHip, mid: KPT.leftKnee, end: KPT.leftAnkle, depth: 0.35 },
  { upper: 'rightUpLeg', midBone: 'rightLeg', endBone: 'rightFoot', root: KPT.rightHip, mid: KPT.rightKnee, end: KPT.rightAnkle, depth: 0.35 }
];

// Forearms: direction (elbow→wrist) + optional axial TWIST from the hand knuckle
// line (indexMCP→pinkyMCP rotates with pronation/supination).
export const FOREARMS = [
  { bone: 'leftForeArm', root: KPT.leftElbow, mid: KPT.leftWrist, rollA: KPT.leftIndexBase, rollB: KPT.leftPinkyBase, depth: 1 },
  { bone: 'rightForeArm', root: KPT.rightElbow, mid: KPT.rightWrist, rollA: KPT.rightIndexBase, rollB: KPT.rightPinkyBase, depth: 1 }
];

// Direction-only bones (swing). `depth` damps noisy monocular z per-bone.
export const SEGMENTS = [
  { bone: 'leftLeg', from: kpt(KPT.leftKnee), to: kpt(KPT.leftAnkle), depth: 0.35 },
  { bone: 'rightLeg', from: kpt(KPT.rightKnee), to: kpt(KPT.rightAnkle), depth: 0.35 },
  { bone: 'leftFoot', from: kpt(KPT.leftAnkle), to: kpt(KPT.leftBigToe), depth: 0.35 },
  { bone: 'rightFoot', from: kpt(KPT.rightAnkle), to: kpt(KPT.rightBigToe), depth: 0.35 },
  // Hands: wrist → middle-finger base. Leaf bones (no child) — bind dir captured
  // from forearm→hand. Drives wrist bend; twist (pronation) is a follow-up.
  { bone: 'leftHand', from: kpt(KPT.leftWrist), to: kpt(KPT.leftMiddleBase), depth: 1 },
  { bone: 'rightHand', from: kpt(KPT.rightWrist), to: kpt(KPT.rightMiddleBase), depth: 1 },
  // NOTE: no 'spine' segment — the Meshy 'Spine' bone's child sits BELOW it
  // (Spine→Spine01 = -y), so aiming it up is a ~180° flip → degenerate chest roll
  // (one shoulder up/down, see §B). Torso lean is driven via Hips instead.
  // NOTE: neck is NOT here — driven by _aimHead (full basis: head-up + face-right
  // for head yaw/roll), not a plain direction aim.
];

// Torso-lean depth: z (monocular fwd/back) is the noisy axis and on the long
// hip→shoulder baseline a little z-jitter tilts the whole upper body. Keep it low
// — vertical + side lean stay responsive, fwd/back is damped (raise via depthScale).
export const HIPS_DEPTH = 0.12;
// Neck-pitch depth: short shoulder→head baseline amplifies z-noise → bobbing head.
export const NECK_DEPTH = 0.3;

// Torso bend is distributed across the spine as a fractional WORLD rotation per
// bone (NOT per-bone direction-aim — Spine's child is -y, that path 180°-flips,
// §B3). Weights are cumulative to 1 at the top, so the upper body fully follows
// the target while each vertebra shares the bend. Hips takes a small share so the
// pelvis barely tilts (legs stay grounded); the curl lives in the spine.
export const SPINE_CHAIN = [
  { key: 'hips', weight: 0.15 },
  { key: 'spine', weight: 0.28 },
  { key: 'spine01', weight: 0.28 },
  { key: 'spine02', weight: 0.29 }
];

// Clavicles: aim each shoulder bone from its rest (clavicle) direction toward the
// observed shoulder joint relative to the shoulder-center → shrug (Y) + slight
// protraction. Subtle — driven at reduced gain. z damped (monocular).
export const CLAVICLES = [
  { bone: 'leftShoulder', joint: KPT.leftShoulder, depth: 0.2 },
  { bone: 'rightShoulder', joint: KPT.rightShoulder, depth: 0.2 }
];

// Resolve logical key → THREE.Bone from a skeleton. Throws on any missing bone —
// a rig lacking a mapped bone is a real error, not something to paper over.
export function resolveRigBones(skeleton) {
  const byName = new Map(skeleton.bones.map((b) => [b.name, b]));
  const rig = {};
  for (const [key, name] of Object.entries(MESHY_BONES)) {
    const bone = byName.get(name);
    if (!bone) throw new Error(`rig bone "${name}" (${key}) not in skeleton`);
    rig[key] = bone;
  }
  return rig;
}
