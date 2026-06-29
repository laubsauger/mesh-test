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

// Directional bones: rig bone aimed from `from(canonical)` toward `to(canonical)`.
export const SEGMENTS = [
  { bone: 'leftArm', from: kpt(KPT.leftShoulder), to: kpt(KPT.leftElbow) },
  { bone: 'leftForeArm', from: kpt(KPT.leftElbow), to: kpt(KPT.leftWrist) },
  { bone: 'rightArm', from: kpt(KPT.rightShoulder), to: kpt(KPT.rightElbow) },
  { bone: 'rightForeArm', from: kpt(KPT.rightElbow), to: kpt(KPT.rightWrist) },
  { bone: 'leftUpLeg', from: kpt(KPT.leftHip), to: kpt(KPT.leftKnee) },
  { bone: 'leftLeg', from: kpt(KPT.leftKnee), to: kpt(KPT.leftAnkle) },
  { bone: 'rightUpLeg', from: kpt(KPT.rightHip), to: kpt(KPT.rightKnee) },
  { bone: 'rightLeg', from: kpt(KPT.rightKnee), to: kpt(KPT.rightAnkle) },
  { bone: 'leftFoot', from: kpt(KPT.leftAnkle), to: kpt(KPT.leftBigToe) },
  { bone: 'rightFoot', from: kpt(KPT.rightAnkle), to: kpt(KPT.rightBigToe) },
  // NOTE: no 'spine' segment — the Meshy 'Spine' bone's child sits BELOW it
  // (Spine→Spine01 = -y), so aiming it up is a ~180° flip → degenerate chest roll
  // (one shoulder up/down, see §B). Torso lean is driven via Hips instead.
  { bone: 'neck', from: shoulderCenter, to: kpt(KPT.nose) }
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
