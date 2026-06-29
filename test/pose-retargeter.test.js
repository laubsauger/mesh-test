import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { Retargeter } from '../src/pose/retargeter.js';
import { KPT, NUM_KPTS } from '../src/pose/rtmw-constants.js';
import { resolveRigBones } from '../src/pose/rig-map.js';

// Build the real 24-bone Meshy hierarchy with plausible local offsets so bone
// directions are non-degenerate. Mirrors the rig in every shipped GLB.
function buildMeshySkeleton() {
  const mk = (name, pos, parent) => {
    const b = new THREE.Bone();
    b.name = name;
    b.position.set(...pos);
    if (parent) parent.add(b);
    return b;
  };
  const hips = mk('Hips', [0, 1.0, 0], null);
  const spine = mk('Spine', [0, 0.2, 0], hips);
  const spine01 = mk('Spine01', [0, 0.2, 0], spine);
  const spine02 = mk('Spine02', [0, 0.2, 0], spine01);
  const neck = mk('neck', [0, 0.15, 0], spine02);
  const head = mk('Head', [0, 0.15, 0], neck);
  mk('head_end', [0, 0.1, 0], head);
  mk('headfront', [0, 0.05, 0.05], head);

  const lsh = mk('LeftShoulder', [0.05, 0.1, 0], spine02);
  const lar = mk('LeftArm', [0.15, 0, 0], lsh);
  const lfa = mk('LeftForeArm', [0.25, 0, 0], lar);
  mk('LeftHand', [0.22, 0, 0], lfa);
  const rsh = mk('RightShoulder', [-0.05, 0.1, 0], spine02);
  const rar = mk('RightArm', [-0.15, 0, 0], rsh);
  const rfa = mk('RightForeArm', [-0.25, 0, 0], rar);
  mk('RightHand', [-0.22, 0, 0], rfa);

  const lul = mk('LeftUpLeg', [0.1, -0.05, 0], hips);
  const lll = mk('LeftLeg', [0, -0.45, 0], lul);
  const lfo = mk('LeftFoot', [0, -0.45, 0], lll);
  mk('LeftToeBase', [0, -0.05, 0.12], lfo);
  const rul = mk('RightUpLeg', [-0.1, -0.05, 0], hips);
  const rll = mk('RightLeg', [0, -0.45, 0], rul);
  const rfo = mk('RightFoot', [0, -0.45, 0], rll);
  mk('RightToeBase', [0, -0.05, 0.12], rfo);

  const root = new THREE.Object3D();
  root.add(hips);
  root.updateMatrixWorld(true);

  const bones = [];
  hips.traverse((o) => { if (o.isBone) bones.push(o); });
  const skeleton = new THREE.Skeleton(bones);
  return { root, skeleton };
}

function synthCanonical() {
  const joints = Array.from({ length: NUM_KPTS }, () => ({ x: 0, y: 0, z: 0, confidence: 0.9 }));
  joints[KPT.leftShoulder] = { x: 0.2, y: 0.5, z: 0, confidence: 0.9 };
  joints[KPT.leftElbow] = { x: 0.35, y: 0.3, z: 0, confidence: 0.9 };
  joints[KPT.leftWrist] = { x: 0.45, y: 0.1, z: 0, confidence: 0.9 };
  joints[KPT.rightShoulder] = { x: -0.2, y: 0.5, z: 0, confidence: 0.9 };
  joints[KPT.rightElbow] = { x: -0.35, y: 0.3, z: 0, confidence: 0.9 };
  joints[KPT.rightWrist] = { x: -0.45, y: 0.1, z: 0, confidence: 0.9 };
  joints[KPT.leftHip] = { x: 0.1, y: 0, z: 0, confidence: 0.9 };
  joints[KPT.leftKnee] = { x: 0.1, y: -0.5, z: 0, confidence: 0.9 };
  joints[KPT.leftAnkle] = { x: 0.1, y: -0.95, z: 0, confidence: 0.9 };
  joints[KPT.leftBigToe] = { x: 0.1, y: -1.0, z: 0.1, confidence: 0.9 };
  joints[KPT.rightHip] = { x: -0.1, y: 0, z: 0, confidence: 0.9 };
  joints[KPT.rightKnee] = { x: -0.1, y: -0.5, z: 0, confidence: 0.9 };
  joints[KPT.rightAnkle] = { x: -0.1, y: -0.95, z: 0, confidence: 0.9 };
  joints[KPT.rightBigToe] = { x: -0.1, y: -1.0, z: 0.1, confidence: 0.9 };
  return { timestampMs: 0, joints };
}

const finiteQuat = (q) => [q.x, q.y, q.z, q.w].every(Number.isFinite);

describe('Retargeter — T12: no NaN, valid bone matrices', () => {
  it('apply produces finite bone quaternions', () => {
    const { skeleton } = buildMeshySkeleton();
    const rt = new Retargeter(skeleton);
    rt.apply(synthCanonical());
    for (const bone of skeleton.bones) {
      expect(finiteQuat(bone.quaternion), `bone ${bone.name} quaternion`).toBe(true);
    }
  });

  it('skeleton.update produces finite boneMatrices (mesh stays visible)', () => {
    const { root, skeleton } = buildMeshySkeleton();
    const rt = new Retargeter(skeleton);
    rt.apply(synthCanonical());
    root.updateMatrixWorld(true);
    skeleton.update();
    for (let i = 0; i < skeleton.boneMatrices.length; i++) {
      expect(Number.isFinite(skeleton.boneMatrices[i]), `boneMatrices[${i}]`).toBe(true);
    }
  });

  it('symmetric pose → symmetric rig (L/R arms mirror)', () => {
    const { root, skeleton } = buildMeshySkeleton();
    const restQuats = skeleton.bones.map((b) => b.quaternion.clone());
    const rt = new Retargeter(skeleton, { restQuats });

    // perfectly symmetric arms raised
    const c = synthCanonical();
    c.joints[KPT.leftShoulder] = { x: 0.2, y: 0.5, z: 0, confidence: 0.9 };
    c.joints[KPT.rightShoulder] = { x: -0.2, y: 0.5, z: 0, confidence: 0.9 };
    c.joints[KPT.leftElbow] = { x: 0.4, y: 0.55, z: 0, confidence: 0.9 };
    c.joints[KPT.rightElbow] = { x: -0.4, y: 0.55, z: 0, confidence: 0.9 };
    c.joints[KPT.leftWrist] = { x: 0.6, y: 0.6, z: 0, confidence: 0.9 };
    c.joints[KPT.rightWrist] = { x: -0.6, y: 0.6, z: 0, confidence: 0.9 };
    c.joints[KPT.nose] = { x: 0, y: 0.9, z: 0, confidence: 0.9 };
    rt.apply(c, { kptThresh: 0.3, mirrorX: true });
    root.updateMatrixWorld(true);

    const rig = resolveRigBones(skeleton);
    const dir = (a, b) => new THREE.Vector3().setFromMatrixPosition(rig[b].matrixWorld)
      .sub(new THREE.Vector3().setFromMatrixPosition(rig[a].matrixWorld)).normalize();
    const la = dir('leftArm', 'leftForeArm');
    const ra = dir('rightArm', 'rightForeArm');
    // mirror across X: la.x ≈ -ra.x, la.y ≈ ra.y, la.z ≈ ra.z
    expect(Math.abs(la.x + ra.x), `arm x mirror (la ${la.x.toFixed(3)} ra ${ra.x.toFixed(3)})`).toBeLessThan(0.02);
    expect(Math.abs(la.y - ra.y), `arm y match (la ${la.y.toFixed(3)} ra ${ra.y.toFixed(3)})`).toBeLessThan(0.02);
    expect(Math.abs(la.z - ra.z), `arm z match (la ${la.z.toFixed(3)} ra ${ra.z.toFixed(3)})`).toBeLessThan(0.02);
  });

  it('null/low-confidence joints do not corrupt bones', () => {
    const { skeleton } = buildMeshySkeleton();
    const rt = new Retargeter(skeleton);
    const canon = synthCanonical();
    canon.joints[KPT.leftElbow].confidence = 0; // gated out
    rt.apply(canon, { kptThresh: 0.3 });
    for (const bone of skeleton.bones) expect(finiteQuat(bone.quaternion)).toBe(true);
  });
});
