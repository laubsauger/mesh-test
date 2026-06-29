// HumanoidRetargeter (T12): aim each rig bone along its canonical segment
// direction. Rotation-only (ignores scale) onto the SHARED batch skeleton; the
// caller snapshots the resulting boneMatrices into the performer's slice (V5,
// V20). First-pass segment alignment — swing-twist + joint limits are follow-ups.
//
// IMPORTANT: never call skeleton.pose(). Its bind is reconstructed from
// boneInverses in the ORIGINAL (pre-normalizeModel) space, so on a normalized
// source it injects a residual scale (mesh shrinks ~0.01 → vanishes, see §B).
// Rest is restored from the rest LOCAL quaternions captured at load, which live
// in the normalized source space.
import * as THREE from 'three';
import { resolveRigBones, SEGMENTS, hipCenter, shoulderCenter, HIPS_DEPTH } from './rig-map.js';

const _boneW = new THREE.Vector3();
const _childW = new THREE.Vector3();
const _target = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _qParent = new THREE.Quaternion();
const _qLocal = new THREE.Quaternion();
const _qDelta = new THREE.Quaternion();
const DEG2RAD = Math.PI / 180;

export class Retargeter {
  constructor(skeleton, { basis = null, restQuats = null } = {}) {
    this.skeleton = skeleton;
    this.rig = resolveRigBones(skeleton);
    this.basis = basis; // optional THREE.Quaternion: canonical-space → rig-space
    this.restQuats = restQuats; // rest LOCAL quats (normalized source space), per bone
    this.bind = new Map(); // boneKey → { restDirWorld, bindWorldQuat, restLocalQuat }
    this.smoothed = new Map(); // boneKey → persistent slerped local quat (T10·2)
    this._captureBind();
  }

  // Restore bones to their rest LOCAL rotations (NOT skeleton.pose()).
  restPose() {
    if (!this.restQuats) return;
    const bones = this.skeleton.bones;
    for (let i = 0; i < bones.length; i += 1) bones[i].quaternion.copy(this.restQuats[i]);
  }

  // Capture rest world direction (bone → child) + bind world orientation for a
  // bone, given which child defines its "forward". Returns false if degenerate.
  _captureBone(boneKey, childBone) {
    const bone = this.rig[boneKey];
    if (!bone || !childBone) return false;
    _boneW.setFromMatrixPosition(bone.matrixWorld);
    _childW.setFromMatrixPosition(childBone.matrixWorld);
    const restDirWorld = _childW.clone().sub(_boneW);
    if (restDirWorld.lengthSq() < 1e-10) return false;
    restDirWorld.normalize();
    const bindWorldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(bindWorldQuat);
    // rest LOCAL rotation (bones are at rest here) — reference for joint limits.
    this.bind.set(boneKey, { restDirWorld, bindWorldQuat, restLocalQuat: bone.quaternion.clone() });
    return true;
  }

  _captureBind() {
    this.restPose();
    for (const bone of this.skeleton.bones) bone.updateWorldMatrix(true, false);

    // Pelvis "forward" axis = up the spine (Hips → Spine). Driving it leans the
    // whole torso (the forward/side bend).
    this._captureBone('hips', this.rig.spine);

    for (const seg of SEGMENTS) {
      const bone = this.rig[seg.bone];
      this._captureBone(seg.bone, bone?.children.find((c) => c.isBone));
    }
  }

  // Aim a captured bone from world dir (to - from). from/to: {x,y,z,confidence}.
  // depthScale damps the noisy z per-bone (arms full, torso/legs/head low).
  // maxAngleDeg clamps rotation from rest (joint limit, T13). follow is the
  // per-render-frame slerp rate toward the target (T10·2) — decouples render
  // smoothness from the slow pose-fps (no stop-motion) and damps jitter.
  _aim(boneKey, from, to, kptThresh, mirrorX, depthScale, maxAngleDeg, follow) {
    const bind = this.bind.get(boneKey);
    if (!bind || !from || !to || from.confidence < kptThresh || to.confidence < kptThresh) return;

    _target.set(to.x - from.x, to.y - from.y, (to.z - from.z) * depthScale);
    if (_target.lengthSq() < 1e-8) return;
    _target.normalize();
    if (mirrorX) _target.x = -_target.x; // selfie video is X-mirrored

    _q.setFromUnitVectors(bind.restDirWorld, _target).multiply(bind.bindWorldQuat);

    const bone = this.rig[boneKey];
    if (bone.parent) {
      bone.parent.getWorldQuaternion(_qParent).invert();
      _qLocal.copy(_qParent.multiply(_q));
    } else {
      _qLocal.copy(_q);
    }

    // Joint limit: clamp the rotation away from rest to maxAngleDeg.
    if (maxAngleDeg < 180) {
      _qDelta.copy(bind.restLocalQuat).invert().multiply(_qLocal);
      const angle = 2 * Math.acos(Math.min(1, Math.abs(_qDelta.w)));
      const maxRad = maxAngleDeg * DEG2RAD;
      if (angle > maxRad && angle > 1e-4) {
        _qLocal.copy(bind.restLocalQuat).slerp(_qLocal, maxRad / angle);
      }
    }

    // Render-rate interpolation: slerp persistent state toward the target.
    let cur = this.smoothed.get(boneKey);
    if (!cur) {
      cur = bind.restLocalQuat.clone();
      this.smoothed.set(boneKey, cur);
    }
    cur.slerp(_qLocal, follow >= 1 ? 1 : follow);
    bone.quaternion.copy(cur);
    bone.updateWorldMatrix(false, false);
  }

  // canonical: { joints: [{x,y,z,confidence}, ...] } in canonical space.
  apply(canonical, { kptThresh = 0.3, mirrorX = true, depthScale = 1, jointLimitDeg = 180, follow = 1 } = {}) {
    this.restPose(); // start from rest each frame so untracked bones stay at rest

    // Pelvis first (root) — leans the whole body; limbs below compensate via
    // their parent's world orientation. Damped depth (avoids forward droop).
    this._aim('hips', hipCenter(canonical), shoulderCenter(canonical), kptThresh, mirrorX, depthScale * HIPS_DEPTH, jointLimitDeg, follow);

    for (const seg of SEGMENTS) {
      this._aim(
        seg.bone, seg.from(canonical), seg.to(canonical),
        kptThresh, mirrorX, depthScale * (seg.depth ?? 1), seg.maxAngle ?? jointLimitDeg, follow
      );
    }
  }
}
