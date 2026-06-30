// HumanoidRetargeter (T12): drive the rig from a canonical pose. Upper limb bones
// get full swing-TWIST via a bend-plane basis (§22); other bones are direction-
// aimed (swing only). Rotation-only onto the SHARED batch skeleton; the caller
// snapshots the resulting boneMatrices into the performer's slice (V5, V20).
//
// IMPORTANT: never call skeleton.pose(). Its bind is reconstructed from
// boneInverses in the ORIGINAL (pre-normalizeModel) space, so on a normalized
// source it injects a residual scale (mesh shrinks ~0.01 → vanishes, see §B).
// Rest is restored from rest LOCAL quaternions captured at load.
import * as THREE from 'three';
import { resolveRigBones, SEGMENTS, LIMBS, FOREARMS, hipCenter, shoulderCenter, headCenter, HIPS_DEPTH } from './rig-map.js';
import { KPT } from './rtmw-constants.js';

const _xAxis = new THREE.Vector3(1, 0, 0);

const _boneW = new THREE.Vector3();
const _childW = new THREE.Vector3();
const _endW = new THREE.Vector3();
const _target = new THREE.Vector3();
const _rootP = new THREE.Vector3();
const _midP = new THREE.Vector3();
const _endP = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _planeN = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _qParent = new THREE.Quaternion();
const _qLocal = new THREE.Quaternion();
const _qDelta = new THREE.Quaternion();
const _mBind = new THREE.Matrix4();
const _mTarget = new THREE.Matrix4();
const DEG2RAD = Math.PI / 180;

// Rotation mapping orthonormal-ish basis (a1,a2) → (b1,b2), written to `out`.
const _a2o = new THREE.Vector3();
const _a3 = new THREE.Vector3();
const _b2o = new THREE.Vector3();
const _b3 = new THREE.Vector3();
function basisQuat(a1, a2, b1, b2, out) {
  _a2o.copy(a2).addScaledVector(a1, -a1.dot(a2)).normalize();
  _a3.crossVectors(a1, _a2o);
  _b2o.copy(b2).addScaledVector(b1, -b1.dot(b2)).normalize();
  _b3.crossVectors(b1, _b2o);
  _mBind.makeBasis(a1, _a2o, _a3).transpose(); // orthonormal → inverse = transpose
  _mTarget.makeBasis(b1, _b2o, _b3).multiply(_mBind); // target * bind⁻¹
  out.setFromRotationMatrix(_mTarget);
}

// Singularity-safe smoothing of a persistent unit axis toward a new candidate.
// `last.lerp(candidate).normalize()` SNAPS when candidate ≈ -last: the lerp
// passes through ~zero magnitude and normalize() amplifies noise into a random
// direction (the 90/180° flip). Mode picks the sign policy:
//   'align' — SIGN-AGNOSTIC axis (bend-plane normals): hemisphere-align candidate
//             (negate if backward) so it never collapses; sign is meaningless.
//   'hold'  — SIGN-IS-THE-SIGNAL axis (body facing/yaw): a backward candidate is a
//             monocular depth-sign FLIP, not a real turn. REJECT it (hold last) so
//             noise can't sweep the body 180°. Only same-hemisphere candidates blend.
const _blend = new THREE.Vector3();
function smoothAxis(last, candidate, follow, mode) {
  const backward = candidate.dot(last) < 0;
  if (mode === 'align' && backward) candidate.negate();
  else if (mode === 'hold' && backward) return last; // depth-sign flip → hold facing
  _blend.copy(last).lerp(candidate, follow);
  if (_blend.lengthSq() < 1e-4) return last; // collapsed → hold, no snap
  return last.copy(_blend).normalize();
}

export class Retargeter {
  constructor(skeleton, { restQuats = null } = {}) {
    this.skeleton = skeleton;
    this.rig = resolveRigBones(skeleton);
    this.restQuats = restQuats; // rest LOCAL quats (normalized source space), per bone
    this.bind = new Map(); // boneKey → { restDirWorld, bindWorldQuat, restLocalQuat, bindPlaneN? }
    this.smoothed = new Map(); // boneKey → persistent slerped local quat (T10·2)
    this.lastPlaneN = new Map(); // limb → last good bend-plane normal (pole stability)
    this.clampStats = new Map(); // V19: boneKey → { raw°, max°, clamped } last frame
    this.clampPeak = new Map(); // V19: boneKey → worst { raw°, max°, over° } since reset
    this._captureBind();
  }

  restPose() {
    if (!this.restQuats) return;
    const bones = this.skeleton.bones;
    for (let i = 0; i < bones.length; i += 1) bones[i].quaternion.copy(this.restQuats[i]);
  }

  _captureBone(boneKey, childBone) {
    const bone = this.rig[boneKey];
    if (!bone || !childBone) return null;
    _boneW.setFromMatrixPosition(bone.matrixWorld);
    _childW.setFromMatrixPosition(childBone.matrixWorld);
    const restDirWorld = _childW.clone().sub(_boneW);
    if (restDirWorld.lengthSq() < 1e-10) return null;
    restDirWorld.normalize();
    const bindWorldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(bindWorldQuat);
    const entry = { restDirWorld, bindWorldQuat, restLocalQuat: bone.quaternion.clone() };
    this.bind.set(boneKey, entry);
    return entry;
  }

  _captureBind() {
    this.restPose();
    for (const bone of this.skeleton.bones) bone.updateWorldMatrix(true, false);

    // Pelvis: rest dir = up the spine; bindAcross = its "right" axis (for yaw).
    const hipsEntry = this._captureBone('hips', this.rig.spine);
    if (hipsEntry) {
      _endP.copy(_xAxis).applyQuaternion(hipsEntry.bindWorldQuat);
      _endP.addScaledVector(hipsEntry.restDirWorld, -hipsEntry.restDirWorld.dot(_endP));
      if (_endP.lengthSq() > 1e-8) hipsEntry.bindAcross = _endP.clone().normalize();
    }

    // Pelvis vertical (squat): rest Hips local Y + the rig's standing leg span
    // (hip→foot world height) → maps canonical pelvis-drop to rig units.
    this.hipsRestPosY = this.rig.hips.position.y;
    _boneW.setFromMatrixPosition(this.rig.leftFoot.matrixWorld);
    _childW.setFromMatrixPosition(this.rig.rightFoot.matrixWorld);
    this.restFootMinY = Math.min(_boneW.y, _childW.y); // rest floor height (WORLD, grounding anchor)
    // foot Y is WORLD (incl normalizeModel scale); hips.position is LOCAL — convert
    // the world ground delta to hips-local via the parent's world scale.
    _endP.setFromMatrixScale(this.rig.hips.parent.matrixWorld);
    this.hipsParentScaleY = _endP.y || 1;
    this.groundOffsetY = 0;

    for (const seg of SEGMENTS) {
      this._captureBone(seg.bone, this.rig[seg.bone]?.children.find((c) => c.isBone));
    }

    // Forearms: rest dir (forearm→hand) + a bind "across" axis (world-x projected
    // perpendicular to the bone) as the roll reference for hand-knuckle twist.
    for (const fa of FOREARMS) {
      const entry = this._captureBone(fa.bone, this.rig[fa.bone]?.children.find((c) => c.isBone));
      if (!entry) continue;
      _endP.copy(_xAxis).applyQuaternion(entry.bindWorldQuat);
      _endP.addScaledVector(entry.restDirWorld, -entry.restDirWorld.dot(_endP));
      if (_endP.lengthSq() > 1e-8) entry.bindAcross = _endP.clone().normalize();
    }

    // Neck: rest dir up + bindAcross "right" (for head yaw/roll via the eye line).
    const neckEntry = this._captureBone('neck', this.rig.head);
    if (neckEntry) {
      _endP.copy(_xAxis).applyQuaternion(neckEntry.bindWorldQuat);
      _endP.addScaledVector(neckEntry.restDirWorld, -neckEntry.restDirWorld.dot(_endP));
      if (_endP.lengthSq() > 1e-8) neckEntry.bindAcross = _endP.clone().normalize();
    }

    // Hands are leaf bones (no child) — derive rest direction from forearm→hand
    // (the hand's extension axis), aimed at wrist→middle-finger.
    for (const handKey of ['leftHand', 'rightHand']) {
      const bone = this.rig[handKey];
      if (!bone?.parent?.isBone) continue;
      _boneW.setFromMatrixPosition(bone.matrixWorld);
      _childW.setFromMatrixPosition(bone.parent.matrixWorld);
      const restDirWorld = _boneW.clone().sub(_childW); // forearm → hand
      if (restDirWorld.lengthSq() < 1e-10) continue;
      restDirWorld.normalize();
      const bindWorldQuat = new THREE.Quaternion();
      bone.getWorldQuaternion(bindWorldQuat);
      this.bind.set(handKey, { restDirWorld, bindWorldQuat, restLocalQuat: bone.quaternion.clone() });
    }

    // Upper limbs: also capture the bind bend-plane normal from the rest rig
    // (upper → mid → end bone positions) for swing-twist.
    for (const limb of LIMBS) {
      const midBone = this.rig[limb.midBone];
      const endBone = this.rig[limb.endBone];
      const entry = this._captureBone(limb.upper, midBone);
      if (!entry || !endBone) continue;
      _boneW.setFromMatrixPosition(this.rig[limb.upper].matrixWorld);
      _childW.setFromMatrixPosition(midBone.matrixWorld);
      _endW.setFromMatrixPosition(endBone.matrixWorld);
      _dir.copy(_childW).sub(_boneW);
      _planeN.copy(_endW).sub(_childW).cross(_dir); // (end-mid) × (mid-root)
      if (_planeN.lengthSq() > 1e-10) entry.bindPlaneN = _planeN.clone().normalize();
    }
  }

  // Set bone local from a target WORLD quat: parent-relative, gain scale, joint-
  // limit clamp, render-rate slerp (T10·2). gain<1 scales the rotation away from
  // rest (tames over-eager bones, e.g. head pitch).
  _setBoneFromWorld(boneKey, bind, qWorld, maxAngleDeg, follow, gain = 1) {
    // Singularity guard: a degenerate aim (basisQuat with a near-parallel basis,
    // or setFromUnitVectors on a ~zero/antiparallel vector) yields a NaN quat.
    // Writing it would NaN this bone's matrix → the whole shared skinned batch
    // vanishes. Skip the update — the bone holds its last good smoothed value.
    if (!Number.isFinite(qWorld.x) || !Number.isFinite(qWorld.y) ||
        !Number.isFinite(qWorld.z) || !Number.isFinite(qWorld.w)) return;
    const bone = this.rig[boneKey];
    if (bone.parent) {
      bone.parent.getWorldQuaternion(_qParent).invert();
      _qLocal.copy(_qParent.multiply(qWorld));
    } else {
      _qLocal.copy(qWorld);
    }

    if (gain < 1) _qLocal.copy(bind.restLocalQuat).slerp(_qLocal, gain);

    if (maxAngleDeg < 180) {
      _qDelta.copy(bind.restLocalQuat).invert().multiply(_qLocal);
      const angle = 2 * Math.acos(Math.min(1, Math.abs(_qDelta.w)));
      const maxRad = maxAngleDeg * DEG2RAD;
      const clamped = angle > maxRad && angle > 1e-4;
      if (clamped) _qLocal.copy(bind.restLocalQuat).slerp(_qLocal, maxRad / angle);
      // V19: how far this bone WANTED to rotate vs the cap it hit (deg).
      const rawDeg = angle / DEG2RAD;
      this.clampStats.set(boneKey, { raw: rawDeg, max: maxAngleDeg, clamped });
      // Peak-hold: the live per-frame value flickers too fast to read, so keep the
      // worst overflow per bone until reset — a momentary face-touch stays legible.
      if (clamped) {
        const p = this.clampPeak.get(boneKey);
        const over = rawDeg - maxAngleDeg;
        if (!p || over > p.over) this.clampPeak.set(boneKey, { raw: rawDeg, max: maxAngleDeg, over });
      }
    }

    let cur = this.smoothed.get(boneKey);
    if (!cur) {
      cur = bind.restLocalQuat.clone();
      this.smoothed.set(boneKey, cur);
    }
    cur.slerp(_qLocal, follow >= 1 ? 1 : follow);
    bone.quaternion.copy(cur);
    bone.updateWorldMatrix(false, false);
  }

  // V19: peak-held clamp overflow per bone since last reset, worst first. Row:
  // { bone, raw°, max°, over° }. Peak-held so a brief gesture stays readable.
  clampReport() {
    const rows = [];
    for (const [bone, s] of this.clampPeak) rows.push({ bone, raw: s.raw, max: s.max, over: s.over });
    rows.sort((a, b) => b.over - a.over);
    return rows;
  }

  resetClampPeak() { this.clampPeak.clear(); }

  // Direction-only (swing) aim. from/to: {x,y,z,confidence}.
  _aim(boneKey, from, to, opts) {
    const bind = this.bind.get(boneKey);
    if (!bind || !from || !to || from.confidence < opts.kptThresh || to.confidence < opts.kptThresh) return;
    _target.set(to.x - from.x, to.y - from.y, (to.z - from.z) * opts.depth);
    if (_target.lengthSq() < 1e-8) return;
    _target.normalize();
    if (opts.mirrorX) _target.x = -_target.x;
    _q.setFromUnitVectors(bind.restDirWorld, _target).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(boneKey, bind, _q, opts.maxAngleDeg, opts.follow, opts.gain ?? 1);
  }

  // Upper-limb aim with swing-twist: orient by bone direction AND bend-plane.
  _aimLimb(limb, canonical, opts) {
    const bind = this.bind.get(limb.upper);
    if (!bind) return;
    const r = canonical.joints[limb.root];
    const m = canonical.joints[limb.mid];
    const e = canonical.joints[limb.end];
    if (!r || !m || r.confidence < opts.kptThresh || m.confidence < opts.kptThresh) return;

    const sx = opts.mirrorX ? -1 : 1;
    _rootP.set(r.x * sx, r.y, r.z * limb.depth);
    _midP.set(m.x * sx, m.y, m.z * limb.depth);
    _dir.copy(_midP).sub(_rootP);
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    const canTwist = opts.swingTwist && bind.bindPlaneN && e && e.confidence >= opts.kptThresh;
    if (canTwist) {
      _endP.set(e.x * sx, e.y, e.z * limb.depth);
      _planeN.copy(_endP).sub(_midP).cross(_dir); // (end-mid) × (dir)
      const planeLen = _planeN.length();
      // Pole-vector stability + smoothing: the bend normal depends on the wrist,
      // so wrist/forearm jitter would roll the UPPER arm ("elbow spins when I turn
      // my wrist"). Reuse the last plane when near-straight, and heavily smooth it
      // (lerp) so twist follows gross changes, not per-frame wrist noise.
      let plane = null;
      if (planeLen > 0.04) {
        _planeN.divideScalar(planeLen);
        let last = this.lastPlaneN.get(limb.upper);
        if (!last) {
          last = _planeN.clone();
          this.lastPlaneN.set(limb.upper, last);
        } else {
          smoothAxis(last, _planeN, opts.planeFollow, 'align'); // bend-plane sign carries no info
        }
        plane = last;
      } else {
        plane = this.lastPlaneN.get(limb.upper) ?? null;
      }
      if (plane) {
        basisQuat(bind.restDirWorld, bind.bindPlaneN, _dir, plane, _q);
        _q.multiply(bind.bindWorldQuat);
        this._setBoneFromWorld(limb.upper, bind, _q, opts.maxAngleDeg, opts.follow);
        return;
      }
    }
    // fallback: swing only
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(limb.upper, bind, _q, opts.maxAngleDeg, opts.follow);
  }

  // Forearm: aim elbow→wrist (swing) + optional axial twist from the hand knuckle
  // line (indexMCP→pinkyMCP), so pronation/supination rotates the forearm.
  _aimForeArm(fa, canonical, opts) {
    const bind = this.bind.get(fa.bone);
    if (!bind) return;
    const r = canonical.joints[fa.root];
    const m = canonical.joints[fa.mid];
    if (!r || !m || r.confidence < opts.kptThresh || m.confidence < opts.kptThresh) return;
    const sx = opts.mirrorX ? -1 : 1;
    _rootP.set(r.x * sx, r.y, r.z * opts.depth);
    _midP.set(m.x * sx, m.y, m.z * opts.depth);
    _dir.copy(_midP).sub(_rootP);
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    if (opts.wristTwist && bind.bindAcross) {
      const a = canonical.joints[fa.rollA];
      const b = canonical.joints[fa.rollB];
      if (a && b && a.confidence >= opts.kptThresh && b.confidence >= opts.kptThresh) {
        _planeN.set((b.x - a.x) * sx, b.y - a.y, (b.z - a.z) * opts.depth);
        _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // perpendicular to forearm
        const len = _planeN.length();
        if (len > 0.02) {
          _planeN.divideScalar(len);
          let last = this.lastPlaneN.get(fa.bone);
          if (!last) { last = _planeN.clone(); this.lastPlaneN.set(fa.bone, last); }
          else smoothAxis(last, _planeN, opts.planeFollow, 'align'); // forearm-roll plane sign carries no info
          basisQuat(bind.restDirWorld, bind.bindAcross, _dir, last, _q);
          _q.multiply(bind.bindWorldQuat);
          this._setBoneFromWorld(fa.bone, bind, _q, opts.maxAngleDeg, opts.follow);
          return;
        }
      }
    }
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(fa.bone, bind, _q, opts.maxAngleDeg, opts.follow);
  }

  // Pelvis with optional YAW: up→torso-up (lean) + right→hip-axis (turning), so
  // the whole upper body rotates when you turn. Falls back to lean-only.
  _aimHips(canonical, opts) {
    const bind = this.bind.get('hips');
    if (!bind) return;
    const hc = hipCenter(canonical);
    const sc = shoulderCenter(canonical);
    const lh = canonical.joints[KPT.leftHip];
    const rh = canonical.joints[KPT.rightHip];
    if (hc.confidence < opts.kptThresh || sc.confidence < opts.kptThresh || !lh || !rh) return;
    const sx = opts.mirrorX ? -1 : 1;
    _dir.set((sc.x - hc.x) * sx, sc.y - hc.y, (sc.z - hc.z) * opts.depth); // torso up (lean)
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    if (opts.yaw && bind.bindAcross && lh.confidence >= opts.kptThresh && rh.confidence >= opts.kptThresh) {
      // hip axis (right). Turning lives in the z-separation, which monocular depth
      // under-reads → amplify z by yawGain so turns register. But near frontal that
      // z is mostly NOISE: amplified, it flips the axis sign frame-to-frame → the
      // whole body snaps 180°. Deadzone it: ignore depth separation smaller than a
      // fraction of the (reliable) horizontal hip width, so frontal pose stays put.
      const rxw = (rh.x - lh.x) * sx;
      const rzAmp = (rh.z - lh.z) * opts.yawGain;
      const rz = Math.abs(rzAmp) < Math.abs(rxw) * 0.18 ? 0 : rzAmp;
      _planeN.set(rxw, rh.y - lh.y, rz);
      _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // perpendicular to up
      const len = _planeN.length();
      if (len > 0.02) {
        _planeN.divideScalar(len);
        let last = this.lastPlaneN.get('hips');
        if (!last) { last = _planeN.clone(); this.lastPlaneN.set('hips', last); }
        else smoothAxis(last, _planeN, Math.max(opts.planeFollow, 0.3), 'hold'); // sign = facing; reject depth-sign flips
        // Degeneracy guard: if the (held) across axis is near-parallel to torso-up,
        // basisQuat's orthogonalization is unstable → random/180° spin. Skip yaw and
        // fall through to lean-only below (a defined no-yaw orientation, still slerp-
        // smoothed by _setBoneFromWorld — no snap).
        if (Math.abs(last.dot(_dir)) < 0.94) {
          basisQuat(bind.restDirWorld, bind.bindAcross, _dir, last, _q);
          _q.multiply(bind.bindWorldQuat);
          this._setBoneFromWorld('hips', bind, _q, opts.maxAngleDeg, opts.follow);
          return;
        }
      }
    }
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat); // lean only
    this._setBoneFromWorld('hips', bind, _q, opts.maxAngleDeg, opts.follow);
  }

  // Head/neck: pitch from head-up direction, + YAW from the nose's horizontal
  // offset vs the ear midpoint (a robust 2D signal — head turn shifts the nose
  // sideways between the ears; no reliance on weak depth).
  _aimHead(canonical, opts) {
    const bind = this.bind.get('neck');
    if (!bind) return;
    const sc = shoulderCenter(canonical);
    const hcen = headCenter(canonical);
    if (sc.confidence < opts.kptThresh || hcen.confidence < opts.kptThresh) return;
    const sx = opts.mirrorX ? -1 : 1;
    _dir.set((hcen.x - sc.x) * sx, hcen.y - sc.y, (hcen.z - sc.z) * opts.depth);
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat); // pitch

    const nose = canonical.joints[KPT.nose];
    const le = canonical.joints[KPT.leftEar];
    const re = canonical.joints[KPT.rightEar];
    if (opts.yaw && nose && le && re && nose.confidence >= opts.kptThresh && le.confidence >= opts.kptThresh && re.confidence >= opts.kptThresh) {
      const earMidX = (le.x + re.x) / 2;
      const earW = Math.abs(re.x - le.x) || 0.1;
      let yawA = (-(nose.x - earMidX) / earW) * sx * opts.yawGain * 0.6;
      yawA = Math.max(-1.2, Math.min(1.2, yawA)); // clamp ±~70°
      _qParent.setFromAxisAngle(_dir, yawA); // yaw about the head's up axis
      _q.premultiply(_qParent);
    }
    this._setBoneFromWorld('neck', bind, _q, opts.maxAngleDeg, opts.follow, opts.gain);
  }

  apply(canonical, { kptThresh = 0.3, mirrorX = true, depthScale = 1, jointLimitDeg = 180, armLimit = 180, follow = 1, swingTwist = true, headGain = 1, planeFollow = 0.12, wristTwist = false, grounding = false, groundFollow = 0.3, bodyYaw = true, yawGain = 2.5 } = {}) {
    this.restPose();
    this.rig.hips.position.y = this.hipsRestPosY; // reset (restPose only touches rotations)
    this.clampStats.clear(); // V19: only THIS frame's aimed bones report clamps
    const base = { kptThresh, mirrorX, follow, maxAngleDeg: jointLimitDeg, swingTwist, planeFollow };

    // Pelvis rotation (leans whole body); upper limbs next (children compensate);
    // forearms (with optional twist); then lower bones / feet / hands / neck. Arms
    // use a tighter limit (over-rotation guard).
    this._aimHips(canonical, { ...base, depth: depthScale * HIPS_DEPTH, yaw: bodyYaw, yawGain });
    for (const limb of LIMBS) {
      const isArm = limb.upper.includes('Arm');
      this._aimLimb(limb, canonical, { ...base, depth: depthScale * limb.depth, maxAngleDeg: isArm ? armLimit : jointLimitDeg });
    }
    for (const fa of FOREARMS) this._aimForeArm(fa, canonical, { ...base, depth: depthScale * fa.depth, wristTwist, maxAngleDeg: armLimit });
    this._aimHead(canonical, { ...base, depth: depthScale * 0.85, yawGain, gain: headGain });
    for (const seg of SEGMENTS) {
      this._aim(seg.bone, seg.from(canonical), seg.to(canonical), {
        ...base,
        depth: depthScale * (seg.depth ?? 1),
        maxAngleDeg: seg.maxAngle ?? jointLimitDeg
      });
    }

    // Foot-anchored grounding (§24/§25): the pose just bent the legs with the
    // pelvis at rest height → the feet moved off the floor. Offset the Hips so the
    // LOWEST (support) foot returns to its rest floor height — so squats lower the
    // body with feet planted, instead of the pelvis floating + legs pulling up.
    if (grounding) {
      // Refresh the whole hips subtree: foot bones are world-updated only when
      // their SEGMENTS aim runs, which is SKIPPED on low foot/toe confidence —
      // leaving foot.matrixWorld stale relative to the moved upper-leg/knee.
      // Reading a stale foot Y here makes grounding bob/sink through the floor.
      this.rig.hips.updateWorldMatrix(false, true);
      _boneW.setFromMatrixPosition(this.rig.leftFoot.matrixWorld);
      _childW.setFromMatrixPosition(this.rig.rightFoot.matrixWorld);
      const lowestFootY = Math.min(_boneW.y, _childW.y);
      const targetDelta = this.restFootMinY - lowestFootY; // + to lift body so foot sits on floor
      // Symmetric low-pass both directions. (An earlier asymmetric "instant lift"
      // clamp guaranteed no floor-clip but turned monocular foot-Y noise into a
      // sawtooth bob — the cure was worse. Smooth evenly; tune via groundFollow.)
      this.groundOffsetY += (targetDelta - this.groundOffsetY) * groundFollow;
      this.rig.hips.position.y = this.hipsRestPosY + this.groundOffsetY / this.hipsParentScaleY; // world→local
    } else {
      this.groundOffsetY = 0;
    }
  }
}
