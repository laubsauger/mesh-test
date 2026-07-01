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
import { resolveRigBones, SEGMENTS, LIMBS, FOREARMS, CLAVICLES, SPINE_CHAIN, hipCenter, shoulderCenter, headCenter } from './rig-map.js';
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
const _qFrac = new THREE.Quaternion(); // torso: fractional world bend Q^f per spine bone
const _qWorldT = new THREE.Quaternion(); // torso: per-bone target world quat
const _qId = new THREE.Quaternion(); // identity (slerp start for Q^f)
const _qTorso = new THREE.Quaternion(); // torso: full world delta Q (rest frame→target)
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
// Soft confidence gate. FULL drive (1) at/above thresh — a confident joint must
// reach its target, not sit half-way toward rest (that under-follows = "pinned"
// near rest). BELOW thresh it fades 1→0 across [thresh-band, thresh] so a dropping
// joint eases toward REST instead of freezing then POPPING on reacquire (what the
// old binary `conf < thresh → return` did).
function confGate(conf, thresh, band = 0.12) {
  if (conf >= thresh) return 1;
  const t = (conf - (thresh - band)) / band; // 0 at thresh-band, 1 at thresh
  return t <= 0 ? 0 : t * t * (3 - 2 * t);
}

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

    // Spine chain: torso bend distributes a world-space rotation across these, so
    // they need only their REST world quat + rest local (no rest dir — Spine's -y
    // child would 180°-flip a direction aim, §B3). hips bind (captured above) is
    // the chain's rest frame reference (restDirWorld = up, bindAcross = right).
    for (const { key } of SPINE_CHAIN) {
      if (key === 'hips') continue; // already captured with dir + across
      const bone = this.rig[key];
      if (!bone) continue;
      const bindWorldQuat = new THREE.Quaternion();
      bone.getWorldQuaternion(bindWorldQuat);
      this.bind.set(key, { bindWorldQuat, restLocalQuat: bone.quaternion.clone() });
    }

    // Clavicles: rest dir = shoulder→arm (the bone's outward axis); aimed toward
    // the observed shoulder joint for shrug/protraction.
    for (const clav of CLAVICLES) {
      this._captureBone(clav.bone, this.rig[clav.bone]?.children.find((c) => c.isBone));
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
      // Peak-hold + hit-count: live value flickers too fast to read, so keep the
      // worst overflow per bone until reset. `hits` (clamped frames) vs the apply
      // count tells a SUSTAINED reach (clamps most frames) from a one-frame
      // inversion spike (peak 180° but hits≈1).
      if (clamped) {
        const p = this.clampPeak.get(boneKey);
        const over = rawDeg - maxAngleDeg;
        if (!p) this.clampPeak.set(boneKey, { raw: rawDeg, max: maxAngleDeg, over, hits: 1 });
        else { p.hits += 1; if (over > p.over) { p.over = over; p.raw = rawDeg; } }
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
  // { bone, raw°, max°, over°, hits, frames, pct }. `pct` = clamped frames / total
  // → distinguishes a sustained reach (high pct) from a 1-frame inversion spike.
  clampReport() {
    const frames = this.applyCount || 0;
    const rows = [];
    for (const [bone, s] of this.clampPeak) {
      rows.push({ bone, raw: s.raw, max: s.max, over: s.over, hits: s.hits, frames, pct: frames ? s.hits / frames : 0 });
    }
    rows.sort((a, b) => b.over - a.over);
    return rows;
  }

  resetClampPeak() { this.clampPeak.clear(); this.applyCount = 0; }

  // Direction-only (swing) aim. from/to: {x,y,z,confidence}. Soft-gated: low joint
  // confidence eases the bone toward rest (gain→0), not freeze→pop.
  _aim(boneKey, from, to, opts) {
    const bind = this.bind.get(boneKey);
    if (!bind || !from || !to) return;
    const gate = confGate(Math.min(from.confidence, to.confidence), opts.kptThresh) * (opts.gain ?? 1);
    _target.set(to.x - from.x, to.y - from.y, (to.z - from.z) * opts.depthScale);
    if (_target.lengthSq() < 1e-8) return;
    _target.normalize();
    if (opts.mirrorX) _target.x = -_target.x;
    _q.setFromUnitVectors(bind.restDirWorld, _target).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(boneKey, bind, _q, opts.maxAngleDeg, opts.follow, gate);
  }

  // Upper-limb aim with swing-twist: orient by bone direction AND bend-plane.
  // Soft-gated on the root/mid joints (low conf → ease toward rest, no pop).
  _aimLimb(limb, canonical, opts) {
    const bind = this.bind.get(limb.upper);
    if (!bind) return;
    const r = canonical.joints[limb.root];
    const m = canonical.joints[limb.mid];
    const e = canonical.joints[limb.end];
    if (!r || !m) return;
    const gate = confGate(Math.min(r.confidence, m.confidence), opts.kptThresh);

    const sx = opts.mirrorX ? -1 : 1;
    _rootP.set(r.x * sx, r.y, r.z * opts.depthScale);
    _midP.set(m.x * sx, m.y, m.z * opts.depthScale);
    _dir.copy(_midP).sub(_rootP);
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    const canTwist = opts.swingTwist && bind.bindPlaneN && e && e.confidence >= opts.kptThresh;
    if (canTwist) {
      _endP.set(e.x * sx, e.y, e.z * opts.depthScale);
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
        this._setBoneFromWorld(limb.upper, bind, _q, opts.maxAngleDeg, opts.follow, gate);
        return;
      }
    }
    // fallback: swing only
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(limb.upper, bind, _q, opts.maxAngleDeg, opts.follow, gate);
  }

  // Forearm: aim elbow→wrist (swing) + optional axial twist from the hand knuckle
  // line (indexMCP→pinkyMCP), so pronation/supination rotates the forearm.
  _aimForeArm(fa, canonical, opts) {
    const bind = this.bind.get(fa.bone);
    if (!bind) return;
    const r = canonical.joints[fa.root];
    const m = canonical.joints[fa.mid];
    if (!r || !m) return;
    const gate = confGate(Math.min(r.confidence, m.confidence), opts.kptThresh);
    const sx = opts.mirrorX ? -1 : 1;
    _rootP.set(r.x * sx, r.y, r.z * opts.depthScale);
    _midP.set(m.x * sx, m.y, m.z * opts.depthScale);
    _dir.copy(_midP).sub(_rootP);
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    if (opts.wristTwist && bind.bindAcross) {
      const a = canonical.joints[fa.rollA];
      const b = canonical.joints[fa.rollB];
      if (a && b && a.confidence >= opts.kptThresh && b.confidence >= opts.kptThresh) {
        _planeN.set((b.x - a.x) * sx, b.y - a.y, (b.z - a.z) * opts.depthScale);
        _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // perpendicular to forearm
        const len = _planeN.length();
        if (len > 0.02) {
          _planeN.divideScalar(len);
          let last = this.lastPlaneN.get(fa.bone);
          if (!last) { last = _planeN.clone(); this.lastPlaneN.set(fa.bone, last); }
          else smoothAxis(last, _planeN, opts.planeFollow, 'align'); // forearm-roll plane sign carries no info
          basisQuat(bind.restDirWorld, bind.bindAcross, _dir, last, _q);
          _q.multiply(bind.bindWorldQuat);
          this._setBoneFromWorld(fa.bone, bind, _q, opts.maxAngleDeg, opts.follow, gate);
          return;
        }
      }
    }
    _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat);
    this._setBoneFromWorld(fa.bone, bind, _q, opts.maxAngleDeg, opts.follow, gate);
  }

  // Torso: build ONE world rotation Q from the pelvis-aligned 3D body frame, then
  // distribute it across the spine chain (hips→spine→spine01→spine02) by cumulative
  // weight so the bend spreads instead of snapping one rigid bone. Q is a world delta
  // (Q^f ∘ bindWorld) so each bone's rest dir is irrelevant (dodges the Spine -y flip,
  // §B3). Lean + tilt + TURN all fall out of the raw 3D positions "for free":
  //   up    = hip→shoulder    (spine axis)
  //   right = left-hip→right-hip (hip axis), orthonormalized ⊥ up
  // NO depth amplification / deadzone / sign-hold — those hacks turned monocular
  // z-noise into whole-body spin (§B8). The horizontal hip separation dominates the
  // hip-axis sign (stable near frontal); a real turn rotates the axis into z on its
  // own. Noise is handled by the input OneEuro + the per-bone slerp (smoothed map) —
  // i.e. we smooth the ROTATION, not a fragile extracted scalar.
  _aimTorso(canonical, opts) {
    const hipsBind = this.bind.get('hips');
    if (!hipsBind) return;
    const hc = hipCenter(canonical);
    const sc = shoulderCenter(canonical);
    const lh = canonical.joints[KPT.leftHip];
    const rh = canonical.joints[KPT.rightHip];
    const sx = opts.mirrorX ? -1 : 1;
    _dir.set((sc.x - hc.x) * sx, sc.y - hc.y, (sc.z - hc.z) * opts.depthScale); // up (spine), 3D
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();
    // Soft gate: low torso confidence → drive the chain toward REST (gain→0), not
    // freeze — eases out/in without a pop.
    const gate = confGate(Math.min(hc.confidence, sc.confidence), opts.kptThresh);

    let haveYaw = false;
    if (opts.yaw && hipsBind.bindAcross && lh.confidence >= opts.kptThresh && rh.confidence >= opts.kptThresh) {
      _planeN.set((rh.x - lh.x) * sx, rh.y - lh.y, (rh.z - lh.z) * opts.depthScale); // hip axis (right), 3D
      _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // orthonormalize ⊥ up
      if (_planeN.lengthSq() > 1e-6) {
        _planeN.normalize();
        basisQuat(hipsBind.restDirWorld, hipsBind.bindAcross, _dir, _planeN, _qTorso); // Q = world delta
        haveYaw = true;
      }
    }
    if (!haveYaw) _qTorso.setFromUnitVectors(hipsBind.restDirWorld, _dir); // lean only (no hip axis)

    // Distribute: cumulative weight f → Q^f at each bone (full Q at the top).
    let f = 0;
    for (const seg of SPINE_CHAIN) {
      f += seg.weight;
      const b = this.bind.get(seg.key);
      if (!b) continue;
      _qFrac.copy(_qId).slerp(_qTorso, Math.min(1, f)); // Q^f
      _qWorldT.copy(_qFrac).multiply(b.bindWorldQuat);
      this._setBoneFromWorld(seg.key, b, _qWorldT, opts.maxAngleDeg, opts.follow, gate);
    }
  }

  // Clavicles: aim each shoulder bone from its rest (shoulder→arm) direction toward
  // the observed shoulder joint relative to the shoulder-center → shrug + slight
  // protraction. Reduced gain (subtle); soft-gated on shoulder confidence.
  _aimClavicles(canonical, opts) {
    const sc = shoulderCenter(canonical);
    const sx = opts.mirrorX ? -1 : 1;
    for (const clav of CLAVICLES) {
      const bind = this.bind.get(clav.bone);
      const j = canonical.joints[clav.joint];
      if (!bind || !j) continue;
      const gate = confGate(Math.min(j.confidence, sc.confidence), opts.kptThresh);
      _dir.set((j.x - sc.x) * sx, j.y - sc.y, (j.z - sc.z) * opts.depthScale);
      if (_dir.lengthSq() < 1e-8) continue;
      _dir.normalize();
      _q.setFromUnitVectors(bind.restDirWorld, _dir).multiply(bind.bindWorldQuat);
      this._setBoneFromWorld(clav.bone, bind, _q, opts.maxAngleDeg, opts.follow, gate * (opts.gain ?? 1));
    }
  }

  // Head/neck: build the head frame from raw 3D (same principle as the torso) —
  //   up    = shoulderCenter→headCenter (pitch)
  //   right = left-ear→right-ear         (yaw + roll)
  // orthonormalized → basisQuat. Pitch, turn AND head-tilt all come from the 3D "for
  // free"; no nose-ear scalar, no yawGain, no z-damping. Ear baseline is short so it's
  // noisier than the hips, but there's no amplification → stable; noise handled by
  // input OneEuro + per-bone slerp.
  _aimHead(canonical, opts) {
    const bind = this.bind.get('neck');
    if (!bind) return;
    const sc = shoulderCenter(canonical);
    const hcen = headCenter(canonical);
    const le = canonical.joints[KPT.leftEar];
    const re = canonical.joints[KPT.rightEar];
    const gate = confGate(Math.min(sc.confidence, hcen.confidence), opts.kptThresh);
    const sx = opts.mirrorX ? -1 : 1;
    _dir.set((hcen.x - sc.x) * sx, hcen.y - sc.y, (hcen.z - sc.z) * opts.depthScale); // up
    if (_dir.lengthSq() < 1e-8) return;
    _dir.normalize();

    let framed = false;
    if (opts.yaw && bind.bindAcross && le && re && le.confidence >= opts.kptThresh && re.confidence >= opts.kptThresh) {
      _planeN.set((re.x - le.x) * sx, re.y - le.y, (re.z - le.z) * opts.depthScale); // ear axis (right)
      _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // orthonormalize ⊥ up
      if (_planeN.lengthSq() > 1e-6) {
        _planeN.normalize();
        basisQuat(bind.restDirWorld, bind.bindAcross, _dir, _planeN, _q); // Q = world delta
        framed = true;
      }
    }
    if (!framed) _q.setFromUnitVectors(bind.restDirWorld, _dir); // pitch only (no ears)
    _q.multiply(bind.bindWorldQuat);
    this._setBoneFromWorld('neck', bind, _q, opts.maxAngleDeg, opts.follow, gate * (opts.gain ?? 1));
  }

  apply(canonical, { kptThresh = 0.3, mirrorX = true, depthScale = 1, jointLimitDeg = 180, armLimit = 180, follow = 1, swingTwist = true, headGain = 1, planeFollow = 0.12, wristTwist = false, grounding = false, groundFollow = 0.3, bodyYaw = true, clavicle = false } = {}) {
    this.restPose();
    this.rig.hips.position.y = this.hipsRestPosY; // reset (restPose only touches rotations)
    this.clampStats.clear(); // V19: only THIS frame's aimed bones report clamps
    this.applyCount = (this.applyCount || 0) + 1; // V19: frames since last clamp reset
    // depthScale = the ONE monocular-z trust knob, applied to every aim's z (no
    // per-bone damping). 1 = raw 3D.
    const base = { kptThresh, mirrorX, follow, maxAngleDeg: jointLimitDeg, swingTwist, planeFollow, depthScale };

    // Torso (distributed across the spine chain); clavicles; upper limbs (children
    // compensate); forearms (with optional twist); then lower bones / feet / hands /
    // neck. Arms use a tighter limit (over-rotation guard). Torso first so the spine
    // is posed before its children (arms/neck) read updated parent world matrices.
    this._aimTorso(canonical, { ...base, depthScale: depthScale * 0.2, yaw: bodyYaw }); // hip/shoulder z noisy
    if (clavicle) this._aimClavicles(canonical, { ...base, depthScale: depthScale * 0.25, gain: 0.6 });
    for (const limb of LIMBS) {
      const isArm = limb.upper.includes('Arm');
      this._aimLimb(limb, canonical, { ...base, depthScale: depthScale * limb.zTrust, maxAngleDeg: isArm ? armLimit : jointLimitDeg });
    }
    for (const fa of FOREARMS) this._aimForeArm(fa, canonical, { ...base, depthScale: depthScale * fa.zTrust, wristTwist, maxAngleDeg: armLimit });
    this._aimHead(canonical, { ...base, depthScale: depthScale * 0.4, yaw: bodyYaw, gain: headGain });
    for (const seg of SEGMENTS) {
      this._aim(seg.bone, seg.from(canonical), seg.to(canonical), {
        ...base,
        depthScale: depthScale * (seg.zTrust ?? 1),
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
