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
const _twist = new THREE.Quaternion(); // swing/twist limit temps
const _swing = new THREE.Quaternion();
const _qInv = new THREE.Quaternion();
const _qClampTmp = new THREE.Quaternion();
const _mBind = new THREE.Matrix4();
const _mTarget = new THREE.Matrix4();
const DEG2RAD = Math.PI / 180;
const TWIST_LIMIT_DEG = 90; // axial roll cap — separate from swing so roll can't
// inflate the reach clamp (that inflation is what yanked bones "weird", user §arch).

// Scale a quaternion's rotation angle down to maxDeg (slerp from identity). Mutates q.
function clampQuatAngle(q, maxDeg) {
  const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));
  const maxRad = maxDeg * DEG2RAD;
  if (angle > maxRad && angle > 1e-4) {
    _qClampTmp.identity().slerp(q, maxRad / angle);
    q.copy(_qClampTmp);
  }
  return angle / DEG2RAD; // pre-clamp angle (deg), for diagnostics
}

// Swing/twist limit: split `delta` (local rotation from rest) into TWIST (axial roll
// about the bone's own length axis) + SWING (the bend/reach), clamp each separately,
// recompose into `out`. Anatomical: roll gets a tight cap, reach a loose one — vs a
// single total-angle clamp where a bit of arbitrary roll ate the reach budget and
// yanked the bone toward rest. `axis` = bone length dir in its LOCAL frame.
function limitSwingTwist(delta, axis, swingMaxDeg, out) {
  const d = delta.x * axis.x + delta.y * axis.y + delta.z * axis.z;
  _twist.set(axis.x * d, axis.y * d, axis.z * d, delta.w);
  if (_twist.lengthSq() < 1e-8) _twist.set(0, 0, 0, 1); else _twist.normalize();
  _swing.copy(delta).multiply(_qInv.copy(_twist).invert()); // delta = swing·twist → swing = delta·twist⁻¹
  const swingRaw = clampQuatAngle(_swing, swingMaxDeg);
  clampQuatAngle(_twist, TWIST_LIMIT_DEG);
  out.copy(_swing).multiply(_twist);
  return swingRaw;
}

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

// FacingEstimator — stable body/head YAW without trusting the flip-prone monocular
// z-sign (measured: shoulder depth axis is near-zero 70% of the time, its sign pure
// noise → the 180° spin). Instead:
//   magnitude ← 2D L↔R width FORESHORTENING (image plane, stable): turn shrinks the
//               projected shoulder/ear width. Scale-normalized by body/head height so
//               distance doesn't fake a turn. Running-max = the frontal reference.
//   sign      ← z-diff, but ONLY consulted once the magnitude proves a REAL turn
//               (past a gate) AND the z-diff is decisive — else the sign is HELD.
//   deadzone  ← near-frontal magnitude → 0 (no yaw at all), so frontal z-noise can't
//               rotate anything. This is the whole fix: sign of ~0 is never applied.
export class FacingEstimator {
  constructor() { this.r0 = 0; this.sign = 1; this.theta = 0; this.pitch = 0; this.pitch0 = undefined; }

  update(l, r, height2D, { deadzone = 0.17, signGate = 0.32, zMin = 0.05, follow = 0.15, gain = 1 } = {}) {
    const width2D = Math.hypot(r.x - l.x, r.y - l.y);
    const ratio = height2D > 1e-4 ? width2D / height2D : 0;
    this.r0 = Math.max(this.r0 * 0.9995, ratio); // running-max frontal ratio (scale-invariant)
    const cos = this.r0 > 1e-4 ? Math.min(1, ratio / this.r0) : 1;
    // SOFT deadzone: subtract it (continuous ramp from 0) instead of a hard cutoff that
    // snapped 0→deadzone at the edge — that snap was the "sudden jump at a threshold".
    const mag = Math.max(0, Math.min(Math.PI / 2, Math.acos(cos) * gain) - deadzone);
    const zdiff = r.z - l.z;
    if (mag > signGate && Math.abs(zdiff) > zMin) this.sign = Math.sign(zdiff) || this.sign;
    this.theta += (this.sign * mag - this.theta) * follow; // temporal smooth
    return this.theta;
  }

  // Smooth a DIRECTLY-provided signed yaw (head nose-offset — already sign-correct 2D).
  // Soft deadzone (subtract, not cutoff) so it eases in without a jump at the edge.
  smoothAngle(raw, { deadzone = 0.17, follow = 0.15 } = {}) {
    const mag = Math.max(0, Math.abs(raw) - deadzone);
    this.theta += (Math.sign(raw) * mag - this.theta) * follow;
    return this.theta;
  }

  // Head PITCH (forward/back nod) from a raw signed value (nose vertical offset from the
  // ear line). Auto-baselines the frontal offset (slow running mean = pitch0) so rest =
  // no nod, then smooths the deviation. No calibration needed.
  smoothPitch(raw, { follow = 0.15 } = {}) {
    if (this.pitch0 === undefined) this.pitch0 = raw;
    this.pitch0 += (raw - this.pitch0) * 0.002; // slow frontal-baseline learn
    this.pitch += ((raw - this.pitch0) - this.pitch) * follow;
    return this.pitch;
  }
}

const _right = new THREE.Vector3();
const _qYaw = new THREE.Quaternion();

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
    this.torsoFacing = new FacingEstimator(); // 2D-foreshortening yaw (flip-safe, no z-sign spin)
    this.headFacing = new FacingEstimator();
    this._flipHold = new Map(); // boneKey → consecutive pose-frames a big jump has persisted
    this._lastCanon = null; // reference → detect a NEW pose frame (vs a repeated render frame)
    this._fresh = false;
    this.flipRejectDeg = 180; // per-bone jump gate (deg); 180 = off. Set from apply().
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
    // localAim = bone's length axis in its OWN local frame (twist axis for the
    // swing/twist limit): world child-dir mapped back through the bind orientation.
    const localAim = restDirWorld.clone().applyQuaternion(_qInv.copy(bindWorldQuat).invert());
    const entry = { restDirWorld, bindWorldQuat, restLocalQuat: bone.quaternion.clone(), localAim };
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
      const localAim = restDirWorld.clone().applyQuaternion(_qInv.copy(bindWorldQuat).invert());
      this.bind.set(handKey, { restDirWorld, bindWorldQuat, restLocalQuat: bone.quaternion.clone(), localAim });
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

    // gain<1 eases the target toward rest (soft confidence / headGain). Preserve the
    // target first: copy(rest) would clobber _qLocal → slerp rest onto itself → the bone
    // PINNED to rest (the head, always gain 0.7, never moved; any soft-gated bone too).
    if (gain < 1) {
      _qClampTmp.copy(_qLocal); // target
      _qLocal.copy(bind.restLocalQuat).slerp(_qClampTmp, gain);
    }

    if (maxAngleDeg < 180) {
      // Plain total-angle clamp on the LOCAL articulation from rest. (Dropped the
      // swing/twist split: it capped roll separately from reach, so a stacked twist
      // pushed the TOTAL past the cap and PINNED bones — the "piece resisting rotation".
      // The real flip fixes are upstream — straightness gates + 2D facing — so a bone
      // never gets a garbage twist target that needs anatomical un-splitting here.)
      _qDelta.copy(bind.restLocalQuat).invert().multiply(_qLocal); // articulation from rest
      const rawDeg = 2 * Math.acos(Math.min(1, Math.abs(_qDelta.w))) / DEG2RAD;
      const maxRad = maxAngleDeg * DEG2RAD;
      if (rawDeg * DEG2RAD > maxRad && rawDeg > 1e-4 / DEG2RAD) {
        // Clamp toward rest by (max/raw): rest → TARGET, not rest → rest. Must preserve
        // the target first — copy(rest) would clobber _qLocal, slerping rest onto itself
        // → every clamped bone snapped to REST (the arm stuck at A-pose). This was latent
        // (only spine hit the else-branch) until the swing/twist path was dropped.
        _qClampTmp.copy(_qLocal); // the aimed target
        _qLocal.copy(bind.restLocalQuat).slerp(_qClampTmp, maxAngleDeg / rawDeg);
      }
      // V19: how far this bone WANTED to swing vs the cap it hit (deg).
      const clamped = rawDeg > maxAngleDeg + 0.01;
      this.clampStats.set(boneKey, { raw: rawDeg, max: maxAngleDeg, clamped });
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
    // PER-BONE JUMP REJECT: a single bone flipping huge in one pose-frame (a leg twisting
    // 180° while you stand still, a hand spinning) is a monocular glitch the whole-pose
    // gate can't see. If the target is a big jump from the held value, DON'T chase it —
    // only accept once it has PERSISTED a couple pose-frames (= real fast motion). Held
    // across render frames (cur doesn't move → the jump stays big); the counter advances
    // only on a fresh pose-frame.
    if (this.flipRejectDeg < 180) {
      const stepDeg = 2 * Math.acos(Math.min(1, Math.abs(cur.dot(_qLocal)))) / DEG2RAD;
      if (stepDeg > this.flipRejectDeg) {
        if (this._fresh) this._flipHold.set(boneKey, (this._flipHold.get(boneKey) || 0) + 1);
        if ((this._flipHold.get(boneKey) || 0) <= 2) { // hold up to 2 pose-frames, then trust it
          bone.quaternion.copy(cur); // suppress: hold last good
          bone.updateWorldMatrix(false, false);
          return;
        }
      } else if (this._fresh) {
        this._flipHold.set(boneKey, 0); // target settled near cur → reset
      }
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
      _target.copy(_endP).sub(_midP); // forearm vector (mid→end)
      const foreLen = _target.length();
      _planeN.copy(_target).cross(_dir); // bend-plane normal = forearm × upperDir
      const planeLen = _planeN.length();
      // STRAIGHTNESS GATE (the arm-roll fix): sin(bend) = |forearm×dir|/|forearm|. A
      // near-straight arm has NO defined bend plane — its normal is noise that flips
      // ~180°/frame (measured up to 172° → the hard axial twist). Only twist a
      // MEANINGFULLY BENT arm (>~25°); straighter → swing-only (roll is invisible on a
      // straight arm anyway). The old |planeLen|>0.04 gate engaged twist at ~8° — deep
      // in the noise. Bent enough → smooth the plane (align: bend-plane sign is free).
      const sinBend = foreLen > 1e-6 ? planeLen / foreLen : 0;
      if (sinBend > 0.42) { // ≈25° elbow bend
        _planeN.divideScalar(planeLen);
        let last = this.lastPlaneN.get(limb.upper);
        if (!last) {
          last = _planeN.clone();
          this.lastPlaneN.set(limb.upper, last);
        } else {
          smoothAxis(last, _planeN, opts.planeFollow, 'align');
        }
        basisQuat(bind.restDirWorld, bind.bindPlaneN, _dir, last, _q);
        _q.multiply(bind.bindWorldQuat);
        this._setBoneFromWorld(limb.upper, bind, _q, opts.maxAngleDeg, opts.follow, gate);
        return;
      }
      // near-straight → fall through to swing-only (no roll)
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
        // Knuckle line (index→pinky) with z DAMPED hard — hand z is the noisiest signal
        // (measured Δz ~0.2), and a noisy z flips the roll axis (same as the upper arm).
        // The palm orientation is stable in 2D; z only refines it.
        _planeN.set((b.x - a.x) * sx, b.y - a.y, (b.z - a.z) * opts.depthScale * 0.3);
        const knuckLen = _planeN.length();
        _planeN.addScaledVector(_dir, -_dir.dot(_planeN)); // perpendicular to forearm
        const perpLen = _planeN.length();
        // GATE: only roll when the knuckle line is meaningfully PERPENDICULAR to the
        // forearm (a defined palm normal). Near-parallel (edge-on palm / degenerate
        // keypoints) → the roll axis is noise that flips → swing-only instead.
        if (knuckLen > 1e-6 && perpLen / knuckLen > 0.5) { // >~30° off the forearm axis
          _planeN.divideScalar(perpLen);
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

    // θ from 2D shoulder-width FORESHORTENING (stable) + hysteresis z-sign + frontal
    // deadzone — NOT the flip-prone 3D hip axis. Updated EVERY frame (even when yaw
    // isn't applied) so the facing-debug overlay reflects live detection; holds last
    // when shoulders drop.
    const ls = canonical.joints[KPT.leftShoulder];
    const rs = canonical.joints[KPT.rightShoulder];
    let theta = this.torsoFacing.theta;
    if (ls && rs && ls.confidence >= opts.kptThresh && rs.confidence >= opts.kptThresh) {
      const height2D = Math.hypot(sc.x - hc.x, sc.y - hc.y); // torso height (scale ref)
      theta = this.torsoFacing.update(
        { x: ls.x * sx, y: ls.y, z: ls.z }, { x: rs.x * sx, y: rs.y, z: rs.z }, height2D, opts.facing);
    }
    let haveYaw = false;
    if (opts.yaw && hipsBind.bindAcross) {
      // Base the target right-axis on the RIG's own right (bindAcross), re-orthogonalized
      // to the current up, then rotate by θ. (Using image-right `sx,0,0` was ±x-flipped
      // vs bindAcross → a 180° offset that made basisQuat map to ~rest = no turn.)
      _right.copy(hipsBind.bindAcross);
      _right.addScaledVector(_dir, -_dir.dot(_right)); // ⊥ current up
      if (_right.lengthSq() > 1e-6) {
        _right.normalize();
        _qYaw.setFromAxisAngle(_dir, theta);
        _right.applyQuaternion(_qYaw); // rotate right into the turn
        basisQuat(hipsBind.restDirWorld, hipsBind.bindAcross, _dir, _right, _qTorso);
        haveYaw = true;
      }
    }
    if (!haveYaw) _qTorso.setFromUnitVectors(hipsBind.restDirWorld, _dir); // lean only (no yaw)

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

    // Head yaw from FACE GEOMETRY: nose horizontal offset from the ear midline. SIGNED
    // + 2D → no z, no flip. Nose is the cleanest marker (measured conf 0.92). Normalized
    // by half the ear span (foreshortens on turn → amplifies toward profile). Updated
    // every frame (debug overlay); applied only when yaw is on.
    const nose = canonical.joints[KPT.nose];
    let theta = this.headFacing.theta;
    let pitch = this.headFacing.pitch;
    if (le && re && le.confidence >= opts.kptThresh && re.confidence >= opts.kptThresh) {
      const leX = le.x * sx; const reX = re.x * sx;
      const earCx = (leX + reX) / 2;
      const halfSpan = Math.abs(reX - leX) / 2;
      // Require the ears meaningfully separated: near profile the span → 0 and the
      // normalization explodes (measured pitch spiking to 17). Only read yaw+pitch when
      // the face is frontal enough to trust; else hold the last smoothed value.
      if (nose && nose.confidence >= opts.kptThresh && halfSpan > 0.03) {
        const rawYaw = Math.asin(Math.max(-1, Math.min(1, (nose.x * sx - earCx) / halfSpan))); // nose X → yaw
        const rawPitch = Math.max(-1, Math.min(1, ((le.y + re.y) / 2 - nose.y) / (2 * halfSpan))); // nose vs ear line → nod
        theta = this.headFacing.smoothAngle(rawYaw, opts.facing);
        pitch = this.headFacing.smoothPitch(rawPitch, opts.facing);
      }
    }
    let framed = false;
    if (opts.yaw && bind.bindAcross && le && re) {
      _right.copy(bind.bindAcross); // rig's own right axis (NOT image-right, which was ±x-flipped)
      _right.addScaledVector(_dir, -_dir.dot(_right)); // ⊥ current up
      if (_right.lengthSq() > 1e-6) {
        _right.normalize();
        _qYaw.setFromAxisAngle(_dir, theta);
        _right.applyQuaternion(_qYaw); // yaw: rotate right about up
        _qYaw.setFromAxisAngle(_right, pitch * (opts.pitchGain ?? 0)); // pitch: tilt up about right (nod)
        _dir.applyQuaternion(_qYaw);
        basisQuat(bind.restDirWorld, bind.bindAcross, _dir, _right, _q);
        framed = true;
      }
    }
    if (!framed) _q.setFromUnitVectors(bind.restDirWorld, _dir); // pitch/roll only (no yaw)
    _q.multiply(bind.bindWorldQuat);
    this._setBoneFromWorld('neck', bind, _q, opts.maxAngleDeg, opts.follow, gate * (opts.gain ?? 1));
  }

  // Reset the frontal-width reference (call on a "stand frontal" calibration frame so
  // the yaw estimator learns THIS person/distance's frontal shoulder/ear width).
  recalibrateFacing() { this.torsoFacing.r0 = 0; this.headFacing.r0 = 0; }

  apply(canonical, { kptThresh = 0.3, mirrorX = true, depthScale = 1, jointLimitDeg = 180, armLimit = 180, follow = 1, swingTwist = true, headGain = 1, planeFollow = 0.12, wristTwist = false, grounding = false, groundFollow = 0.3, bodyYaw = true, facingDeadzone = 0.17, facingSmooth = 0.15, facingBodyGain = 1, headPitchGain = 2, flipReject = 70, clavicle = false } = {}) {
    this._fresh = canonical !== this._lastCanon; // new pose-frame vs a repeated render-frame
    this._lastCanon = canonical;
    this.flipRejectDeg = flipReject;
    this.restPose();
    this.rig.hips.position.y = this.hipsRestPosY; // reset (restPose only touches rotations)
    this.clampStats.clear(); // V19: only THIS frame's aimed bones report clamps
    this.applyCount = (this.applyCount || 0) + 1; // V19: frames since last clamp reset
    // depthScale = the ONE monocular-z trust knob, applied to every aim's z (no
    // per-bone damping). 1 = raw 3D.
    const base = { kptThresh, mirrorX, follow, maxAngleDeg: jointLimitDeg, swingTwist, planeFollow, depthScale };
    const facing = { deadzone: facingDeadzone, follow: facingSmooth, gain: facingBodyGain }; // 2D-yaw estimator tuning

    // Torso (distributed across the spine chain); clavicles; upper limbs (children
    // compensate); forearms (with optional twist); then lower bones / feet / hands /
    // neck. Arms use a tighter limit (over-rotation guard). Torso first so the spine
    // is posed before its children (arms/neck) read updated parent world matrices.
    this._aimTorso(canonical, { ...base, depthScale: depthScale * 0.2, yaw: bodyYaw, facing }); // hip/shoulder z noisy
    if (clavicle) this._aimClavicles(canonical, { ...base, depthScale: depthScale * 0.25, gain: 0.6 });
    for (const limb of LIMBS) {
      const isArm = limb.upper.includes('Arm');
      this._aimLimb(limb, canonical, { ...base, depthScale: depthScale * limb.zTrust, maxAngleDeg: isArm ? armLimit : jointLimitDeg });
    }
    for (const fa of FOREARMS) this._aimForeArm(fa, canonical, { ...base, depthScale: depthScale * fa.zTrust, wristTwist, maxAngleDeg: armLimit });
    this._aimHead(canonical, { ...base, depthScale: depthScale * 0.4, yaw: bodyYaw, gain: headGain, facing, pitchGain: headPitchGain });
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

    // DEBUG: compare the estimator θ to the yaw ACTUALLY applied to the mesh bones, so
    // we can tell "estimator right, mesh wrong" from "not applied". Refresh the subtree
    // then read spine02 + neck WORLD yaw (twist about world-up) vs their rest.
    this.rig.hips.updateWorldMatrix(false, true);
    this.debugFacing = {
      bodyTheta: this.torsoFacing.theta * 180 / Math.PI,
      headTheta: this.headFacing.theta * 180 / Math.PI,
      spineYaw: this._yawAboutUp('spine02', this.bind.get('spine02')?.bindWorldQuat),
      neckYaw: this._yawAboutUp('neck', this.bind.get('neck')?.bindWorldQuat)
    };
  }

  // DEBUG: yaw (rotation about world-up) a bone's WORLD orientation carries vs its rest
  // bind — the yaw that actually reaches the mesh. Compare to the estimator θ.
  _yawAboutUp(boneKey, bindWorldQuat) {
    const bone = this.rig[boneKey];
    if (!bone || !bindWorldQuat) return 0;
    bone.getWorldQuaternion(_q); // current world orientation
    _qDelta.copy(_q).multiply(_qInv.copy(bindWorldQuat).invert()); // world delta from rest
    return 2 * Math.atan2(_qDelta.y, _qDelta.w) * 180 / Math.PI; // twist about world-up (deg, signed)
  }
}
