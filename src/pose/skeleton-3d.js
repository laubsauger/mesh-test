// 3D pose-skeleton debug helper — the DETECTED canonical pose drawn as industry-standard
// octahedron bones + joint-marker spheres, always-on-top (depthTest off) so it's readable
// over the mesh instead of a single-pixel line. Colour: face markers magenta, body cyan.
import * as THREE from 'three';
import { BODY_BONES } from './topology.js';

const FACE = new Set([0, 1, 2, 3, 4]); // nose, eyes, ears
const jointColor = (i) => (FACE.has(i) ? 0xff5cf0 : 0x00e5ff);

const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);

export class PoseSkeleton3D extends THREE.Group {
  constructor() {
    super();
    this.frustumCulled = false;
    this.joints = [];
    this.bones = [];
    const sphere = new THREE.SphereGeometry(1, 10, 8);
    for (let i = 0; i < 23; i += 1) {
      const m = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({
        color: jointColor(i), depthTest: false, depthWrite: false, transparent: true, opacity: 0.95
      }));
      m.renderOrder = 1001;
      m.frustumCulled = false;
      m.scale.setScalar(0.024);
      this.add(m);
      this.joints.push(m);
    }
    const oct = new THREE.OctahedronGeometry(1, 0); // Blender-style bone octahedron
    for (const [a, b] of BODY_BONES) {
      const m = new THREE.Mesh(oct, new THREE.MeshBasicMaterial({
        color: 0x7fd4ff, depthTest: false, depthWrite: false, transparent: true, opacity: 0.8
      }));
      m.renderOrder = 1000;
      m.frustumCulled = false;
      this.add(m);
      this.bones.push({ m, a, b });
    }
  }

  // canonical = CanonicalPoseObservation (pelvis-centered, +y up). The raw canonical is
  // NOT scale-stable (bbox jitter pulses the keypoint magnitudes), so we normalize per
  // frame by the TORSO length (pelvis→shoulder-center) to a fixed world size, smoothed —
  // the bones stay a constant length instead of scaling up/down. `targetTorso` = world
  // torso length. Octahedron stretched along its length → tapered bone.
  update(canonical, kptThresh, targetTorso = 0.5, width = 0.02) {
    const j = canonical?.joints;
    if (!j) { this.visible = false; return; }
    this.visible = true;
    const ls = j[5]; const rs = j[6]; // shoulders (pelvis = origin)
    // Scale from the 2D torso (x/y ONLY): x/y are crop-normalized, but z is a DIFFERENT
    // normalization (Z_RANGE, ~4× bigger) that jitters separately — mixing them made the
    // skeleton scale/distort on all axes. Damp z (ZD) so depth matches the x/y scale.
    const torso2d = ls && rs ? Math.hypot((ls.x + rs.x) / 2, (ls.y + rs.y) / 2) : 0;
    const raw = torso2d > 1e-3 ? targetTorso / torso2d : (this._scale || 1);
    this._scale = this._scale ? this._scale + (raw - this._scale) * 0.12 : raw; // smooth
    const scale = this._scale;
    const ZD = 0.35; // depth damp → z in the same visual scale as x/y
    const put = (v, p) => v.set(p.x * scale, p.y * scale, p.z * scale * ZD);
    for (let i = 0; i < 23; i += 1) {
      const p = j[i];
      const vis = !!(p && p.confidence >= kptThresh);
      this.joints[i].visible = vis;
      if (vis) put(this.joints[i].position, p);
    }
    for (const b of this.bones) {
      const pa = j[b.a];
      const pb = j[b.b];
      const vis = !!(pa && pb && pa.confidence >= kptThresh && pb.confidence >= kptThresh);
      b.m.visible = vis;
      if (!vis) continue;
      put(_va, pa);
      put(_vb, pb);
      b.m.position.copy(_va).add(_vb).multiplyScalar(0.5);
      _dir.copy(_vb).sub(_va);
      const len = _dir.length();
      if (len > 1e-6) { _dir.divideScalar(len); b.m.quaternion.setFromUnitVectors(_up, _dir); }
      b.m.scale.set(width, Math.max(0.01, len / 2), width); // long axis = bone length
    }
  }
}
