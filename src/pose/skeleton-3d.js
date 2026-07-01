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

  // canonical = CanonicalPoseObservation (pelvis-centered, +y up). scale maps its
  // normalized units to world (≈ character height). Bones span joint→joint; the octahedron
  // is stretched along its length so it reads as a tapered bone.
  update(canonical, kptThresh, scale, width = 0.02) {
    const j = canonical?.joints;
    if (!j) { this.visible = false; return; }
    this.visible = true;
    for (let i = 0; i < 23; i += 1) {
      const p = j[i];
      const vis = !!(p && p.confidence >= kptThresh);
      this.joints[i].visible = vis;
      if (vis) this.joints[i].position.set(p.x * scale, p.y * scale, p.z * scale);
    }
    for (const b of this.bones) {
      const pa = j[b.a];
      const pb = j[b.b];
      const vis = !!(pa && pb && pa.confidence >= kptThresh && pb.confidence >= kptThresh);
      b.m.visible = vis;
      if (!vis) continue;
      _va.set(pa.x * scale, pa.y * scale, pa.z * scale);
      _vb.set(pb.x * scale, pb.y * scale, pb.z * scale);
      b.m.position.copy(_va).add(_vb).multiplyScalar(0.5);
      _dir.copy(_vb).sub(_va);
      const len = _dir.length();
      if (len > 1e-6) { _dir.divideScalar(len); b.m.quaternion.setFromUnitVectors(_up, _dir); }
      b.m.scale.set(width, Math.max(0.01, len / 2), width); // long axis = bone length
    }
  }
}
