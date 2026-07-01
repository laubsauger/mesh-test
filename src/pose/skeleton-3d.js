// 3D pose-skeleton debug helper — the DETECTED canonical pose drawn as industry-standard
// octahedron bones + joint-marker spheres, always-on-top (depthTest off) so it's readable
// over the mesh instead of a single-pixel line. Colour: face markers magenta, body cyan.
import * as THREE from 'three';

const FACE = new Set([0, 1, 2, 3, 4]); // nose, eyes, ears
const jointColor = (i) => (FACE.has(i) ? 0xff5cf0 : i >= 91 ? 0xffb020 : 0x00e5ff); // face magenta, hands orange, body cyan

const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);

export class PoseSkeleton3D extends THREE.Group {
  // jointIndices = keypoint ids to show as spheres; edges = [a,b] pairs to draw as bones.
  constructor(jointIndices, edges) {
    super();
    this.frustumCulled = false;
    this.joints = new Map();
    this.bones = [];
    const sphere = new THREE.SphereGeometry(1, 10, 8);
    for (const i of jointIndices) {
      const isHand = i >= 91;
      const m = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({
        color: jointColor(i), depthTest: false, depthWrite: false, transparent: true, opacity: 0.95
      }));
      m.renderOrder = 1001;
      m.frustumCulled = false;
      m.scale.setScalar(isHand ? 0.012 : 0.024); // smaller for the dense hand joints
      this.add(m);
      this.joints.set(i, m);
    }
    const oct = new THREE.OctahedronGeometry(1, 0); // Blender-style bone octahedron
    for (const [a, b] of edges) {
      const m = new THREE.Mesh(oct, new THREE.MeshBasicMaterial({
        color: a >= 91 ? 0xffb020 : 0x7fd4ff, depthTest: false, depthWrite: false, transparent: true, opacity: 0.8
      }));
      m.renderOrder = 1000;
      m.frustumCulled = false;
      this.add(m);
      this.bones.push({ m, a, b });
    }
  }

  // worldJoints = array indexed by keypoint (0..22): {x,y,z} in WORLD coords (already
  // mesh-aligned + scaled by the caller), or null/undefined if not confident. The helper
  // just draws spheres at the joints + a stretched octahedron along each bone edge.
  update(worldJoints, width = 0.022) {
    for (const [i, m] of this.joints) {
      const p = worldJoints[i];
      m.visible = !!p;
      if (p) m.position.set(p.x, p.y, p.z);
    }
    for (const b of this.bones) {
      const pa = worldJoints[b.a];
      const pb = worldJoints[b.b];
      const vis = !!(pa && pb);
      b.m.visible = vis;
      if (!vis) continue;
      _va.set(pa.x, pa.y, pa.z);
      _vb.set(pb.x, pb.y, pb.z);
      b.m.position.copy(_va).add(_vb).multiplyScalar(0.5);
      _dir.copy(_vb).sub(_va);
      const len = _dir.length();
      if (len > 1e-6) { _dir.divideScalar(len); b.m.quaternion.setFromUnitVectors(_up, _dir); }
      b.m.scale.set(width, Math.max(0.01, len / 2), width); // long axis = bone length
    }
  }
}
