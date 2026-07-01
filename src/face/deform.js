// Face deform (T28/T30, extracted from main.js in T31 / V31, I.faceModules). Crowd-facing
// GPU side: builds the per-model mask + GPU buffers, the config-driven TSL deform node,
// the per-frame expression write, and the debug vertex-cloud overlay. Pure of editor UI
// (that's editor.js, which imports this). App singletons come via initDeform(ctx) so the
// function bodies stay identical to the original inline code (V34 behaviour-preserving).
import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import { cos, float, mix, sign, sin, storage, uniform, uint, vec3 } from 'three/tsl';
import { FACE_REGIONS, generateFaceMask, encodeFaceMask, decodeFaceMask, assertMaskFits } from '../pose/face-mask.js';
import { assetUrl } from '../asset-url.js';
import { MESHY_BONES } from '../pose/rig-map.js';
import { runtime } from './face-runtime.js';

// App singletons, injected once by main.js (initDeform) after the scene exists.
let scene; let state; let crowdBatches; let statusEl;
export function initDeform(ctx) { ({ scene, state, crowdBatches, statusEl } = ctx); }

const _faceMaskOverlays = new Set(); // debug vertex clouds, recolored on region change
const _faceMaskV = new THREE.Vector3();
const _defAxis = new THREE.Vector3();
export const faceMaskOverlays = _faceMaskOverlays; // main.js clears these on rebuild

// Flatten a possibly-interleaved attribute into a plain typed array [i*n+c] — the
// pure generator (face-mask.js) assumes that contiguous layout.
export function flatAttr(attr, n, Ctor) {
  const out = new Ctor(attr.count * n);
  for (let i = 0; i < attr.count; i += 1) for (let c = 0; c < n; c += 1) out[i * n + c] = attr.getComponent(i, c);
  return out;
}

// Load-or-generate the region mask for a batch's head mesh, then stand up the GPU
// buffers the deform reads: a per-instance expression buffer + a per-vertex mask
// buffer + hinge/amount uniforms. Sidecar `.bin` (T29 painter output) wins when
// present + valid; else auto-gen from geometry (V27). No head bone / no head verts →
// face-drive off for this model (⊥ crash the crowd, V28).
export async function buildFaceMask(batch, mesh, skinnedMeshes) {
  batch.faceMask = null;
  batch.faceExpr = null;

  // Head mesh + Head bone index from the skinned mesh's own skeleton (available now,
  // before skeletonStates are built in the compute loop).
  let headMesh = null; let headBoneIndex = -1; let headVerts = -1;
  for (const sm of skinnedMeshes) {
    const hbi = sm.skeleton ? sm.skeleton.bones.findIndex((b) => b.name === MESHY_BONES.head) : -1;
    const si = sm.geometry.getAttribute('skinIndex');
    const sw = sm.geometry.getAttribute('skinWeight');
    if (hbi < 0 || !si || !sw) continue;
    let n = 0;
    for (let i = 0; i < si.count; i += 1) {
      for (let k = 0; k < 4; k += 1) { if (si.getComponent(i, k) === hbi && sw.getComponent(i, k) > 0.5) { n += 1; break; } }
    }
    if (n > headVerts) { headVerts = n; headMesh = sm; headBoneIndex = hbi; }
  }
  if (!headMesh || headVerts <= 0) { console.warn(`[face] ${mesh.name}: no head-weighted verts / '${MESHY_BONES.head}' bone — face-drive off`); return; }

  const geom = headMesh.geometry;
  const vertexCount = geom.getAttribute('position').count;

  let mask = null; let source = 'generated';
  try {
    const res = await fetch(assetUrl(`models/${mesh.id}.face-mask.bin`));
    if (res.ok) { mask = assertMaskFits(decodeFaceMask(await res.arrayBuffer()), vertexCount); source = 'loaded (.bin)'; }
  } catch (err) {
    console.warn(`[face] ${mesh.name}: sidecar invalid (${err.message}) — regenerating`);
  }
  if (!mask) {
    mask = generateFaceMask({
      positions: flatAttr(geom.getAttribute('position'), 3, Float32Array),
      skinIndices: flatAttr(geom.getAttribute('skinIndex'), 4, Uint16Array),
      skinWeights: flatAttr(geom.getAttribute('skinWeight'), 4, Float32Array),
      count: vertexCount,
      headBoneIndex,
      forwardSign: 1
    });
  }

  // Per-instance expression = FLAT floats (8/instance) so the deform can index the
  // driver scalar by a per-region uniform int (configurable regions, T30).
  const nWalkers = batch.walkers.length;
  batch.faceExpr = new StorageBufferAttribute(nWalkers * 8, 1);
  batch.faceExprNode = storage(batch.faceExpr, 'float', nWalkers * 8).toReadOnly();

  // Per-vertex mask buffer: 2 vec4/vertex = [jaw,lowerLip,mouthCorner,upperLidL |
  // upperLidR,browL,browR,_], normalized 0..1. Filled from mask.data; re-uploaded live
  // when the editor repaints/re-seeds.
  const maskAttr = new StorageBufferAttribute(vertexCount * 2, 4);
  uploadMaskBuffer(mask, maskAttr);
  const maskNode = storage(maskAttr, 'vec4', vertexCount * 2).toReadOnly();

  // Per-region config uniforms (driver/type/amount/dir/mirror/hinge) — every field
  // live-tunable from the editor without a shader rebuild (T30). faceOn gates the whole
  // deform (0 when faceDrive off). Values synced from mask.config by syncConfigUniforms.
  batch.faceUniforms = {
    headHeight: uniform(mask.headHeight || 1),
    faceOn: uniform(0),
    regions: mask.config.map(() => ({
      driver: uniform(0), type: uniform(0), amount: uniform(0), mirrorX: uniform(0),
      dir: uniform(new THREE.Vector3()), hingeOrigin: uniform(new THREE.Vector3()), hingeAxis: uniform(new THREE.Vector3(1, 0, 0))
    }))
  };

  batch.faceMask = { mask, headMesh, headBoneIndex, vertexCount, source, maskNode, maskAttr, geom, forwardSign: 1 };
  syncConfigUniforms(batch);
  console.log(`[face] ${mesh.name}: mask ${source}, ${vertexCount} verts, head bone ${headBoneIndex}`);
  buildFaceMaskOverlay(batch);
}

// Fill the GPU mask buffer from mask.data (region-major uint8 → per-vertex 2×vec4).
export function uploadMaskBuffer(mask, maskAttr) {
  const arr = maskAttr.array;
  const D = mask.data; const N = mask.vertexCount;
  for (let i = 0; i < N; i += 1) {
    const o = i * 8;
    arr[o] = D[i] / 255; arr[o + 1] = D[N + i] / 255; arr[o + 2] = D[2 * N + i] / 255; arr[o + 3] = D[3 * N + i] / 255;
    arr[o + 4] = D[4 * N + i] / 255; arr[o + 5] = D[5 * N + i] / 255; arr[o + 6] = D[6 * N + i] / 255; arr[o + 7] = 0;
  }
  maskAttr.needsUpdate = true;
}

// Face deform (T28/T30 / V29): displace the BIND-space head vertex before skinning, so
// the head bone carries the motion. GENERAL config-driven form (T30 configurable
// regions): each of the 7 regions independently contributes s = driver·weight·amount,
// applied as either a TRANSLATE (dir·s·headHeight) or a HINGE-rotate (angle s about its
// axis), blended by the region's `type` uniform. Every field is a uniform → the editor
// retargets drivers / switches type / moves hinges with NO shader rebuild. Normals
// unchanged (V29 note).
export function faceDeformNode(pos, sourceVertex, meshInstance, fm, batch) {
  const u = batch.faceUniforms;
  const sv2 = sourceVertex.mul(uint(2));
  const maskA = fm.maskNode.element(sv2);            // regions 0..3
  const maskB = fm.maskNode.element(sv2.add(uint(1))); // regions 4..6, _
  const weight = [maskA.x, maskA.y, maskA.z, maskA.w, maskB.x, maskB.y, maskB.z];
  const instBase = meshInstance.mul(uint(8));
  const hh = u.headHeight;
  let delta = vec3(0, 0, 0);
  for (let r = 0; r < 7; r += 1) {
    const rc = u.regions[r];
    const driver = batch.faceExprNode.element(instBase.add(uint(rc.driver))); // dynamic driver select
    const s = driver.mul(weight[r]).mul(rc.amount).mul(u.faceOn);
    // translate: dir·s·headHeight, with mirrorX flipping dir.x by the vertex's side.
    const mx = mix(float(1), sign(pos.x.sub(rc.hingeOrigin.x)), rc.mirrorX);
    const tDelta = vec3(rc.dir.x.mul(mx), rc.dir.y, rc.dir.z).mul(s).mul(hh);
    // hinge: Rodrigues rotate (pos-origin) about axis by angle s.
    const v = pos.sub(rc.hingeOrigin);
    const k = rc.hingeAxis;
    const c = cos(s); const sn = sin(s);
    const hDelta = v.mul(c).add(k.cross(v).mul(sn)).add(k.mul(k.dot(v)).mul(c.oneMinus())).sub(v);
    delta = delta.add(mix(tDelta, hDelta, rc.type));
  }
  return pos.add(delta);
}

// Push mask.config → the per-region deform uniforms (live edits, no rebuild).
export function syncConfigUniforms(batch) {
  const u = batch.faceUniforms; if (!u) return;
  const cfg = batch.faceMask.mask.config;
  for (let r = 0; r < cfg.length; r += 1) {
    const c = cfg[r]; const ru = u.regions[r];
    ru.driver.value = c.driver; ru.type.value = c.type; ru.amount.value = c.amount; ru.mirrorX.value = c.mirrorX ? 1 : 0;
    ru.dir.value.set(c.dir[0], c.dir[1], c.dir[2]);
    ru.hingeOrigin.value.set(c.hingeOrigin[0], c.hingeOrigin[1], c.hingeOrigin[2]);
    _defAxis.set(c.hingeAxis[0], c.hingeAxis[1], c.hingeAxis[2]);
    if (_defAxis.lengthSq() < 1e-8) _defAxis.set(1, 0, 0); else _defAxis.normalize();
    ru.hingeAxis.value.copy(_defAxis);
  }
  u.headHeight.value = batch.faceMask.mask.headHeight || 1;
}

// Debug cloud: head verts at bind-pose world position, colored by the selected
// region's weight (dark → hot). Standalone Points → cannot disturb the render path.
export function buildFaceMaskOverlay(batch) {
  if (!batch.faceMask) return;
  const { headMesh, vertexCount } = batch.faceMask;
  headMesh.updateWorldMatrix(true, false);
  const src = headMesh.geometry.getAttribute('position');
  const positions = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i += 1) {
    _faceMaskV.fromBufferAttribute(src, i).applyMatrix4(headMesh.matrixWorld);
    positions[i * 3] = _faceMaskV.x; positions[i * 3 + 1] = _faceMaskV.y; positions[i * 3 + 2] = _faceMaskV.z;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3));
  // Additive + no depth-test: zero-weight verts are black → invisible; region verts
  // glow THROUGH the crowd (the overlay sits at the hidden template at the origin, so
  // depth-testing buried it behind the real instances).
  const pts = new THREE.Points(g, new THREE.PointsMaterial({
    size: 0.02, sizeAttenuation: true, vertexColors: true, depthTest: false, transparent: true, blending: THREE.AdditiveBlending
  }));
  pts.name = `face-mask-overlay-${batch.mesh.id}`;
  pts.frustumCulled = false;
  pts.renderOrder = 1002;
  pts.visible = state.faceMaskDebug;
  pts.userData.batch = batch;
  scene.add(pts);
  batch.faceMaskOverlay = pts;
  _faceMaskOverlays.add(pts);
  recolorFaceMaskOverlay(pts);
}

export function recolorFaceMaskOverlay(pts) {
  const { mask, vertexCount } = pts.userData.batch.faceMask;
  const region = Math.max(0, FACE_REGIONS.indexOf(state.faceMaskRegion));
  const col = pts.geometry.getAttribute('color');
  const off = region * vertexCount;
  for (let i = 0; i < vertexCount; i += 1) {
    const w = mask.data[off + i] / 255;
    col.setXYZ(i, w * 1.4, w * w * 0.9, w * 0.12); // black(0) → red → yellow (additive)
  }
  col.needsUpdate = true;
}

export function refreshFaceMaskOverlays() {
  for (const pts of _faceMaskOverlays) { pts.visible = state.faceMaskDebug; if (state.faceMaskDebug) recolorFaceMaskOverlay(pts); }
}

// Persist the current masks so painter edits / a chosen auto-gen survive. Browser
// can't write public/ → download; drop the files into public/models/ and commit.
export function downloadFaceMasks() {
  let n = 0;
  for (const batch of crowdBatches) {
    if (!batch.faceMask) continue;
    const blob = new Blob([encodeFaceMask(batch.faceMask.mask)], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${batch.mesh.id}.face-mask.bin`;
    a.click();
    URL.revokeObjectURL(a.href);
    n += 1;
  }
  statusEl.textContent = n ? `Downloaded ${n} face mask(s) → commit into public/models/` : 'No face masks to download.';
}

// T28: push the tracked face expression into a walker's per-instance slot. Pose
// performers get the live scalars (when faceDrive is on); everyone else = neutral.
export function writeFaceExpr(walker, batch) {
  if (!batch.faceExpr) return;
  const arr = batch.faceExpr.array;
  const o = walker.batchIndex * 8;
  // debugFaceOverride (dev/CDP): force an expression on EVERY instance, independent of
  // webcam/pose — lets the test harness drive the deform without a live face.
  const e = runtime.debugFaceOverride
    || ((walker.boneSource === 'pose' && state.faceDrive && runtime.latestFaceExpr) ? runtime.latestFaceExpr : null);
  if (e) {
    arr[o] = e.jawOpen; arr[o + 1] = e.smile; arr[o + 2] = e.pucker; arr[o + 3] = e.blinkL;
    arr[o + 4] = e.blinkR; arr[o + 5] = e.browL; arr[o + 6] = e.browR; arr[o + 7] = 0;
  } else {
    for (let k = 0; k < 8; k += 1) arr[o + k] = 0;
  }
}

// Gate the whole deform each frame (per-region amounts live in the config uniforms,
// synced by syncConfigUniforms on edit).
export function updateFaceUniforms(batch) {
  const u = batch.faceUniforms;
  if (!u) return;
  u.faceOn.value = state.faceDrive ? 1 : 0;
}
