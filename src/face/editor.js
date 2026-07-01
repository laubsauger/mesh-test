// Face-mask editor (T29/T30, extracted from main.js in T31 / V31, I.faceModules). Owns
// the dedicated edit-head, brush painting, Move/Brush tools, two-click hinge, undo/redo,
// CPU preview deform, and the editor-store action registry (I.editorUI). Depends on
// deform.js (GPU side) + face-mask.js + editor-store.js — NOT on main.js. App singletons
// arrive via initEditor(ctx); function bodies are unchanged from the original inline code
// (V34 behaviour-preserving).
import * as THREE from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { attribute, mix, texture, uniform, vec3 } from 'three/tsl';
import { FACE_REGIONS, generateFaceMask, decodeFaceMask, assertMaskFits } from '../pose/face-mask.js';
import { registerActions, setEditorState, editorState, actions } from '../editor/editor-store.js';
import { HOTKEYS, REGION_KEYS, matchesHotkey } from '../editor/keymap.js';
import { uploadMaskBuffer, syncConfigUniforms, recolorFaceMaskOverlay, refreshFaceMaskOverlays, downloadFaceMasks, flatAttr } from './deform.js';
import { runtime } from './face-runtime.js';

// App singletons, injected once by main.js (initEditor) after the scene exists.
let scene; let camera; let controls; let renderer; let state; let crowdBatches; let statusEl;
export function initEditor(ctx) { ({ scene, camera, controls, renderer, state, crowdBatches, statusEl } = ctx); }

export const faceEditor = {
  active: false, batchIndex: 0, region: 'jaw', radius: 0.04, strength: 0.5,
  tool: 'brush', // 'camera' (orbit only) | 'brush' (paint; rotate/pan off, zoom on)
  erase: false, symmetric: true, hingeMode: false, editHead: null, batch: null, painting: false,
  hingeStage: 0, hingeGizmo: null, // two-click hinge (0=origin next, 1=axis next); axis line
  brushRing: null, prevDrift: false, // Photoshop-style cursor ring; saved drift state
  tempTool: null, // Ctrl-hold momentary tool (effective tool = tempTool || tool)
  history: [], histIndex: -1, lastSnapT: 0, lastSnapTag: '' // undo/redo ring
};
const REGION_HEX = { jaw: 0xff5c5c, lowerLip: 0xff9f43, mouthCorner: 0xfeca57, upperLidL: 0x54a0ff, upperLidR: 0x5f27cd, browL: 0x1dd1a1, browR: 0x00d2d3 };
const _brushN = new THREE.Vector3();
const _nMat = new THREE.Matrix3();
const _zAxis = new THREE.Vector3(0, 0, 1);
const REGION_MIRROR = { upperLidL: 'upperLidR', upperLidR: 'upperLidL', browL: 'browR', browR: 'browL' };
const _rayNDC = new THREE.Vector2();
const _raycaster = new THREE.Raycaster();
const _vLocal = new THREE.Vector3();

function editBatch() {
  const list = crowdBatches.filter((b) => b.faceMask);
  return list[Math.min(faceEditor.batchIndex, list.length - 1)] || null;
}

export function setFaceEditMode(on) {
  faceEditor.active = on;
  if (faceEditor.editHead) {
    scene.remove(faceEditor.editHead);
    faceEditor.editHead.geometry.dispose(); faceEditor.editHead.material.dispose();
    faceEditor.editHead = null;
  }
  faceEditor.batch = null;
  if (faceEditor.hingeGizmo) faceEditor.hingeGizmo.visible = false;
  if (faceEditor.brushRing) faceEditor.brushRing.visible = false;
  if (!on) {
    // Restore camera drift + full orbit on exit.
    faceEditor.tempTool = null;
    state.cameraDrift = faceEditor.prevDrift; controls.autoRotate = faceEditor.prevDrift;
    controls.enableRotate = true; controls.enablePan = true;
    return;
  }
  const batch = editBatch();
  if (!batch) { statusEl.textContent = 'Face editor: no model with a face mask.'; faceEditor.active = false; return; }
  faceEditor.batch = batch;
  // Auto-stop camera drift while editing (spinning the head fights painting).
  faceEditor.prevDrift = state.cameraDrift;
  state.cameraDrift = false; controls.autoRotate = false;
  buildEditHead(batch);
  frameEditHead();
  faceEditSetTool(faceEditor.tool);
  faceEditResetHistory();
  statusEl.textContent = `Face editor: ${batch.mesh.name} — Brush tool paints, Move tool orbits (B/V).`;
}

// Effective tool = the Ctrl-held momentary override, else the selected tool.
const effTool = () => faceEditor.tempTool || faceEditor.tool;

// Apply orbit/paint control mode for a tool: Camera = full orbit, no paint; Brush =
// paint (rotate/pan off so drags paint, zoom stays on).
function applyToolControls(t) {
  const brush = t === 'brush';
  controls.enableRotate = !brush; controls.enablePan = !brush; controls.enableZoom = true; controls.enabled = true;
  if (!brush && faceEditor.brushRing) faceEditor.brushRing.visible = false;
}

// Tool select (V/B or the panel). Clears any momentary override.
export function faceEditSetTool(t) {
  faceEditor.tool = t;
  faceEditor.tempTool = null;
  applyToolControls(t);
  setEditorState({ tool: t });
}

// Ctrl-hold: momentarily swap camera↔brush (orbit while painting) without changing the
// selected tool or the UI; reverts on release. (Mac: ⌘ handles undo/save, so Ctrl is
// free.) editorKeyDown/Up drive this.
function setTempTool(on) {
  if (on && !faceEditor.tempTool) { faceEditor.tempTool = faceEditor.tool === 'brush' ? 'camera' : 'brush'; applyToolControls(faceEditor.tempTool); }
  else if (!on && faceEditor.tempTool) { faceEditor.tempTool = null; applyToolControls(faceEditor.tool); }
}

// Static clone of the head geometry, placed to the left of the crowd, scaled to a
// comfortable size, vertex-colored by the current region weight.
function editHeadMap(headMesh) { // pull the base color map off the head material (may be array)
  const m = Array.isArray(headMesh.material) ? headMesh.material[0] : headMesh.material;
  return m && m.map ? m.map : null;
}

function buildEditHead(batch) {
  const src = batch.faceMask.geom;
  const N = batch.faceMask.vertexCount;
  const g = new THREE.BufferGeometry();
  const posAttr = src.getAttribute('position').clone();
  g.setAttribute('position', posAttr);
  faceEditor.bindPositions = Float32Array.from(posAttr.array); // bind copy for CPU preview deform
  if (src.index) g.setIndex(src.index.clone());
  if (src.getAttribute('uv')) g.setAttribute('uv', src.getAttribute('uv').clone()); // for texture blend
  g.setAttribute('aWeight', new THREE.BufferAttribute(new Float32Array(N), 1)); // selected-region weight
  g.computeBoundingBox();

  // Node material: base = mix(grey, model texture, texAlpha); the selected region's
  // weight highlights over it in hot orange. So you can paint against the real face
  // texture (alpha slider) or a flat grey, + wireframe.
  const mat = new MeshBasicNodeMaterial({ side: THREE.DoubleSide });
  faceEditor.texAlpha = uniform(0);
  const map = editHeadMap(batch.faceMask.headMesh);
  const texRGB = map ? texture(map).rgb : vec3(0.16, 0.16, 0.18);
  const base = mix(vec3(0.14, 0.14, 0.16), texRGB, faceEditor.texAlpha);
  const w = attribute('aWeight', 'float');
  mat.colorNode = mix(base, vec3(1.0, 0.36, 0.1), w.mul(0.9));
  const mesh = new THREE.Mesh(g, mat);
  mesh.name = 'face-edit-head';
  mesh.frustumCulled = false;
  mesh.renderOrder = 2000;
  const size = new THREE.Vector3(); g.boundingBox.getSize(size);
  const s = 1.2 / (size.y || 1); // fit ~1.2 world units tall
  mesh.scale.setScalar(s);
  const center = new THREE.Vector3(); g.boundingBox.getCenter(center);
  mesh.position.set(-1.8 - center.x * s, 1.2 - center.y * s, -center.z * s); // left of the crowd, eye level
  scene.add(mesh);
  faceEditor.editHead = mesh;
  recolorEditHead();
}

function frameEditHead() {
  const mesh = faceEditor.editHead; if (!mesh) return;
  const box = new THREE.Box3().setFromObject(mesh);
  const c = new THREE.Vector3(); box.getCenter(c);
  const sz = new THREE.Vector3(); box.getSize(sz);
  const r = Math.max(sz.x, sz.y, sz.z) || 1;
  controls.target.copy(c);
  camera.position.set(c.x + r * 0.2, c.y, c.z + r * 2.6);
  controls.update();
}

// Write the selected region's per-vertex weight into aWeight (the material highlights it).
function recolorEditHead() {
  const mesh = faceEditor.editHead; const batch = faceEditor.batch;
  if (!mesh || !batch) return;
  const N = batch.faceMask.vertexCount;
  const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const D = batch.faceMask.mask.data;
  const wa = mesh.geometry.getAttribute('aWeight');
  for (let i = 0; i < N; i += 1) wa.setX(i, D[r * N + i] / 255);
  wa.needsUpdate = true;
}

// Write one vertex's region weight → mask.data (uint8) + GPU maskAttr (normalized).
function setMaskWeight(batch, region, i, w01) {
  const N = batch.faceMask.vertexCount;
  const v = Math.max(0, Math.min(255, Math.round(w01 * 255)));
  batch.faceMask.mask.data[region * N + i] = v;
  batch.faceMask.maskAttr.array[i * 8 + region] = v / 255;
}

// Brush at a WORLD point: all edit-head verts within radius get add/erase × falloff.
// Symmetric mirrors across the head center-x (and swaps L/R regions for lids/brows).
export function faceEditPaint(worldPoint, erase) {
  const batch = faceEditor.batch; const mesh = faceEditor.editHead;
  if (!batch || !mesh) return;
  const region = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const pos = mesh.geometry.getAttribute('position');
  const N = batch.faceMask.vertexCount;
  const D = batch.faceMask.mask.data;
  mesh.worldToLocal(_vLocal.copy(worldPoint)); // → geometry-local coords
  const R = faceEditor.radius; const R2 = R * R;
  const str = faceEditor.strength * (erase ? -1 : 1);
  const cx = batch.faceMask.mask.hinge.origin[0];
  const paintAt = (px, py, pz, reg) => {
    for (let i = 0; i < N; i += 1) {
      const dx = pos.getX(i) - px; const dy = pos.getY(i) - py; const dz = pos.getZ(i) - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > R2) continue;
      const fall = 1 - Math.sqrt(d2) / R;
      setMaskWeight(batch, reg, i, D[reg * N + i] / 255 + str * fall);
    }
  };
  paintAt(_vLocal.x, _vLocal.y, _vLocal.z, region);
  if (faceEditor.symmetric) {
    const mreg = FACE_REGIONS.indexOf(REGION_MIRROR[faceEditor.region] || faceEditor.region);
    paintAt(2 * cx - _vLocal.x, _vLocal.y, _vLocal.z, mreg);
  }
  batch.faceMask.maskAttr.needsUpdate = true;
  recolorEditHead();
}

// Re-run auto-gen (optionally flipping the assumed +z forward), re-upload, recolor.
export function faceEditReseed(flipForward) {
  const batch = faceEditor.batch || editBatch(); if (!batch) return;
  const fm = batch.faceMask;
  if (flipForward) fm.forwardSign = -(fm.forwardSign || 1);
  const geom = fm.geom;
  fm.mask = generateFaceMask({
    positions: flatAttr(geom.getAttribute('position'), 3, Float32Array),
    skinIndices: flatAttr(geom.getAttribute('skinIndex'), 4, Uint16Array),
    skinWeights: flatAttr(geom.getAttribute('skinWeight'), 4, Float32Array),
    count: fm.vertexCount, headBoneIndex: fm.headBoneIndex, forwardSign: fm.forwardSign
  });
  uploadMaskBuffer(fm.mask, fm.maskAttr);
  syncConfigUniforms(batch);
  recolorEditHead();
  deformEditHead();
  pushRegionConfigToStore();
  if (batch.faceMaskOverlay) recolorFaceMaskOverlay(batch.faceMaskOverlay);
  faceEditCommit('reseed');
  statusEl.textContent = `Re-seeded ${batch.mesh.name} (forward ${fm.forwardSign > 0 ? '+z' : '-z'}).`;
}

export function faceEditClearRegion() {
  const batch = faceEditor.batch; if (!batch) return;
  const region = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const N = batch.faceMask.vertexCount;
  for (let i = 0; i < N; i += 1) setMaskWeight(batch, region, i, 0);
  batch.faceMask.maskAttr.needsUpdate = true;
  recolorEditHead();
  faceEditCommit('clear');
}

function faceEditRaycast(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  _rayNDC.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  _raycaster.setFromCamera(_rayNDC, camera);
  return _raycaster.intersectObject(faceEditor.editHead, false)[0] || null;
}

export function onFaceEditDown(e) {
  if (!faceEditor.active || !faceEditor.editHead || effTool() !== 'brush') return; // Camera tool → orbit only
  if (e.button === 3 || e.button === 4) return; // aux = undo/redo (window listener)
  const hit = faceEditRaycast(e);
  if (!hit) return;
  if (faceEditor.hingeMode) { faceEditSetHinge(hit.point); return; }
  faceEditor.painting = true;
  faceEditPaint(hit.point, e.button === 2 || faceEditor.erase);
}

// In Brush tool: follow the cursor with the ring (always) + paint while dragging.
export function onFaceEditMove(e) {
  if (!faceEditor.active || effTool() !== 'brush' || !faceEditor.editHead) {
    if (faceEditor.brushRing) faceEditor.brushRing.visible = false;
    return;
  }
  const hit = faceEditRaycast(e);
  updateBrushRing(hit);
  if (faceEditor.painting && hit) faceEditPaint(hit.point, faceEditor.erase || (e.buttons === 2));
}

export function onFaceEditUp() {
  if (faceEditor.painting) { faceEditor.painting = false; faceEditCommit('paint'); } // one undo/stroke
}

// Keyboard shortcuts (editor only). Everything routes through the store actions so the
// UI stays in sync; the labels shown on the buttons come from the same HOTKEYS map.
export function onEditorKeyDown(e) {
  if (!faceEditor.active) return;
  const tag = e.target?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return; // don't hijack typing
  if (e.key === 'Control') { setTempTool(true); return; } // Ctrl-hold: momentary tool swap
  const H = HOTKEYS;
  // Meta combos first (⌘/Ctrl).
  if (matchesHotkey(e, H.redo)) { actions.redo?.(); e.preventDefault(); return; }
  if (matchesHotkey(e, H.undo)) { actions.undo?.(); e.preventDefault(); return; }
  if (matchesHotkey(e, H.save)) { actions.save?.(); e.preventDefault(); return; }
  if (e.metaKey || e.ctrlKey) return; // leave other meta combos alone
  // Plain / shift keys.
  if (matchesHotkey(e, H.camera)) actions.setTool?.('camera');
  else if (matchesHotkey(e, H.brush)) actions.setTool?.('brush');
  else if (matchesHotkey(e, H.paint)) actions.setMode?.('paint');
  else if (matchesHotkey(e, H.erase)) actions.setMode?.('erase');
  else if (matchesHotkey(e, H.hinge)) actions.setMode?.('hinge');
  else if (matchesHotkey(e, H.symmetric)) actions.setSymmetric?.(!faceEditor.symmetric);
  else if (matchesHotkey(e, H.radiusDown)) actions.setRadius?.(Math.max(0.005, +(faceEditor.radius - 0.01).toFixed(3)));
  else if (matchesHotkey(e, H.radiusUp)) actions.setRadius?.(Math.min(0.3, +(faceEditor.radius + 0.01).toFixed(3)));
  else if (matchesHotkey(e, H.flip)) actions.reseed?.(true);   // ⇧R
  else if (matchesHotkey(e, H.reseed)) actions.reseed?.(false); // R
  else if (matchesHotkey(e, H.clear)) actions.clearRegion?.();
  else { const i = REGION_KEYS.indexOf(e.key); if (i >= 0 && i < FACE_REGIONS.length) actions.setRegion?.(FACE_REGIONS[i]); else return; }
  e.preventDefault();
}

export function onEditorKeyUp(e) {
  if (e.key === 'Control') setTempTool(false);
}

// Photoshop-style brush cursor: a ring on the surface at the paint radius, colored by
// the region (red when erasing), opacity ≈ strength, plus a faint inner falloff ring.
function buildBrushRing() {
  const grp = new THREE.Group();
  const mk = (inner, outer, opacity) => {
    const m = new THREE.Mesh(
      new THREE.RingGeometry(inner, outer, 48),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity, depthTest: false, side: THREE.DoubleSide })
    );
    grp.add(m); return m;
  };
  grp.userData.outer = mk(0.94, 1.0, 0.9); // radius edge
  grp.userData.inner = mk(0.46, 0.5, 0.4); // ~50% falloff hint
  grp.renderOrder = 2100; grp.frustumCulled = false; grp.visible = false;
  scene.add(grp);
  return grp;
}
function updateBrushRing(hit) {
  if (!faceEditor.brushRing) faceEditor.brushRing = buildBrushRing();
  const ring = faceEditor.brushRing;
  const show = faceEditor.active && effTool() === 'brush' && faceEditor.editHead && hit && !faceEditor.hingeMode;
  ring.visible = show;
  if (!show) return;
  ring.scale.setScalar(faceEditor.radius * faceEditor.editHead.scale.x); // local radius → world
  ring.position.copy(hit.point);
  const n = hit.face
    ? _brushN.copy(hit.face.normal).applyMatrix3(_nMat.getNormalMatrix(faceEditor.editHead.matrixWorld)).normalize()
    : _brushN.set(0, 0, 1);
  ring.quaternion.setFromUnitVectors(_zAxis, n);
  const hex = faceEditor.erase ? 0xff4444 : (REGION_HEX[faceEditor.region] || 0xffffff);
  ring.userData.outer.material.color.setHex(hex);
  ring.userData.inner.material.color.setHex(hex);
  ring.userData.outer.material.opacity = 0.35 + 0.6 * faceEditor.strength;
}

// Two-click hinge editing (T30): 1st click sets the current region's hinge ORIGIN (and
// switches it to hinge type); 2nd click sets the AXIS direction (origin→point). Updates
// the config + gizmo live.
function faceEditSetHinge(worldPoint) {
  const batch = faceEditor.batch; if (!batch) return;
  const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const c = batch.faceMask.mask.config[r];
  faceEditor.editHead.worldToLocal(_vLocal.copy(worldPoint));
  if (!faceEditor.hingeStage) {
    c.hingeOrigin = [_vLocal.x, _vLocal.y, _vLocal.z];
    c.type = 1; // hinging a region implies hinge type
    faceEditor.hingeStage = 1;
    statusEl.textContent = `${faceEditor.region}: hinge origin set — click a 2nd point for the axis.`;
  } else {
    const dx = _vLocal.x - c.hingeOrigin[0]; const dy = _vLocal.y - c.hingeOrigin[1]; const dz = _vLocal.z - c.hingeOrigin[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    c.hingeAxis = [dx / len, dy / len, dz / len];
    faceEditor.hingeStage = 0;
    statusEl.textContent = `${faceEditor.region}: hinge axis set.`;
  }
  syncConfigUniforms(batch); deformEditHead(); updateHingeGizmo(); pushRegionConfigToStore();
  faceEditCommit('hinge');
}

// Cyan line showing the current region's hinge (origin ± axis). Visible when the region
// is hinge-type or hinge mode is on.
function updateHingeGizmo() {
  const batch = faceEditor.batch; const eh = faceEditor.editHead;
  if (!batch || !eh) { if (faceEditor.hingeGizmo) faceEditor.hingeGizmo.visible = false; return; }
  const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const c = batch.faceMask.mask.config[r];
  const show = c.type === 1 || faceEditor.hingeMode;
  if (!faceEditor.hingeGizmo) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x00ffff, depthTest: false, transparent: true }));
    line.renderOrder = 2001; line.frustumCulled = false; scene.add(line);
    faceEditor.hingeGizmo = line;
  }
  const line = faceEditor.hingeGizmo; line.visible = show;
  if (!show) return;
  eh.updateWorldMatrix(true, false);
  const o = new THREE.Vector3(c.hingeOrigin[0], c.hingeOrigin[1], c.hingeOrigin[2]);
  const a = new THREE.Vector3(c.hingeAxis[0], c.hingeAxis[1], c.hingeAxis[2]);
  if (a.lengthSq() < 1e-8) a.set(1, 0, 0); else a.normalize();
  const L = (batch.faceMask.mask.headHeight || 1) * 0.6;
  const p1 = o.clone().addScaledVector(a, -L).applyMatrix4(eh.matrixWorld);
  const p2 = o.clone().addScaledVector(a, L).applyMatrix4(eh.matrixWorld);
  const arr = line.geometry.getAttribute('position').array;
  arr[0] = p1.x; arr[1] = p1.y; arr[2] = p1.z; arr[3] = p2.x; arr[4] = p2.y; arr[5] = p2.z;
  line.geometry.getAttribute('position').needsUpdate = true;
}

// Mirror the current region's config into the store so the React config panel reflects it.
function pushRegionConfigToStore() {
  const batch = faceEditor.batch; if (!batch) return;
  const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  setEditorState({ regionConfig: { ...batch.faceMask.mask.config[r] } });
}

// --- Undo/redo (T30 polish): commit the RESULTING mask (weights + config) AFTER each
// mutation → history[histIndex] is always the current state, so undo/redo just restore a
// neighbour. Continuous edits (amount-slider drag, tag 'config') coalesce into one step.
const cloneConfig = (cfg) => cfg.map((c) => ({ ...c, dir: c.dir.slice(), hingeOrigin: c.hingeOrigin.slice(), hingeAxis: c.hingeAxis.slice() }));
function faceEditCommit(tag) {
  const batch = faceEditor.batch; if (!batch) return;
  const now = performance.now();
  const coalesce = tag === 'config' && tag === faceEditor.lastSnapTag && now - faceEditor.lastSnapT < 700 && faceEditor.histIndex >= 0;
  faceEditor.lastSnapTag = tag; faceEditor.lastSnapT = now;
  const m = batch.faceMask.mask;
  const snap = { data: Uint8Array.from(m.data), config: cloneConfig(m.config) };
  if (coalesce) { faceEditor.history[faceEditor.histIndex] = snap; return; } // replace top (no new step)
  faceEditor.history = faceEditor.history.slice(0, faceEditor.histIndex + 1); // drop redo tail
  faceEditor.history.push(snap);
  if (faceEditor.history.length > 60) faceEditor.history.shift();
  faceEditor.histIndex = faceEditor.history.length - 1;
  updateUndoState();
}
function faceEditRestore(snap) {
  const batch = faceEditor.batch; if (!batch || !snap) return;
  const m = batch.faceMask.mask;
  m.data.set(snap.data);
  m.config = cloneConfig(snap.config);
  uploadMaskBuffer(m, batch.faceMask.maskAttr);
  syncConfigUniforms(batch); recolorEditHead(); deformEditHead(); updateHingeGizmo(); pushRegionConfigToStore();
  if (batch.faceMaskOverlay) recolorFaceMaskOverlay(batch.faceMaskOverlay);
}
export function faceEditUndo() { if (faceEditor.histIndex > 0) { faceEditor.histIndex -= 1; faceEditRestore(faceEditor.history[faceEditor.histIndex]); updateUndoState(); } }
export function faceEditRedo() { if (faceEditor.histIndex < faceEditor.history.length - 1) { faceEditor.histIndex += 1; faceEditRestore(faceEditor.history[faceEditor.histIndex]); updateUndoState(); } }
function updateUndoState() {
  setEditorState({ canUndo: faceEditor.histIndex > 0, canRedo: faceEditor.histIndex < faceEditor.history.length - 1 });
}
function faceEditResetHistory() { faceEditor.history = []; faceEditor.histIndex = -1; faceEditor.lastSnapTag = ''; faceEditCommit('init'); }

// CPU preview deform for the edit-head (T30 test-drive): the SAME config-driven form as
// the GPU faceDeformNode, on the static edit-head so sliders show the effect WYSIWYG.
const ZERO_EXPR = { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 };
const EXPR_ORDER = ['jawOpen', 'smile', 'pucker', 'blinkL', 'blinkR', 'browL', 'browR'];
const _defAxis = new THREE.Vector3();
function deformEditHead() {
  const mesh = faceEditor.editHead; const batch = faceEditor.batch;
  if (!mesh || !faceEditor.bindPositions || !batch) return;
  const fm = batch.faceMask; const N = fm.vertexCount; const D = fm.mask.data;
  const cfg = fm.mask.config; const hh = fm.mask.headHeight || 1;
  const e = runtime.debugFaceOverride || ZERO_EXPR;
  const drv = EXPR_ORDER.map((k) => e[k]); // driver index → scalar value
  const bind = faceEditor.bindPositions;
  const pos = mesh.geometry.getAttribute('position');
  for (let i = 0; i < N; i += 1) {
    const bx = bind[i * 3]; const by = bind[i * 3 + 1]; const bz = bind[i * 3 + 2];
    let x = bx; let y = by; let z = bz;
    for (let r = 0; r < 7; r += 1) {
      const c = cfg[r];
      const s = drv[c.driver] * (D[r * N + i] / 255) * c.amount;
      if (s === 0) continue;
      if (c.type === 1) { // hinge (Rodrigues about c.hingeAxis through c.hingeOrigin)
        const ox = c.hingeOrigin[0]; const oy = c.hingeOrigin[1]; const oz = c.hingeOrigin[2];
        _defAxis.set(c.hingeAxis[0], c.hingeAxis[1], c.hingeAxis[2]);
        if (_defAxis.lengthSq() < 1e-8) _defAxis.set(1, 0, 0); else _defAxis.normalize();
        const kx = _defAxis.x; const ky = _defAxis.y; const kz = _defAxis.z;
        const vx = bx - ox; const vy = by - oy; const vz = bz - oz;
        const cc = Math.cos(s); const sn = Math.sin(s); const oneC = 1 - cc;
        const dot = kx * vx + ky * vy + kz * vz;
        const crx = ky * vz - kz * vy; const cry = kz * vx - kx * vz; const crz = kx * vy - ky * vx;
        x += (ox + vx * cc + crx * sn + kx * dot * oneC) - bx;
        y += (oy + vy * cc + cry * sn + ky * dot * oneC) - by;
        z += (oz + vz * cc + crz * sn + kz * dot * oneC) - bz;
      } else { // translate along dir (mirrorX flips x by side), scaled by headHeight
        const mx = c.mirrorX ? Math.sign(bx - c.hingeOrigin[0]) : 1;
        x += c.dir[0] * mx * s * hh; y += c.dir[1] * s * hh; z += c.dir[2] * s * hh;
      }
    }
    pos.setXYZ(i, x, y, z);
  }
  pos.needsUpdate = true;
}

// Push the current model list to the store (badges).
function syncEditorStore() {
  const models = crowdBatches.filter((b) => b.faceMask).map((b, i) => ({ index: i, name: b.mesh.name }));
  setEditorState({ models, modelIndex: faceEditor.batchIndex });
}

// Editor camera presets around the edit head.
function faceEditFrameView(preset) {
  const mesh = faceEditor.editHead; if (!mesh) return;
  const box = new THREE.Box3().setFromObject(mesh);
  const c = new THREE.Vector3(); box.getCenter(c);
  const sz = new THREE.Vector3(); box.getSize(sz);
  const r = Math.max(sz.x, sz.y, sz.z) || 1;
  const d = r * 2.4;
  const off = { front: [0, 0, d], left: [-d, 0, 0.01], right: [d, 0, 0.01], '3q': [d * 0.6, r * 0.2, d * 0.7] }[preset] || [0, 0, d];
  controls.target.copy(c);
  camera.position.set(c.x + off[0], c.y + off[1], c.z + off[2]);
  controls.update();
}

// Load a .bin override in the editor (decode → guard → upload → recolor + preview).
function faceEditLoadMask(arrayBuffer) {
  const batch = faceEditor.batch; if (!batch) return;
  try {
    const mask = assertMaskFits(decodeFaceMask(arrayBuffer), batch.faceMask.vertexCount);
    batch.faceMask.mask = mask;
    uploadMaskBuffer(mask, batch.faceMask.maskAttr);
    syncConfigUniforms(batch);
    recolorEditHead();
    deformEditHead();
    updateHingeGizmo();
    pushRegionConfigToStore();
    faceEditCommit('load');
    setEditorState({ status: 'Loaded .bin' });
  } catch (err) {
    setEditorState({ status: `Load failed: ${err.message}` });
  }
}

// Register the editor actions the React UI calls (I.editorUI). main.js owns three;
// React only reads the store + invokes these.
export function registerEditorActions() {
  registerActions({
    setOpen(v) { faceEditor.active = v; setFaceEditMode(v); if (v) { syncEditorStore(); pushRegionConfigToStore(); updateHingeGizmo(); } setEditorState({ open: v }); },
    setModel(i) { faceEditor.batchIndex = i; setFaceEditMode(true); deformEditHead(); pushRegionConfigToStore(); updateHingeGizmo(); setEditorState({ modelIndex: i }); },
    setRegion(r) { faceEditor.region = r; faceEditor.hingeStage = 0; recolorEditHead(); pushRegionConfigToStore(); updateHingeGizmo(); setEditorState({ region: r }); },
    setTool(t) { faceEditSetTool(t); },
    setMode(m) { faceEditor.erase = m === 'erase'; faceEditor.hingeMode = m === 'hinge'; faceEditor.hingeStage = 0; updateHingeGizmo(); setEditorState({ mode: m }); },
    setRadius(v) { faceEditor.radius = v; setEditorState({ radius: v }); },
    setStrength(v) { faceEditor.strength = v; setEditorState({ strength: v }); },
    setSymmetric(v) { faceEditor.symmetric = v; setEditorState({ symmetric: v }); },
    // Patch the CURRENT region's deform config (driver/type/amount/dir/mirrorX) live.
    setRegionConfig(patch) {
      const batch = faceEditor.batch; if (!batch) return;
      const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
      Object.assign(batch.faceMask.mask.config[r], patch);
      syncConfigUniforms(batch); deformEditHead(); updateHingeGizmo(); pushRegionConfigToStore();
      faceEditCommit('config');
    },
    reseed(flip) { faceEditReseed(flip); },
    clearRegion() { faceEditClearRegion(); deformEditHead(); },
    undo() { faceEditUndo(); },
    redo() { faceEditRedo(); },
    save() { downloadFaceMasks(); },
    loadMask(buf) { faceEditLoadMask(buf); },
    setExpr(expr) { runtime.debugFaceOverride = expr; deformEditHead(); setEditorState({ expr }); },
    resetExpr() { runtime.debugFaceOverride = null; deformEditHead(); setEditorState({ expr: { ...ZERO_EXPR } }); },
    frameView(v) { faceEditFrameView(v); },
    setTexAlpha(v) { if (faceEditor.texAlpha) faceEditor.texAlpha.value = v; setEditorState({ overlays: { ...editorState.overlays, texAlpha: v } }); },
    setOverlay(k, v) {
      const o = { ...editorState.overlays, [k]: v };
      if (k === 'wireframe' && faceEditor.editHead) faceEditor.editHead.material.wireframe = v;
      if (k === 'maskCloud') { state.faceMaskDebug = v; refreshFaceMaskOverlays(); }
      if (k === 'crowd') for (const b of crowdBatches) b.source.traverse((c) => { if (c.name?.endsWith('-gpu-instances')) c.visible = v; });
      setEditorState({ overlays: o });
    }
  });
}
