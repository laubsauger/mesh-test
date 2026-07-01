import './styles.css';
import * as THREE from 'three';
import { RenderPipeline, StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import {
  Fn,
  add,
  attributeArray,
  cos,
  emissive,
  float,
  instanceIndex,
  mix,
  sign,
  sin,
  mrt,
  normalView,
  output,
  pass,
  storage,
  transformNormal,
  transformNormalToView,
  uint,
  uniform,
  vec3,
  vec4,
  vertexIndex
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { Inspector } from 'three/addons/inspector/Inspector.js';
import { CHOREOGRAPHIES, computeWalkerPhase } from './choreography.js';
import { boneSourceForIndex } from './bone-source.js';
import { assetUrl } from './asset-url.js';
import { POSE_EPS } from './pose/ep.js';
import { BODY_BONES } from './pose/topology.js';
import { probeEPs } from './pose/ep-probe.js';
import { RTMWPoseProvider } from './pose/rtmw-provider.js';
import { WorkerPoseProvider } from './pose/worker-pose-provider.js';
import { SidecarPoseProvider } from './pose/sidecar-pose-provider.js';
import { NativeCapturePoseProvider } from './pose/native-capture-provider.js';
import { drawOverlay } from './pose/overlay-2d.js';
import { toCanonical, mirrorCanonical } from './pose/observation-adapter.js';
import { Retargeter } from './pose/retargeter.js';
import { FaceExpressionExtractor } from './pose/face-expression.js';
import { generateFaceMask, decodeFaceMask, encodeFaceMask, assertMaskFits, FACE_REGIONS } from './pose/face-mask.js';
import { MESHY_BONES } from './pose/rig-map.js';
import { registerActions, setEditorState, editorState, actions as editorActions } from './editor/editor-store.js';
import { mountEditor } from './editor/mount.jsx';
import { KPT, NUM_KPTS, RTMW_VARIANTS, YOLO_RES_OPTIONS } from './pose/rtmw-constants.js';
import { CanonicalSmoother } from './pose/one-euro.js';
import { PoseRecorder, PosePlayer } from './pose/recorder.js';
import { PoseSkeleton3D } from './pose/skeleton-3d.js';
import { MediaPipeProvider } from './pose/mediapipe-provider.js';
import { toCanonicalFromMediaPipe, blendshapesToExpr } from './pose/mediapipe-adapter.js';

const canvas = document.querySelector('#scene');
const statusEl = document.querySelector('#status');
const posePipEl = document.querySelector('#pose-pip');
const poseVideoEl = document.querySelector('#pose-video');
const poseOverlayEl = document.querySelector('#pose-overlay');
const posePipLabel = document.querySelector('#pose-pip-label');
const calibOverlay = document.querySelector('#calib-overlay');
const calibTextEl = document.querySelector('#calib-text');
const calibSkeletonEl = document.querySelector('#calib-skeleton');
const calibSkeletonCtx = calibSkeletonEl.getContext('2d');

const preloaderEl = document.querySelector('#preloader');
const preloaderBar = document.querySelector('#preloader-bar');
const preloaderLabel = document.querySelector('#preloader-label');
const preloaderPct = document.querySelector('#preloader-pct');

// Crowd GPU-skinning packs EVERY instance's skinned vertices into ONE storage
// buffer (walkers × vertexCount × 2 vec4 = ×32 bytes). WebGPURenderer requests a
// 'compatibility' adapter, which caps maxStorageBufferBindingSize at 128 MiB → a
// HARD freeze just past ~418 of a ~10k-vert mesh (the binding overflows, the device
// rejects it, the loop hangs). Request a CORE adapter at its max binding/buffer size
// and hand the device to the renderer — pushes the wall out to the adapter's real
// limit (GiB-scale on desktop). No WebGPU → throw (no WebGL fallback exists anyway).
async function createHighLimitDevice() {
  if (!navigator.gpu) throw new Error('WebGPU unavailable (no navigator.gpu)');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU: no GPU adapter');
  const lim = adapter.limits;
  return adapter.requestDevice({
    requiredFeatures: [...adapter.features], // mirror the renderer: enable all the adapter has
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize
    }
  });
}
const device = await createHighLimitDevice();

const renderer = new WebGPURenderer({
  canvas,
  antialias: true,
  alpha: true,
  device
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.inspector = new Inspector();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x010102);
scene.fog = new THREE.FogExp2(0x010102, 0.034);

const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 180);
camera.position.set(0, 1.5, 5.5); // dev default: close on a single performer

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.055;
controls.target.set(0, 0.95, 0);
// Free framing: full vertical orbit, wide zoom range, screen-space pan.
controls.maxPolarAngle = Math.PI;
controls.minPolarAngle = 0;
controls.minDistance = 0.5;
controls.maxDistance = 300;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.panSpeed = 1.0;
controls.zoomSpeed = 1.2;
const cameraOffset = new THREE.Vector3();

// T29: face-mask editor pointer handlers (paint the edit head when edit mode is on;
// handlers are no-ops otherwise). Right-click erases → suppress its context menu.
renderer.domElement.addEventListener('pointerdown', (e) => onFaceEditDown(e));
renderer.domElement.addEventListener('pointermove', (e) => onFaceEditMove(e));
window.addEventListener('pointerup', () => onFaceEditUp());
renderer.domElement.addEventListener('contextmenu', (e) => { if (faceEditor.active) e.preventDefault(); });

// 3D debug overlay: the canonical pose (what drives the rig) drawn as a line
// skeleton at the pose performer, to spot retarget discrepancies vs the mesh.
const poseDebugGeom = new THREE.BufferGeometry();
poseDebugGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(BODY_BONES.length * 2 * 3), 3));
const poseDebugLines = new THREE.LineSegments(
  poseDebugGeom,
  new THREE.LineBasicMaterial({ color: 0x39ff14, depthTest: false, transparent: true })
);
poseDebugLines.frustumCulled = false;
poseDebugLines.renderOrder = 999;
poseDebugLines.visible = false;
scene.add(poseDebugLines);

// Magenta = the ACTUAL driven rig bones (placed on the mesh via the instance
// matrix). Green pose vs magenta bones → see exactly how retarget interpreted it.
const RIG_EDGE_NAMES = [
  ['Hips', 'Spine'], ['Spine', 'Spine01'], ['Spine01', 'Spine02'], ['Spine02', 'neck'], ['neck', 'Head'],
  ['Spine02', 'LeftShoulder'], ['LeftShoulder', 'LeftArm'], ['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'],
  ['Spine02', 'RightShoulder'], ['RightShoulder', 'RightArm'], ['RightArm', 'RightForeArm'], ['RightForeArm', 'RightHand'],
  ['Hips', 'LeftUpLeg'], ['LeftUpLeg', 'LeftLeg'], ['LeftLeg', 'LeftFoot'],
  ['Hips', 'RightUpLeg'], ['RightUpLeg', 'RightLeg'], ['RightLeg', 'RightFoot']
];
const poseRigGeom = new THREE.BufferGeometry();
poseRigGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(RIG_EDGE_NAMES.length * 2 * 3), 3));
const poseRigLines = new THREE.LineSegments(
  poseRigGeom,
  new THREE.LineBasicMaterial({ color: 0xff2bd6, depthTest: false, transparent: true })
);
poseRigLines.frustumCulled = false;
poseRigLines.renderOrder = 1000;
poseRigLines.visible = false;
scene.add(poseRigLines);

// Joint dots (lines are 1px in WebGPU — dots make the overlays readable).
const poseDebugPoints = new THREE.Points(poseDebugGeom, new THREE.PointsMaterial({ color: 0x39ff14, size: 0.05, sizeAttenuation: true, depthTest: false, transparent: true }));
const poseRigPoints = new THREE.Points(poseRigGeom, new THREE.PointsMaterial({ color: 0xff2bd6, size: 0.055, sizeAttenuation: true, depthTest: false, transparent: true }));
for (const p of [poseDebugPoints, poseRigPoints]) { p.frustumCulled = false; p.renderOrder = 1001; p.visible = false; scene.add(p); }
const _instMat = new THREE.Matrix4();
const _vA = new THREE.Vector3();

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(360, 360),
  new THREE.MeshStandardMaterial({
    color: 0x050507,
    roughness: 0.58,
    metalness: 0.18
  })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.02;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(360, 180, 0x193047, 0x090d13);
grid.material.transparent = true;
grid.material.opacity = 0.28;
grid.position.y = 0.005;
scene.add(grid);

const hemi = new THREE.HemisphereLight(0x354765, 0x030204, 0.95);
scene.add(hemi);

const fill = new THREE.AmbientLight(0x171c2a, 0.52);
scene.add(fill);

const key = new THREE.DirectionalLight(0xb9dcff, 5.8);
key.position.set(-7, 10, 7);
key.castShadow = true;
// 2048 so the single shadow map still resolves per-character shadows once the
// frustum is stretched to cover a large crowd (see updateArmyBounds).
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 40;
key.shadow.camera.left = -18;
key.shadow.camera.right = 18;
key.shadow.camera.top = 18;
key.shadow.camera.bottom = -18;
key.shadow.bias = -0.00035;
key.shadow.normalBias = 0.035;
scene.add(key);

// Pink rim — kept modest so it accents character edges without flooding the
// whole floor (a strong non-shadow fill washes out the key light's shadows).
const rim = new THREE.DirectionalLight(0xff3f81, 3);
rim.position.set(8, 5.5, -8);
scene.add(rim);

const crossFill = new THREE.DirectionalLight(0x8fbfff, 2.4);
crossFill.position.set(6, 4.8, 8);
scene.add(crossFill);

const heroSpot = new THREE.SpotLight(0xe8f7ff, 24, 34, Math.PI * 0.3, 0.6, 1.25);
heroSpot.position.set(0, 9, 9);
heroSpot.target.position.set(0, 0.8, 0);
heroSpot.castShadow = true;
heroSpot.shadow.mapSize.set(1024, 1024);
heroSpot.shadow.bias = -0.00025;
heroSpot.shadow.normalBias = 0.03;
scene.add(heroSpot);
scene.add(heroSpot.target);

const cameraFill = new THREE.SpotLight(0xdcecff, 10, 80, Math.PI * 0.4, 0.92, 0.7);
cameraFill.position.copy(camera.position);
cameraFill.target.position.copy(controls.target);
cameraFill.castShadow = false;
scene.add(cameraFill);
scene.add(cameraFill.target);

// Lower decay so these roaming colored accents spread across the crowd instead
// of lighting only a tiny sliver under themselves.
const floorGlow = new THREE.PointLight(0x38fff0, 12, 22, 1.4);
floorGlow.position.set(-5, 0.75, 3.2);
scene.add(floorGlow);

const backGlow = new THREE.PointLight(0x7d2dff, 16, 32, 1.3);
backGlow.position.set(0, 2.4, -8);
scene.add(backGlow);

const sideGlow = new THREE.PointLight(0xff315e, 14, 22, 1.4);
sideGlow.position.set(5.5, 2.6, 2.5);
scene.add(sideGlow);

const worldUp = new THREE.Vector3(0, 1, 0);
const performerWorldPosition = new THREE.Vector3();
const cameraDirection = new THREE.Vector3();
const cameraSide = new THREE.Vector3();
const instancePosition = new THREE.Vector3();
const instanceScale = new THREE.Vector3();
const instanceQuaternion = new THREE.Quaternion();
const instanceEuler = new THREE.Euler(0, 0, 0, 'YXZ');
const normalizationMatrix = new THREE.Matrix4();
const transformMatrix = new THREE.Matrix4();

// Bounds of the whole army (XZ spread), used to fan the accent lights and
// shadow frusta across every performer instead of clustering them at origin.
const armyBounds = {
  centerX: 0,
  centerZ: 0,
  halfX: 4,
  halfZ: 4,
  radius: 6
};

const scenePass = pass(scene, camera);
scenePass.setMRT(mrt({
  output,
  normal: normalView,
  emissive
}));

const scenePassColor = scenePass.getTextureNode('output');
const scenePassNormal = scenePass.getTextureNode('normal');
const scenePassDepth = scenePass.getTextureNode('depth');
const scenePassEmissive = scenePass.getTextureNode('emissive');
const aoPass = ao(scenePassDepth, scenePassNormal, camera);
aoPass.radius.value = 0.95;
aoPass.thickness.value = 1.35;
aoPass.distanceFallOff.value = 0.62;
aoPass.distanceExponent.value = 1.35;
aoPass.scale.value = 0.85;
aoPass.samples.value = 8;
aoPass.resolutionScale = 0.5;

const bloomPass = bloom(scenePassEmissive, 0.5, 0.45, 0.2);
const renderPipeline = new RenderPipeline(renderer);
// AO toggle: mix the AO term toward 1.0 (no darkening) via a uniform, so the
// checkbox is a real on/off — not just the intensity slider near zero.
const aoFactor = uniform(1);
const aoTerm = mix(float(1), aoPass.getTextureNode().r, aoFactor);
renderPipeline.outputNode = scenePassColor
  .mul(vec4(vec3(aoTerm), 1))
  .add(bloomPass);

const loader = new GLTFLoader();
const timer = new THREE.Timer();
timer.connect(document);
const assetCache = new Map();
const walkers = [];
const crowdBatches = [];
const skeletonStates = new Map();
const statsState = {
  fps: 0,
  calls: 0,
  triangles: 0,
  lines: 0,
  points: 0,
  geometries: 0,
  textures: 0
};
let manifest = null;
let rebuildId = 0;

// Each "lit performer" spawns a real spotlight rig — forward rendering can't
// scale to hundreds, so hard-cap it. The army itself is lit by the scene lights
// regardless; these rigs are extra per-character hero lighting for a few.
const MAX_PERFORMER_RIGS = 24;

// "idle" animation = load geometry/skeleton but play no clip → the rig rests at
// its bind pose. This is the base for pose mode: with no mixer advancing, the
// clip never fights the pose writer. (A proper blendable idle clip is future
// work — see SPEC.) Lives alongside the real clip names in the Animation list.
const ANIM_IDLE = 'idle';

const state = {
  selectedMeshIds: [],
  animationName: 'Walking',
  arrangement: 'line',
  movement: 'in place',
  count: 1,
  speed: 1,
  choreography: 'desync',
  choreoDelay: 0.6,
  posePerformerCount: 1,
  // webgpu EP (only viable one here): wasm OOM-crashes the tab — this app already
  // holds ~206MB of GLB chars, and wasm would load the 369MB model into the wasm
  // heap on top (B5). webgpu keeps the model on the GPU so it fits, but shares the
  // device with the renderer → contends (inference climbs under load, V23). The
  // pose loop yields a render frame between inferences to ease that; a dedicated
  // worker (T20) is the real isolation.
  poseEP: 'webgpu',
  poseEnabled: true, // dev default: pose on at load
  poseWorker: true, // run inference in a Worker (own GPU device, off main thread)
  poseBackend: 'worker', // 'worker' = onnxruntime-web (webgpu) | 'sidecar' = native ORT (TRT/CUDA) over WS | 'mediapipe' = in-browser BlazePose+Face
  poseMediaPipeHands: false, // mediapipe: also run the 21-pt hand landmarker (extra cost)
  poseSidecarDebug: false, // sidecar: console per-frame timing/flow trace
  poseReadback: 'bitmap', // sidecar frame snapshot: 'bitmap'(createImageBitmap) | 'videoframe'(WebCodecs)
  posePreview: true, // sidecar-native: push a low-res webcam JPEG so the overlay isn't blind
  poseNativeDevice: 0, // sidecar-native: camera index the sidecar opens (use Scan Cameras to find it)
  scanCameras: () => scanCameras(), // button: ask the sidecar which camera indices are live
  poseSendMaxSide: 640, // sidecar: longest edge shipped over the wire. The models only need
  // yolo@320 + a 384×288 crop, so full-res (1280 = 3.6MB/frame) just bloats readback+transport
  // (~19ms + ~27ms) for no accuracy gain. 640 = 0.9MB → ~4× less data. Bump if a DISTANT
  // subject's crop gets too soft; watch Readback + Transport in Pose Stats for the sweet spot.
  poseRtmwVariant: 'l', // rtmw3d size: l(faster, 219MB) | x(largest, 352MB). Both 3D, 384×288, 3-axis.
  // NOTE: rtmw3d only ships as l and x — there is NO 3D "m" (the "m" upstream is 2D rtmw, no depth).
  poseYoloRes: 320, // yolo detector input res: 320/384/512 (files committed)
  poseKptThresh: 0.3,
  faceMaskDebug: false, // T27: show face-mask region weights as a colored vertex cloud
  faceMaskRegion: 'jaw', // which region the overlay colors
  faceDrive: true, // T28: apply face deform (jaw/lips/lids/brows) from expression scalars
  faceJawAngle: 0.5, // max jaw hinge open (radians, ~28°) at jawOpen=1
  faceLipDrop: 0.06, // lower-lip drop as a fraction of head height at jawOpen=1
  faceCornerMove: 0.05, // mouth-corner spread(smile)/narrow(pucker), fraction of head height
  faceLidDrop: 0.06, // upper-lid close travel, fraction of head height at blink=1
  faceBrowRaise: 0.04, // brow raise, fraction of head height at brow=1
  poseRetarget: true,
  poseMirrorX: false,
  poseMirror: true,
  poseDepthScale: 1, // 1 = accurate (>1 exaggerates depth → arms over-rotate)
  poseArmLimit: 110, // tighter joint limit for arms (over-rotation guard)
  poseSmoothing: true,
  poseSmoothMinCutoff: 2,
  poseSmoothBeta: 0.02,
  poseJointLimit: 150,
  // Body yaw ON: the torso frame is now built from the RAW 3D hip-axis (no depth
  // amplification), so turn/tilt/pivot come straight from the pelvis-aligned data.
  // The horizontal hip separation dominates the sign → stable near frontal (the old
  // yawGain-amplified scalar is what spun it, §B8/§B8b). Off = lean-only.
  poseBodyYaw: true, // 2D-foreshortening yaw (flip-safe, validated). Head from nose-offset (great), body from shoulder width
  poseFacingDeadzone: 0.17, // rad — below this |yaw| → 0 (frontal holds still; kills near-frontal flips)
  poseFacingSmooth: 0.15, // yaw temporal smoothing (lower = smoother/laggier)
  poseFacingBodyGain: 1.0, // body (torso) yaw strength — the shoulder read is ~half the head; raise to make the torso follow the arrow
  poseHeadPitchGain: 2.0, // head forward/back nod strength (nose-Y). 0 = off; negate if it nods the wrong way
  poseClavicle: false, // shoulder shrug/protraction — unproven, opt-in
  poseFollow: 0.5, // legacy per-frame blend (used only when Rig Smoothing = 0)
  poseSmoothTime: 35, // ms half-life for the frame-rate-INDEPENDENT rig blend. Lower = snappier/less lag, higher = smoother. 0 = off (use poseFollow)
  poseSwingTwist: true,
  poseWristTwist: true,
  poseTwistSmooth: 0.12,
  poseHeadGain: 0.7,
  poseFacingDebug: false, // draw detected body/head facing as arrows (PiP compass + 3D on performer)
  poseRejectOutliers: true,
  poseFlipReject: 70, // per-bone: a target jump > this° in one pose-frame is held unless it persists (kills leg/hand 180° flips). 180 = off
  poseMaxJump: 0.5,
  poseHoldMs: 2000,
  poseBoneGate: false, // OFF: 3D limb-length gate rejects 32-40% of TURNED frames (monocular z varies) → freeze. Velocity gate covers teleports.
  poseGrounding: true,
  poseGroundFollow: 0.3,
  poseOverlay: true,
  poseDebug3D: false,
  poseDebugScale: 1, // fine-tune on top of the auto mesh-height scale (1 = matched to the rig)
  poseDebugHeight: 1,
  poseDetectEveryN: 2,
  poseRecording: false,
  poseReplaying: false,
  spacing: 2.8,
  scale: 1,
  sizeVariance: 0,
  proportionVariance: 0,
  travel: 0,
  disorder: 0,
  wanderRadius: 4,
  wanderSpeed: 0.35,
  floorLightBase: 9,
  backLightBase: 12,
  keyLight: 5.8,
  rimLight: 3,
  crossFill: 2.4,
  cameraFill: 10,
  performerKey: 3.2,
  performerFill: 0,
  performerRim: 0,
  performerLightCount: 0,
  exposure: 1.15,
  fog: 0.034,
  ao: 0.85,
  aoEnabled: true,
  bloom: 0.5,
  lightMotion: true,
  glowLights: true,
  quality: 'performance',
  emissiveBoost: 1,
  cameraDrift: true,
  cameraDriftSpeed: 0.4,
  cameraDistance: 5.5,
  floorGrid: true,
  glowShadows: false,
  shufflePhase() {
    for (const walker of walkers) walker.seed = Math.random();
  },
  applyQualityPreset() {
    applyQualityPreset(state.quality);
  },
  toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  },
  calibratePose() {
    calibPhase = 'ready';
    calibPhaseStart = performance.now();
  },
  // V19: dump the peak-held joint-limit clamp overflow (worst since last reset) as
  // a copyable console table, then reset peaks so the next gesture starts clean.
  dumpClamps() {
    if (!lastPoseRetargeter) { console.warn('[clamp] no pose retargeter active'); return; }
    const rows = lastPoseRetargeter.clampReport();
    if (!rows.length) { console.log('[clamp] nothing clamped since reset'); }
    else console.table(rows.map((r) => ({
      bone: r.bone, 'wanted°': Math.round(r.raw), 'cap°': r.max, 'over°': Math.round(r.over),
      hits: r.hits, frames: r.frames, '%': Math.round(r.pct * 100)
    })));
    lastPoseRetargeter.resetClampPeak();
  },
  savePoseRecording() {
    if (!poseRecorder.length) { statusEl.textContent = 'Nothing recorded.'; return; }
    const blob = new Blob([JSON.stringify(poseRecorder.toSession())], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pose-${poseRecorder.length}f.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    statusEl.textContent = `Saved ${poseRecorder.length} frames.`;
  },
  loadPoseRecording() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        poseRecorder.load(JSON.parse(await file.text()));
        statusEl.textContent = `Loaded ${poseRecorder.length} frames — enable Replay.`;
      } catch (error) {
        statusEl.textContent = `Load failed: ${error.message}`;
      }
    };
    input.click();
  },
  runEpProbe() {
    statusEl.textContent = 'EP probe running… (console for detail)';
    probeEPs(POSE_EPS, { runs: 20 })
      .then(({ results, winner }) => {
        const summary = results
          .map((r) => (r.ok ? `${r.ep} ${r.median.toFixed(0)}ms` : `${r.ep} FAIL`))
          .join(' · ');
        statusEl.textContent = `EP probe: ${summary}${winner ? ` → win ${winner}` : ''}`;
      })
      .catch((error) => {
        statusEl.textContent = `EP probe failed: ${error.message}`;
        console.error(error);
      });
  }
};

init().catch((error) => {
  console.error(error);
  statusEl.textContent = error.message;
  const label = document.querySelector('#preloader-label');
  if (label) label.textContent = error.message;
});

async function init() {
  setPreloader(0.04, 'Initializing renderer…');
  await renderer.init();
  resize();
  renderer.setAnimationLoop(render);

  // Drift = OrbitControls.autoRotate: orbits around the CURRENT target at the
  // user's manual distance (zoom/pan persist). Pauses during an active drag and
  // resumes — so manual framing and drift coexist.
  controls.autoRotate = state.cameraDrift;
  controls.autoRotateSpeed = state.cameraDriftSpeed;

  // Fullscreen hides the status line (clean VJ output); resize to fill.
  document.addEventListener('fullscreenchange', () => {
    document.body.classList.toggle('is-fullscreen', !!document.fullscreenElement);
    resize();
  });

  setPreloader(0.1, 'Loading manifest…');
  const response = await fetch(assetUrl('bipeds-manifest.json'));
  if (!response.ok) {
    throw new Error('Missing biped manifest. Run npm run optimize:bipeds.');
  }

  manifest = await response.json();
  if (!manifest.meshes?.length) throw new Error('Biped manifest is empty.');

  buildInspectorControls();
  registerEditorActions(); // T30: wire the React editor's actions
  mountEditor();           // mount the React editor overlay (hidden until opened)

  // Stream download progress for the first build into the preloader bar.
  setPreloader(0.15, 'Downloading characters…');
  onAssetProgress = (loaded, total) => {
    const frac = total > 0 ? loaded / total : 0;
    const mb = (bytes) => (bytes / 1048576).toFixed(0);
    setPreloader(0.15 + 0.85 * frac, `Downloading characters… ${mb(loaded)} / ${mb(total)} MB`);
  };
  await rebuildWalkers();
  onAssetProgress = null;
  hidePreloader();

  window.addEventListener('resize', resize);

  // dev default: auto-start pose (prompts for the webcam on load).
  if (state.poseEnabled) startPose();
}

function setPreloader(frac, label) {
  const pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
  if (preloaderBar) preloaderBar.style.width = `${pct}%`;
  if (preloaderPct) preloaderPct.textContent = `${pct}%`;
  if (label && preloaderLabel) preloaderLabel.textContent = label;
}

function hidePreloader() {
  setPreloader(1, 'Ready');
  preloaderEl?.classList.add('is-hidden');
}

function buildInspectorControls() {
  // dev default: single mesh selected (pose drives one performer).
  state.selectedMeshIds = manifest.meshes.length ? [manifest.meshes[0].id] : [];

  const performanceGroup = renderer.inspector.createParameters('VJ Layer / Scene');
  performanceGroup.add(state, 'count', 1, 500).name('Count').onChange((value) => {
    state.count = Math.round(value);
    rebuildWalkers();
  });
  performanceGroup.add(state, 'animationName', animationNames()).name('Animation').onChange((value) => {
    state.animationName = value;
    rebuildWalkers();
  });
  performanceGroup.add(state, 'arrangement', ['line', 'arc', 'stagger', 'circle', 'grid', 'random', 'spiral']).name('Arrangement').onChange((value) => {
    state.arrangement = value;
    arrangeWalkers();
  });
  performanceGroup.add(state, 'movement', ['in place', 'lane loop', 'local wander', 'space wander', 'orbit drift']).name('Movement').onChange(() => arrangeWalkers());
  performanceGroup.add(state, 'speed', 0, 3, 0.01).name('Speed');
  performanceGroup.add(state, 'choreography', CHOREOGRAPHIES).name('Choreography');
  performanceGroup.add(state, 'choreoDelay', 0, 1, 0.01).name('Choreo Delay');
  performanceGroup.add(state, 'spacing', 0.2, 24, 0.05).name('Spacing').onChange(() => arrangeWalkers());
  performanceGroup.add(state, 'scale', 0.3, 2.5, 0.01).name('Scale').onChange(() => {
    arrangeWalkers();
  });
  performanceGroup.add(state, 'sizeVariance', 0, 1, 0.01).name('Size Variance');
  performanceGroup.add(state, 'proportionVariance', 0, 1, 0.01).name('Proportions');
  performanceGroup.add(state, 'travel', 0, 8, 0.01).name('Travel').onChange(() => arrangeWalkers());
  performanceGroup.add(state, 'disorder', 0, 1, 0.01).name('Disorder').onChange(() => arrangeWalkers());
  performanceGroup.add(state, 'wanderRadius', 0, 40, 0.1).name('Wander Radius');
  performanceGroup.add(state, 'wanderSpeed', 0, 3, 0.01).name('Wander Speed');
  performanceGroup.add(state, 'toggleFullscreen').name('Fullscreen');
  performanceGroup.add(state, 'cameraDrift').name('Camera Drift').listen().onChange((value) => {
    controls.autoRotate = value;
  });
  performanceGroup.add(state, 'cameraDriftSpeed', -3, 3, 0.05).name('Drift Speed').onChange((value) => {
    controls.autoRotateSpeed = value;
  });
  performanceGroup.add(state, 'cameraDistance', controls.minDistance, controls.maxDistance, 0.1)
    .name('Camera Distance').onChange((value) => {
      cameraOffset.copy(camera.position).sub(controls.target);
      if (cameraOffset.lengthSq() < 1e-6) cameraOffset.set(0, 0, 1);
      camera.position.copy(controls.target).addScaledVector(cameraOffset.normalize(), value);
      controls.update();
    });
  performanceGroup.add(state, 'floorGrid').name('Floor Grid').onChange((value) => {
    grid.visible = value;
  });
  performanceGroup.add(state, 'shufflePhase').name('Shuffle Phase');

  const lightingGroup = renderer.inspector.createParameters('VJ Layer / Lighting');
  lightingGroup.add(state, 'quality', ['performance', 'balanced', 'cinematic']).name('Quality').onChange((value) => {
    applyQualityPreset(value);
  });
  lightingGroup.add(state, 'applyQualityPreset').name('Apply Quality');
  lightingGroup.add(state, 'exposure', 0.1, 2.2, 0.01).name('Exposure').onChange((value) => {
    renderer.toneMappingExposure = value;
  });
  lightingGroup.add(state, 'fog', 0, 0.09, 0.001).name('Fog').onChange((value) => {
    scene.fog.density = value;
  });
  lightingGroup.add(state, 'keyLight', 0, 14, 0.1).name('Key').onChange((value) => {
    key.intensity = value;
  });
  lightingGroup.add(state, 'rimLight', 0, 16, 0.1).name('Rim').onChange((value) => {
    rim.intensity = value;
  });
  lightingGroup.add(state, 'crossFill', 0, 8, 0.1).name('Cross Fill').onChange((value) => {
    crossFill.intensity = value;
  }).listen();
  lightingGroup.add(state, 'cameraFill', 0, 24, 0.1).name('Camera Fill').onChange((value) => {
    cameraFill.intensity = value;
  }).listen();
  lightingGroup.add(state, 'performerKey', 0, 10, 0.1).name('Performer Key').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  }).listen();
  lightingGroup.add(state, 'performerFill', 0, 6, 0.05).name('Performer Fill').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  }).listen();
  lightingGroup.add(state, 'performerRim', 0, 8, 0.05).name('Performer Rim').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  }).listen();
  lightingGroup.add(state, 'performerLightCount', 0, MAX_PERFORMER_RIGS, 1).name('Lit Performers').onChange((value) => {
    state.performerLightCount = Math.min(Math.round(value), MAX_PERFORMER_RIGS);
    rebuildWalkers();
  });
  lightingGroup.add(state, 'glowLights').name('Glow Lights').listen();
  lightingGroup.add(state, 'lightMotion').name('Light Motion').listen();
  lightingGroup.add(state, 'floorLightBase', 0, 24, 0.1).name('Floor Glow');
  lightingGroup.add(state, 'backLightBase', 0, 28, 0.1).name('Back Glow');
  lightingGroup.add(state, 'aoEnabled').name('AO Enabled').listen().onChange((on) => {
    aoFactor.value = on ? 1 : 0;
  });
  lightingGroup.add(state, 'ao', 0, 3, 0.01).name('GTAO').onChange((value) => {
    aoPass.scale.value = value;
  }).listen();
  lightingGroup.add(state, 'bloom', 0, 3, 0.01).name('Bloom').onChange((value) => {
    bloomPass.strength.value = value;
  }).listen();
  lightingGroup.add(state, 'glowShadows').name('Glow Shadows (slow)').onChange(applyGlowShadows);
  lightingGroup.add(state, 'emissiveBoost', 0, 5, 0.01).name('Emissive').onChange(() => {
    for (const batch of crowdBatches) applyEmissiveBoost(batch.source);
  });

  const meshGroup = renderer.inspector.createParameters('VJ Layer / Meshes');
  for (const mesh of manifest.meshes) {
    const meshState = { enabled: state.selectedMeshIds.includes(mesh.id) };
    meshGroup.add(meshState, 'enabled').name(mesh.name).onChange((enabled) => {
      if (enabled) {
        state.selectedMeshIds = [...new Set([...state.selectedMeshIds, mesh.id])];
      } else {
        state.selectedMeshIds = state.selectedMeshIds.filter((id) => id !== mesh.id);
      }
      rebuildWalkers();
    });
  }

  const poseGroup = renderer.inspector.createParameters('VJ Layer / Pose');
  poseGroup.add(state, 'poseEnabled').name('Webcam Pose').listen().onChange((on) => {
    if (on) startPose();
    else stopPose();
  });
  // No artificial cap — all pose performers share one retarget (O(1)), so this
  // can match Count. 500 = the Count slider's max.
  poseGroup.add(state, 'posePerformerCount', 0, 500, 1).name('Pose Performers').onChange((value) => {
    state.posePerformerCount = Math.round(value);
    for (const walker of walkers) walker.boneSource = boneSourceForIndex(walker.index, state.posePerformerCount);
  });
  poseGroup.add(state, 'poseWorker').name('Inference Worker').listen();
  poseGroup.add(state, 'poseEP', POSE_EPS).name('Pose EP');
  poseGroup.add(state, 'poseBackend', ['worker', 'sidecar', 'sidecar-native', 'mediapipe']).name('Backend (web|native)').onChange(() => {
    // benchmark switch: tear down the running provider so startPose rebuilds with
    // the chosen backend (onnxruntime-web worker vs the native TRT/CUDA sidecar).
    if (poseProvider) { stopPose(); if (state.poseEnabled) startPose(); }
  });
  poseGroup.add(state, 'poseMediaPipeHands').name('MediaPipe Hands').onChange(() => {
    if (poseProvider instanceof MediaPipeProvider) { stopPose(); if (state.poseEnabled) startPose(); }
  });
  // NOT a model resolution — the downscaled webcam frame shipped to the sidecar
  // (transport only; it letterboxes→yolo@≤640 and crops→rtmw 384×288 from it). Nothing
  // consumes >640, so the slider caps there +headroom (768) for distant subjects only.
  poseGroup.add(state, 'poseSendMaxSide', 320, 768, 32).name('Sidecar Wire Frame px').onChange((v) => {
    if (poseProvider instanceof SidecarPoseProvider) poseProvider.sendMaxSide = Math.round(v);
  });
  poseGroup.add(state, 'poseSidecarDebug').name('Sidecar Debug (console)').onChange((on) => {
    if (poseProvider instanceof SidecarPoseProvider) poseProvider.debug = on;
  });
  // A/B the frame-snapshot method live — watch 'Capture' in Pose Stats. videoframe =
  // WebCodecs new VideoFrame(video) (sync, no decode/copy); bitmap = createImageBitmap.
  poseGroup.add(state, 'poseReadback', ['bitmap', 'videoframe']).name('Sidecar Capture').onChange((v) => {
    if (poseProvider instanceof SidecarPoseProvider) poseProvider.readback = v;
  });
  // Native-capture: the backend owns the camera, so the only way the overlay isn't
  // blind is a low-res preview pushed back. Toggle live; cost shows in Pose Stats.
  poseGroup.add(state, 'posePreview').name('Native Preview Feed').onChange((on) => {
    if (poseProvider instanceof NativeCapturePoseProvider) poseProvider.setPreview(on);
  });
  poseGroup.add(state, 'scanCameras').name('Scan Cameras (find index)');
  poseGroup.add(state, 'poseNativeDevice', 0, 8, 1).name('Native Device').onChange(() => {
    if (poseProvider instanceof NativeCapturePoseProvider) { stopPose(); if (state.poseEnabled) startPose(); }
  });
  // Model selection — a reload (swaps the .onnx), so restart pose. Variant applies
  // to the worker backend here; the sidecar picks its model via launch flags.
  poseGroup.add(state, 'poseRtmwVariant', RTMW_VARIANTS).name('RTMW Variant').onChange(() => {
    if (poseProvider) { stopPose(); if (state.poseEnabled) startPose(); }
  });
  poseGroup.add(state, 'poseYoloRes', YOLO_RES_OPTIONS).name('YOLO Res').onChange(() => {
    if (poseProvider) { stopPose(); if (state.poseEnabled) startPose(); }
  });
  poseGroup.add(state, 'poseKptThresh', 0, 1, 0.01).name('Kpt Threshold');
  poseGroup.add(state, 'poseRetarget').name('Pose Retarget').listen();
  poseGroup.add(state, 'poseMirror').name('Mirror Pose').listen();
  poseGroup.add(state, 'poseMirrorX').name('Mirror X (raw)').listen();
  poseGroup.add(state, 'poseDepthScale', 0, 3, 0.01).name('Depth Scale');
  poseGroup.add(state, 'poseSmoothing').name('Smoothing').listen();
  poseGroup.add(state, 'poseSmoothMinCutoff', 0.1, 8, 0.1).name('Smooth Min Cutoff');
  poseGroup.add(state, 'poseSmoothBeta', 0, 0.2, 0.001).name('Smooth Beta');
  poseGroup.add(state, 'poseJointLimit', 30, 180, 1).name('Joint Limit °');
  poseGroup.add(state, 'poseArmLimit', 30, 180, 1).name('Arm Limit °');
  poseGroup.add(state, 'poseSmoothTime', 0, 150, 1).name('Rig Smoothing (ms)');
  poseGroup.add(state, 'poseFollow', 0.05, 1, 0.01).name('Pose Follow (legacy)');
  poseGroup.add(state, 'poseBodyYaw').name('Body Yaw (2D turn)').listen();
  poseGroup.add(state, 'poseFacingDeadzone', 0, 0.6, 0.01).name('Yaw Deadzone');
  poseGroup.add(state, 'poseFacingSmooth', 0.02, 0.5, 0.01).name('Yaw Smooth');
  poseGroup.add(state, 'poseFacingBodyGain', 0.5, 3, 0.05).name('Body Yaw Gain');
  poseGroup.add(state, 'poseHeadPitchGain', -4, 4, 0.1).name('Head Pitch (nod)');
  poseGroup.add(state, 'poseClavicle').name('Clavicles (shrug)').listen();
  poseGroup.add(state, 'poseSwingTwist').name('Swing-Twist').listen();
  poseGroup.add(state, 'poseTwistSmooth', 0.02, 1, 0.01).name('Twist Smooth');
  poseGroup.add(state, 'poseWristTwist').name('Wrist Twist (hands)').listen();
  poseGroup.add(state, 'poseHeadGain', 0.1, 1.5, 0.01).name('Head Gain');
  poseGroup.add(state, 'poseRejectOutliers').name('Reject Jumps').listen();
  poseGroup.add(state, 'poseFlipReject', 20, 180, 5).name('Bone Flip Reject (°)');
  poseGroup.add(state, 'poseMaxJump', 0.1, 1.5, 0.01).name('Max Jump');
  poseGroup.add(state, 'poseHoldMs', 200, 6000, 50).name('Hold Last (ms)');
  poseGroup.add(state, 'calibratePose').name('Calibrate (hold A-pose)');
  poseGroup.add(state, 'dumpClamps').name('Dump Clamps → console');
  poseGroup.add(state, 'poseBoneGate').name('Bone-Length Gate').listen();
  poseGroup.add(state, 'poseGrounding').name('Grounding (feet)').listen();
  poseGroup.add(state, 'poseGroundFollow', 0.05, 1, 0.01).name('Ground Smooth');
  poseGroup.add(state, 'poseOverlay').name('2D Overlay').listen();
  poseGroup.add(state, 'poseFacingDebug').name('Facing Debug (arrows)').listen().onChange((on) => {
    if (!on) { facingArrowBody.visible = false; facingArrowHead.visible = false; }
  });
  poseGroup.add(state, 'poseDebug3D').name('Pose Debug 3D').listen();
  poseGroup.add(state, 'poseDebugScale', 0.2, 3, 0.01).name('Debug Scale');
  poseGroup.add(state, 'poseDebugHeight', 0, 3, 0.01).name('Debug Height');
  poseGroup.add(state, 'poseDetectEveryN', 1, 6, 1).name('Detect Every N');

  const poseStatsGroup = renderer.inspector.createParameters('VJ Layer / Pose Stats (ms)');
  poseStatsGroup.add(poseStats, 'backend').name('Backend').listen();
  poseStatsGroup.add(poseStats, 'poseFps').name('Pose FPS').listen();
  poseStatsGroup.add(poseStats, 'detectMs').name('Detect (yolo)').listen();
  poseStatsGroup.add(poseStats, 'preprocessMs').name('Preprocess').listen();
  poseStatsGroup.add(poseStats, 'inferenceMs').name('RTMW Inference').listen();
  poseStatsGroup.add(poseStats, 'decodeMs').name('Decode').listen();
  poseStatsGroup.add(poseStats, 'captureMs').name('Capture (sidecar)').listen();
  poseStatsGroup.add(poseStats, 'previewEncMs').name('Preview enc (native)').listen();
  poseStatsGroup.add(poseStats, 'previewDecMs').name('Preview dec (native)').listen();
  poseStatsGroup.add(poseStats, 'previewKb').name('Preview KB (native)').listen();
  poseStatsGroup.add(poseStats, 'readbackMs').name('Readback (sidecar)').listen();
  poseStatsGroup.add(poseStats, 'serverMs').name('Server total (sidecar)').listen();
  poseStatsGroup.add(poseStats, 'transportMs').name('Transport/wire (sidecar)').listen();
  poseStatsGroup.add(poseStats, 'roundMs').name('Round-trip (sidecar)').listen();
  poseStatsGroup.add(poseStats, 'frames').name('Frames sent/recv').listen();
  poseStatsGroup.add(poseStats, 'overlayMs').name('Overlay Draw').listen();
  poseStatsGroup.add(poseStats, 'retargetMs').name('Retarget/frame').listen();
  poseStatsGroup.add(poseStats, 'clamp').name('Clamp (want→cap)').listen();
  poseStatsGroup.add(poseStats, 'facing').name('Facing θ→applied').listen();
  poseGroup.add(state, 'poseRecording').name('Record').listen().onChange((on) => {
    if (on) {
      poseRecorder.start({
        createdAt: new Date().toISOString(),
        inputWidth: poseProvider?.video?.videoWidth || 0,
        inputHeight: poseProvider?.video?.videoHeight || 0
      });
      statusEl.textContent = 'Recording…';
    } else {
      poseRecorder.stop();
      statusEl.textContent = `Recorded ${poseRecorder.length} frames.`;
    }
  });
  poseGroup.add(state, 'poseReplaying').name('Replay').listen().onChange((on) => {
    if (on) startReplay();
    else stopReplay();
  });
  poseGroup.add(state, 'savePoseRecording').name('Save Recording');
  poseGroup.add(state, 'loadPoseRecording').name('Load Recording');
  poseGroup.add(state, 'runEpProbe').name('Probe EPs (console)');

  // T26: face expression scalars (read-only live) + neutral recalibrate. Phase 1 —
  // extraction only; the mask/deform (T27/T28) consume these later.
  const faceGroup = renderer.inspector.createParameters('VJ Layer / Face');
  faceGroup.add(faceStats, 'neutral').name('Neutral baseline').listen();
  faceGroup.add(faceStats, 'jawOpen').name('Jaw Open').listen();
  faceGroup.add(faceStats, 'smile').name('Smile').listen();
  faceGroup.add(faceStats, 'pucker').name('Pucker').listen();
  faceGroup.add(faceStats, 'blinkL').name('Blink L').listen();
  faceGroup.add(faceStats, 'blinkR').name('Blink R').listen();
  faceGroup.add(faceStats, 'browL').name('Brow L').listen();
  faceGroup.add(faceStats, 'browR').name('Brow R').listen();
  faceGroup.add({ recalibrate: () => faceExpr.recalibrateNeutral() }, 'recalibrate').name('Recalibrate Neutral (rest face)');
  // T28: face deform toggle + amount tuning (live uniforms, no rebuild).
  faceGroup.add(state, 'faceDrive').name('Face Drive (deform)');
  faceGroup.add(state, 'faceJawAngle', 0, 1.2, 0.01).name('Jaw Angle (rad)');
  faceGroup.add(state, 'faceLipDrop', 0, 0.2, 0.005).name('Lip Drop');
  faceGroup.add(state, 'faceCornerMove', 0, 0.2, 0.005).name('Mouth Corner');
  faceGroup.add(state, 'faceLidDrop', 0, 0.2, 0.005).name('Lid Close');
  faceGroup.add(state, 'faceBrowRaise', 0, 0.2, 0.005).name('Brow Raise');
  // T27: region-mask debug + persistence.
  faceGroup.add(state, 'faceMaskDebug').name('Mask Debug (verts)').onChange(refreshFaceMaskOverlays);
  faceGroup.add(state, 'faceMaskRegion', FACE_REGIONS).name('Mask Region').onChange(refreshFaceMaskOverlays);
  faceGroup.add({ download: downloadFaceMasks }, 'download').name('Download Face Masks (.bin)');
  // T30: the full mask editor is the React overlay — this just opens it.
  faceGroup.add({ open: () => editorActions.setOpen && editorActions.setOpen(true) }, 'open').name('Open Face Editor ▸');

  const statsGroup = renderer.inspector.createParameters('VJ Layer / Render Stats');
  statsGroup.add(statsState, 'fps').name('FPS').listen();
  statsGroup.add(statsState, 'calls').name('Draw Calls').listen();
  statsGroup.add(statsState, 'triangles').name('Triangles').listen();
  statsGroup.add(statsState, 'lines').name('Lines').listen();
  statsGroup.add(statsState, 'points').name('Points').listen();
  statsGroup.add(statsState, 'geometries').name('Geometries').listen();
  statsGroup.add(statsState, 'textures').name('Textures').listen();

  // Collapse the bulky setup groups by default — focus lands on Pose.
  performanceGroup.close(); // Scene
  lightingGroup.close();
  meshGroup.close();
}

function applyGlowShadows() {
  // Point-light shadows are 6-face cubemaps — very expensive, especially with a
  // large crowd. Off by default; opt-in via the inspector toggle.
  for (const light of [floorGlow, backGlow, sideGlow]) {
    light.castShadow = state.glowShadows;
    light.shadow.mapSize.set(1024, 1024);
    light.shadow.bias = -0.004;
    light.shadow.normalBias = 0.04;
    light.shadow.camera.near = 0.4;
  }
}

function applyQualityPreset(preset) {
  if (preset === 'cinematic') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    key.shadow.mapSize.set(2048, 2048);
    heroSpot.shadow.mapSize.set(2048, 2048);
    aoPass.samples.value = 24;
    aoPass.resolutionScale = 0.75;
    aoPass.scale.value = state.ao = 1.45;
    bloomPass.strength.value = state.bloom = 0.65;
    crossFill.intensity = state.crossFill = 3.2;
    cameraFill.intensity = state.cameraFill = 14;
    state.performerKey = 4.2;
    state.performerFill = 1.2;
    state.performerRim = 2.2;
  } else if (preset === 'balanced') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    key.shadow.mapSize.set(2048, 2048);
    heroSpot.shadow.mapSize.set(1024, 1024);
    aoPass.samples.value = 12;
    aoPass.resolutionScale = 0.6;
    aoPass.scale.value = state.ao = 1.05;
    bloomPass.strength.value = state.bloom = 0.55;
    crossFill.intensity = state.crossFill = 2.7;
    cameraFill.intensity = state.cameraFill = 12;
    state.performerKey = 3.6;
    state.performerFill = 0.45;
    state.performerRim = 0.8;
  } else {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    key.shadow.mapSize.set(2048, 2048);
    heroSpot.shadow.mapSize.set(1024, 1024);
    aoPass.samples.value = 8;
    aoPass.resolutionScale = 0.5;
    aoPass.scale.value = state.ao = 0.85;
    bloomPass.strength.value = state.bloom = 0.5;
    crossFill.intensity = state.crossFill = 2.4;
    cameraFill.intensity = state.cameraFill = 10;
    state.performerKey = 3.2;
    state.performerFill = 0;
    state.performerRim = 0;
  }

  state.quality = preset;
  key.shadow.camera.updateProjectionMatrix();
  heroSpot.shadow.camera.updateProjectionMatrix();

  for (const walker of walkers) applyPerformerLighting(walker);
  resize();
}

function animationNames() {
  const selectedMeshes = activeMeshes();
  const names = new Set();
  for (const mesh of selectedMeshes) {
    for (const animation of mesh.animations) names.add(animation);
  }

  const preferred = [...names].sort((a, b) => {
    if (a === 'Walking') return -1;
    if (b === 'Walking') return 1;
    if (a === 'Running') return -1;
    if (b === 'Running') return 1;
    return a.localeCompare(b);
  });
  // idle is always available (no clip needed) — base pose for pose mode.
  return [ANIM_IDLE, ...preferred];
}

async function rebuildWalkers() {
  const id = ++rebuildId;
  statusEl.textContent = 'Loading meshes...';

  clearWalkers();

  const meshes = activeMeshes();
  if (meshes.length === 0) {
    statusEl.textContent = 'No performers selected.';
    return;
  }

  for (let i = 0; i < state.count; i += 1) {
    const mesh = meshes[i % meshes.length];
    const clipAsset = selectAnimation(mesh, state.animationName);
    walkers.push(createWalker(mesh, clipAsset, i));
  }

  const batchKeys = [...new Set(walkers.map((walker) => walker.batchKey))];
  const batches = await Promise.all(batchKeys.map((batchKey) => {
    const walker = walkers.find((candidate) => candidate.batchKey === batchKey);
    return loadCrowdBatch(walker.mesh, walker.animationAsset, walkers.filter((candidate) => candidate.batchKey === batchKey));
  }));
  if (id !== rebuildId) return;

  for (const batch of batches) {
    crowdBatches.push(batch);
    scene.add(batch.source);
  }

  arrangeWalkers();
  statusEl.textContent = `${walkers.length} performers / ${state.animationName}`;
}

function activeMeshes() {
  return manifest.meshes.filter((mesh) => state.selectedMeshIds.includes(mesh.id));
}

function selectAnimation(mesh, name) {
  // Each character is now a single GLB carrying every kept clip; the animation
  // is selected by clip name at load time, not by a separate URL.
  if (name === ANIM_IDLE) return { name: ANIM_IDLE, url: mesh.url, idle: true };
  if (mesh.animations.includes(name)) return { name, url: mesh.url };

  throw new Error(`${mesh.name} does not include ${name}. Select meshes with a shared animation.`);
}

function createWalker(mesh, animationAsset, index) {
  return {
    index,
    mesh,
    animationAsset,
    batchKey: `${mesh.id}:${animationAsset.name}`,
    boneSource: boneSourceForIndex(index, state.posePerformerCount),
    basePosition: new THREE.Vector3(),
    position: new THREE.Vector3(),
    rotationY: 0,
    seed: hashRandom(index + 1),
    // Per-instance random factors — proof of true individual instancing: same
    // skin buffer, independent animation phase, size, and proportions.
    sizeRand: hashRandom(index * 7 + 3),
    propRand: new THREE.Vector3(hashRandom(index * 11 + 1), hashRandom(index * 13 + 5), hashRandom(index * 17 + 9)),
    duration: 1,
    laneOffset: 0,
    batch: null,
    batchIndex: -1,
    lightRig: index < Math.min(state.performerLightCount, MAX_PERFORMER_RIGS) ? createPerformerLightRig(index) : null
  };
}

async function loadCrowdBatch(mesh, animationAsset, batchWalkers) {
  const gltf = await loadGltf(animationAsset.url);
  const isIdle = animationAsset.idle === true;
  const clip = isIdle
    ? null
    : (gltf.animations.find((candidate) => candidate.name === animationAsset.name) ?? gltf.animations[0]);

  if (!isIdle && !clip) {
    throw new Error(`${mesh.name} / ${animationAsset.name} has no animation clip.`);
  }

  const source = SkeletonUtils.clone(gltf.scene);
  source.name = `crowd-source-${mesh.id}`;
  const skinnedMeshes = [];

  source.traverse((child) => {
    if (child.isSkinnedMesh) skinnedMeshes.push(child);
    if (!child.isMesh && !child.isSkinnedMesh) return;
    child.frustumCulled = false;
    child.castShadow = true;
    child.receiveShadow = true;
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      child.material = materials.map((material) => preparePerformerMaterial(material));
      if (child.material.length === 1) child.material = child.material[0];
    }
  });

  if (skinnedMeshes.length === 0) {
    throw new Error(`${mesh.name} / ${animationAsset.name} has no SkinnedMesh to instance.`);
  }

  normalizeModel(source);
  applyEmissiveBoost(source);

  // idle: no mixer/action — the rig stays at its bind pose (won't fight pose).
  const mixer = isIdle ? null : new THREE.AnimationMixer(source);
  const action = isIdle ? null : mixer.clipAction(clip);
  if (action) action.play();

  const batch = {
    mesh,
    animationAsset,
    source,
    sourceScale: 1,
    sourceOffset: new THREE.Vector3(),
    walkers: batchWalkers,
    mixer,
    action,
    idle: isIdle,
    duration: clip ? clip.duration : 1,
    instanceMatrices: new StorageBufferAttribute(batchWalkers.length, 16),
    instanceMatricesNode: null,
    computeNodes: [],
    skeletonStates: []
  };

  batch.instanceMatricesNode = storage(batch.instanceMatrices, 'mat4', batchWalkers.length).toReadOnly();

  for (let i = 0; i < batchWalkers.length; i += 1) {
    const walker = batchWalkers[i];
    walker.batch = batch;
    walker.batchIndex = i;
    walker.duration = batch.duration;
  }

  source.updateMatrixWorld(true);
  // T27/T28: build the face mask + its GPU buffers (expr + mask storage, hinge/amount
  // uniforms) BEFORE the compute nodes, so createComputedSkinnedMesh can inject the
  // face deform for the head mesh (reads batch.faceExprNode / batch.faceMask.maskNode).
  await buildFaceMask(batch, mesh, skinnedMeshes);
  for (const skinnedMesh of skinnedMeshes) createComputedSkinnedMesh(skinnedMesh, batch);

  // Rest LOCAL quats in the NORMALIZED source space — bones are still at GLB rest
  // here (no mixer.setTime yet). The retargeter restores from these instead of
  // skeleton.pose() (which would inject the normalization scale → vanish, §B).
  const restSkeleton = batch.skeletonStates[0]?.skeleton;
  batch.restQuats = restSkeleton ? restSkeleton.bones.map((b) => b.quaternion.clone()) : null;

  source.traverse((child) => {
    if ((child.isMesh || child.isSkinnedMesh) && !child.name.endsWith('-gpu-instances')) {
      child.visible = false;
    }
  });
  source.visible = true;
  return batch;
}

function preparePerformerMaterial(material) {
  const clone = material.clone();
  clone.roughness = Math.max(0.32, Math.min(1, clone.roughness ?? 0.58));
  clone.metalness = Math.max(clone.metalness ?? 0, 0.08);
  clone.envMapIntensity = 1.1;
  clone.needsUpdate = true;
  return clone;
}

function createSourceVertexAttribute(geometry) {
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const data = new Float32Array(position.count * 8);

  for (let i = 0; i < position.count; i += 1) {
    const offset = i * 8;
    data[offset + 0] = position.getX(i);
    data[offset + 1] = position.getY(i);
    data[offset + 2] = position.getZ(i);
    data[offset + 3] = 1;
    data[offset + 4] = normal.getX(i);
    data[offset + 5] = normal.getY(i);
    data[offset + 6] = normal.getZ(i);
    data[offset + 7] = 0;
  }

  return new StorageBufferAttribute(data, 4);
}

function createSkinIndexAttribute(attribute) {
  const data = new Uint32Array(attribute.count * 4);

  for (let i = 0; i < attribute.count; i += 1) {
    const offset = i * 4;
    data[offset + 0] = attribute.getX(i);
    data[offset + 1] = attribute.getY(i);
    data[offset + 2] = attribute.getZ(i);
    data[offset + 3] = attribute.getW(i);
  }

  return new StorageBufferAttribute(data, 4);
}

function createSkinWeightAttribute(attribute) {
  const data = new Float32Array(attribute.count * 4);

  for (let i = 0; i < attribute.count; i += 1) {
    const offset = i * 4;
    data[offset + 0] = attribute.getX(i);
    data[offset + 1] = attribute.getY(i);
    data[offset + 2] = attribute.getZ(i);
    data[offset + 3] = attribute.getW(i);
  }

  return new StorageBufferAttribute(data, 4);
}

function getSkeletonState(skinnedMesh, batch) {
  const key = `${batch.mesh.id}:${batch.animationAsset.name}:${skinnedMesh.uuid}`;
  let skeletonState = skeletonStates.get(key);

  if (!skeletonState) {
    const boneCount = skinnedMesh.skeleton.bones.length;
    const boneMatrices = new StorageBufferAttribute(batch.walkers.length * boneCount, 16);
    skeletonState = {
      skeleton: skinnedMesh.skeleton,
      boneCount,
      boneMatrices,
      boneMatricesNode: storage(boneMatrices, 'mat4', boneMatrices.count).toReadOnly()
    };
    skeletonStates.set(key, skeletonState);
  }

  batch.skeletonStates.push(skeletonState);
  return skeletonState;
}

function createComputedSkinnedMesh(sourceMesh, batch) {
  // Render geometry must be non-indexed: the vertex stage reads the computed
  // vertices buffer by `instanceIndex * vertexCount + vertexIndex`, which only
  // lines up with the per-vertex compute output when vertexIndex is sequential.
  // An indexed draw scrambles those reads and the mesh collapses into spikes.
  const geometry = sourceMesh.geometry.clone();
  const position = geometry.getAttribute('position');
  const skinIndex = geometry.getAttribute('skinIndex');
  const skinWeight = geometry.getAttribute('skinWeight');

  if (!position || !skinIndex || !skinWeight) {
    throw new Error(`${batch.mesh.name} has a SkinnedMesh without position, skinIndex, or skinWeight attributes.`);
  }

  const material = Array.isArray(sourceMesh.material)
    ? sourceMesh.material.map((entry) => entry.clone())
    : sourceMesh.material.clone();
  const vertexCount = position.count;
  const skeletonState = getSkeletonState(sourceMesh, batch);
  const sourceVertices = storage(createSourceVertexAttribute(geometry), 'vec4', vertexCount * 2).toReadOnly();
  const skinIndices = storage(createSkinIndexAttribute(skinIndex), 'uvec4', vertexCount).toReadOnly();
  const skinWeights = storage(createSkinWeightAttribute(skinWeight), 'vec4', vertexCount).toReadOnly();
  const bindMatrix = uniform(sourceMesh.bindMatrix, 'mat4');
  const bindMatrixInverse = uniform(sourceMesh.bindMatrixInverse, 'mat4');
  const sourceMatrix = uniform(sourceMesh.matrixWorld, 'mat4');
  // Fail loud, not freeze: the skinned-vertex storage buffer must fit the device's
  // storage-binding cap. Past it the device silently rejects the binding and the
  // render loop hangs (the old ~418 freeze). Throw a clear, actionable error instead.
  const vertBytes = batch.walkers.length * vertexCount * 2 * 16; // 2 vec4 × 16B
  const bindCap = device.limits.maxStorageBufferBindingSize;
  if (vertBytes > bindCap) {
    throw new Error(
      `[crowd] ${batch.mesh.name}: skinned-vertex buffer ${(vertBytes / 2 ** 20).toFixed(0)}MiB ` +
      `exceeds device storage-binding cap ${(bindCap / 2 ** 20).toFixed(0)}MiB at count ` +
      `${batch.walkers.length} (${vertexCount} verts/mesh). Lower count or split the mesh into sub-batches.`
    );
  }
  const vertices = attributeArray(batch.walkers.length * vertexCount * 2, 'vec4');

  const computeNode = Fn(() => {
    const sourceVertex = instanceIndex.mod(uint(vertexCount));
    const meshInstance = instanceIndex.div(uint(vertexCount));
    const sourceOffset = sourceVertex.mul(uint(2));
    const targetOffset = instanceIndex.mul(uint(2));
    const boneOffset = meshInstance.mul(uint(skeletonState.boneCount));
    const indices = skinIndices.element(sourceVertex);
    const weights = skinWeights.element(sourceVertex);
    // T28: displace the bind-space head vertex from the face expression before
    // skinning (only the head mesh carries a mask). Non-head meshes skin unchanged.
    let srcPos = sourceVertices.element(sourceOffset).xyz;
    if (batch.faceMask && sourceMesh === batch.faceMask.headMesh) {
      srcPos = faceDeformNode(srcPos, sourceVertex, meshInstance, batch.faceMask, batch);
    }
    const skinVertex = bindMatrix.mul(vec4(srcPos, 1));
    const boneMatX = skeletonState.boneMatricesNode.element(boneOffset.add(indices.x));
    const boneMatY = skeletonState.boneMatricesNode.element(boneOffset.add(indices.y));
    const boneMatZ = skeletonState.boneMatricesNode.element(boneOffset.add(indices.z));
    const boneMatW = skeletonState.boneMatricesNode.element(boneOffset.add(indices.w));
    const skinMatrix = add(
      weights.x.mul(boneMatX),
      weights.y.mul(boneMatY),
      weights.z.mul(boneMatZ),
      weights.w.mul(boneMatW)
    );
    const skinPosition = bindMatrixInverse.mul(add(
      boneMatX.mul(weights.x).mul(skinVertex),
      boneMatY.mul(weights.y).mul(skinVertex),
      boneMatZ.mul(weights.z).mul(skinVertex),
      boneMatW.mul(weights.w).mul(skinVertex)
    )).xyz;
    const skinNormal = bindMatrixInverse.mul(skinMatrix).mul(bindMatrix).transformDirection(sourceVertices.element(sourceOffset.add(uint(1))).xyz).xyz;
    const instanceMatrix = batch.instanceMatricesNode.element(meshInstance);
    const meshPosition = sourceMatrix.mul(vec4(skinPosition, 1)).xyz;
    const meshNormal = transformNormal(skinNormal, sourceMatrix);

    vertices.element(targetOffset).assign(vec4(instanceMatrix.mul(vec4(meshPosition, 1)).xyz, 1));
    vertices.element(targetOffset.add(uint(1))).assign(vec4(transformNormal(meshNormal, instanceMatrix), 0));
  })().compute(batch.walkers.length * vertexCount).setName(`Compute ${batch.mesh.name}`);

  const meshVertex = instanceIndex.mul(uint(vertexCount)).add(vertexIndex).mul(uint(2));
  const applyMaterialNodes = (entry) => {
    entry.positionNode = vertices.element(meshVertex).xyz;
    entry.normalNode = transformNormalToView(vertices.element(meshVertex.add(uint(1))).xyz).toVarying();
    entry.needsUpdate = true;
  };

  if (Array.isArray(material)) {
    for (const entry of material) applyMaterialNodes(entry);
  } else {
    applyMaterialNodes(material);
  }

  // Must be an InstancedMesh, not a Mesh with `.count`. The renderer only gives
  // an object its own pipeline bindings when `isInstancedMesh || count > 1`
  // (RenderObject cache key). A plain Mesh with count === 1 shares bindings with
  // every other single-instance batch, so they all collapse onto one buffer and
  // render the same bind-pose at the origin. InstancedMesh always binds per-object
  // and keeps `instanceIndex` active even for a single instance.
  const computedMesh = new THREE.InstancedMesh(geometry, material, batch.walkers.length);
  computedMesh.name = `${sourceMesh.name || 'skinned'}-gpu-instances`;
  // Per-instance world transforms are baked into the computed vertices buffer,
  // so the native instanceMatrix stays identity (otherwise it double-transforms).
  const identity = new THREE.Matrix4();
  for (let i = 0; i < batch.walkers.length; i += 1) computedMesh.setMatrixAt(i, identity);
  computedMesh.instanceMatrix.needsUpdate = true;
  computedMesh.castShadow = true;
  computedMesh.receiveShadow = true;
  computedMesh.frustumCulled = false;
  batch.source.add(computedMesh);
  batch.computeNodes.push(computeNode);
}

// --- T27: face-region mask (build / overlay / persist) ------------------------
const _faceMaskOverlays = new Set(); // debug vertex clouds, recolored on region change
const _faceMaskV = new THREE.Vector3();

// Flatten a possibly-interleaved attribute into a plain typed array [i*n+c] — the
// pure generator (face-mask.js) assumes that contiguous layout.
function flatAttr(attr, n, Ctor) {
  const out = new Ctor(attr.count * n);
  for (let i = 0; i < attr.count; i += 1) for (let c = 0; c < n; c += 1) out[i * n + c] = attr.getComponent(i, c);
  return out;
}

// Load-or-generate the region mask for a batch's head mesh, then stand up the GPU
// buffers the deform reads: a per-instance expression buffer + a per-vertex mask
// buffer + hinge/amount uniforms. Sidecar `.bin` (T29 painter output) wins when
// present + valid; else auto-gen from geometry (V27). No head bone / no head verts →
// face-drive off for this model (⊥ crash the crowd, V28).
async function buildFaceMask(batch, mesh, skinnedMeshes) {
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

  // Per-instance expression buffer: 2 vec4/instance = [jawOpen,smile,pucker,blinkL |
  // blinkR,browL,browR,_]. Updated CPU-side each frame (writeFaceExpr), read in compute.
  const nWalkers = batch.walkers.length;
  batch.faceExpr = new StorageBufferAttribute(nWalkers * 2, 4);
  batch.faceExprNode = storage(batch.faceExpr, 'vec4', nWalkers * 2).toReadOnly();

  // Per-vertex mask buffer: 2 vec4/vertex = [jaw,lowerLip,mouthCorner,upperLidL |
  // upperLidR,browL,browR,_], normalized 0..1. Filled from mask.data; re-uploaded live
  // when the editor (T29) repaints/re-seeds.
  const maskAttr = new StorageBufferAttribute(vertexCount * 2, 4);
  uploadMaskBuffer(mask, maskAttr);
  const maskNode = storage(maskAttr, 'vec4', vertexCount * 2).toReadOnly();

  // Hinge + amount uniforms (amounts driven live from state each frame; 0 when
  // faceDrive is off → no deform without a rebuild).
  const axis = new THREE.Vector3(mask.hinge.axis[0], mask.hinge.axis[1], mask.hinge.axis[2]);
  if (axis.lengthSq() < 1e-8) axis.set(1, 0, 0); else axis.normalize();
  batch.faceUniforms = {
    hingeOrigin: uniform(new THREE.Vector3(mask.hinge.origin[0], mask.hinge.origin[1], mask.hinge.origin[2])),
    hingeAxis: uniform(axis),
    headHeight: uniform(mask.headHeight || 1),
    jawAngle: uniform(0),
    lipDrop: uniform(0),
    cornerMove: uniform(0),
    lidDrop: uniform(0),
    browRaise: uniform(0)
  };

  batch.faceMask = { mask, headMesh, headBoneIndex, vertexCount, source, maskNode, maskAttr, geom, forwardSign: 1 };
  console.log(`[face] ${mesh.name}: mask ${source}, ${vertexCount} verts, head bone ${headBoneIndex}`);
  buildFaceMaskOverlay(batch);
}

// Fill the GPU mask buffer from mask.data (region-major uint8 → per-vertex 2×vec4).
function uploadMaskBuffer(mask, maskAttr) {
  const arr = maskAttr.array;
  const D = mask.data; const N = mask.vertexCount;
  for (let i = 0; i < N; i += 1) {
    const o = i * 8;
    arr[o] = D[i] / 255; arr[o + 1] = D[N + i] / 255; arr[o + 2] = D[2 * N + i] / 255; arr[o + 3] = D[3 * N + i] / 255;
    arr[o + 4] = D[4 * N + i] / 255; arr[o + 5] = D[5 * N + i] / 255; arr[o + 6] = D[6 * N + i] / 255; arr[o + 7] = 0;
  }
  maskAttr.needsUpdate = true;
}

// Face deform (T28 / V29): displace the BIND-space head vertex before skinning, so the
// head bone carries the motion. Jaw = hinge-rotate (Rodrigues about the ear-line axis);
// lips/lids/brows = translate along mesh-local up. Normals unchanged (V29 note).
function faceDeformNode(pos, sourceVertex, meshInstance, fm, batch) {
  const u = batch.faceUniforms;
  const mi2 = meshInstance.mul(uint(2));
  const sv2 = sourceVertex.mul(uint(2));
  const maskA = fm.maskNode.element(sv2);            // jaw, lowerLip, mouthCorner, upperLidL
  const maskB = fm.maskNode.element(sv2.add(uint(1))); // upperLidR, browL, browR, _
  const exprA = batch.faceExprNode.element(mi2);       // jawOpen, smile, pucker, blinkL
  const exprB = batch.faceExprNode.element(mi2.add(uint(1))); // blinkR, browL, browR, _

  // Jaw hinge — Rodrigues rotate (pos-origin) about the unit axis by θ, add the delta.
  const theta = u.jawAngle.mul(exprA.x).mul(maskA.x);
  const v = pos.sub(u.hingeOrigin);
  const k = u.hingeAxis;
  const c = cos(theta); const s = sin(theta);
  const vrot = v.mul(c).add(k.cross(v).mul(s)).add(k.mul(k.dot(v)).mul(c.oneMinus()));
  let p = pos.add(vrot.sub(v));

  const hh = u.headHeight;
  // lower lip follows jaw open, drops in mesh-local −y.
  p = p.add(vec3(0, u.lipDrop.mul(hh).mul(exprA.x).mul(maskA.y).negate(), 0));
  // mouth corners: smile spreads outward (±x), pucker narrows (net = smile−pucker).
  const dir = sign(pos.x.sub(u.hingeOrigin.x));
  const corner = u.cornerMove.mul(hh).mul(exprA.y.sub(exprA.z)).mul(maskA.z);
  p = p.add(vec3(dir.mul(corner), 0, 0));
  // upper lids close downward (−y): L=(blinkL,maskA.w), R=(blinkR,maskB.x).
  const lid = u.lidDrop.mul(hh);
  p = p.add(vec3(0, lid.mul(exprA.w).mul(maskA.w).add(lid.mul(exprB.x).mul(maskB.x)).negate(), 0));
  // brows raise (+y): L=(browL,maskB.y), R=(browR,maskB.z).
  const brow = u.browRaise.mul(hh);
  p = p.add(vec3(0, brow.mul(exprB.y).mul(maskB.y).add(brow.mul(exprB.z).mul(maskB.z)), 0));
  return p;
}

// Debug cloud: head verts at bind-pose world position, colored by the selected
// region's weight (dark → hot). Standalone Points → cannot disturb the render path.
function buildFaceMaskOverlay(batch) {
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

function recolorFaceMaskOverlay(pts) {
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

function refreshFaceMaskOverlays() {
  for (const pts of _faceMaskOverlays) { pts.visible = state.faceMaskDebug; if (state.faceMaskDebug) recolorFaceMaskOverlay(pts); }
}

// Persist the current masks so painter edits / a chosen auto-gen survive. Browser
// can't write public/ → download; drop the files into public/models/ and commit.
function downloadFaceMasks() {
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

// --- T29: face-mask EDITOR — paint region weights on a dedicated edit-head -----------
// The crowd is GPU-instanced (no per-instance geometry to click), so the editor spawns
// ONE static head (the model's own head geometry) beside the crowd as a paint target.
// Raycast it → brush weights into the region → live-update the GPU mask (crowd deforms
// immediately) → download the .bin. Also re-seeds / flips the auto-gen forward axis and
// re-places the jaw hinge — the fixes for the wrong-region failures on stylized heads.
const faceEditor = {
  active: false, batchIndex: 0, region: 'jaw', radius: 0.04, strength: 0.5,
  erase: false, symmetric: true, hingeMode: false, editHead: null, batch: null, painting: false
};
const REGION_MIRROR = { upperLidL: 'upperLidR', upperLidR: 'upperLidL', browL: 'browR', browR: 'browL' };
const _rayNDC = new THREE.Vector2();
const _raycaster = new THREE.Raycaster();
const _vLocal = new THREE.Vector3();
const _vTmp = new THREE.Vector3();

function editBatch() {
  const list = crowdBatches.filter((b) => b.faceMask);
  return list[Math.min(faceEditor.batchIndex, list.length - 1)] || null;
}

function setFaceEditMode(on) {
  faceEditor.active = on;
  if (faceEditor.editHead) {
    scene.remove(faceEditor.editHead);
    faceEditor.editHead.geometry.dispose(); faceEditor.editHead.material.dispose();
    faceEditor.editHead = null;
  }
  faceEditor.batch = null;
  if (!on) return;
  const batch = editBatch();
  if (!batch) { statusEl.textContent = 'Face editor: no model with a face mask.'; faceEditor.active = false; return; }
  faceEditor.batch = batch;
  buildEditHead(batch);
  frameEditHead();
  statusEl.textContent = `Face editor: ${batch.mesh.name} — click the head to paint "${faceEditor.region}" (Erase/Symmetric in panel).`;
}

// Static clone of the head geometry, placed to the left of the crowd, scaled to a
// comfortable size, vertex-colored by the current region weight.
function buildEditHead(batch) {
  const src = batch.faceMask.geom;
  const g = new THREE.BufferGeometry();
  const posAttr = src.getAttribute('position').clone();
  g.setAttribute('position', posAttr);
  faceEditor.bindPositions = Float32Array.from(posAttr.array); // bind copy for CPU preview deform
  if (src.index) g.setIndex(src.index.clone());
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(batch.faceMask.vertexCount * 3), 3));
  g.computeBoundingBox();
  const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
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

// Edit head color: grey base, current region weight → hot red/yellow.
function recolorEditHead() {
  const mesh = faceEditor.editHead; const batch = faceEditor.batch;
  if (!mesh || !batch) return;
  const N = batch.faceMask.vertexCount;
  const r = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const D = batch.faceMask.mask.data;
  const col = mesh.geometry.getAttribute('color');
  for (let i = 0; i < N; i += 1) {
    const w = D[r * N + i] / 255;
    col.setXYZ(i, 0.12 + 0.88 * w, 0.12 * (1 - w) + 0.5 * w * w, 0.12 * (1 - w));
  }
  col.needsUpdate = true;
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
function faceEditPaint(worldPoint, erase) {
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

function applyHingeUniforms(batch) {
  const u = batch.faceUniforms; const h = batch.faceMask.mask.hinge;
  u.hingeOrigin.value.set(h.origin[0], h.origin[1], h.origin[2]);
  const ax = _vTmp.set(h.axis[0], h.axis[1], h.axis[2]);
  if (ax.lengthSq() < 1e-8) ax.set(1, 0, 0); else ax.normalize();
  u.hingeAxis.value.copy(ax);
  u.headHeight.value = batch.faceMask.mask.headHeight || 1;
}

// Re-run auto-gen (optionally flipping the assumed +z forward), re-upload, recolor.
function faceEditReseed(flipForward) {
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
  applyHingeUniforms(batch);
  recolorEditHead();
  if (batch.faceMaskOverlay) recolorFaceMaskOverlay(batch.faceMaskOverlay);
  statusEl.textContent = `Re-seeded ${batch.mesh.name} (forward ${fm.forwardSign > 0 ? '+z' : '-z'}).`;
}

function faceEditClearRegion() {
  const batch = faceEditor.batch; if (!batch) return;
  const region = Math.max(0, FACE_REGIONS.indexOf(faceEditor.region));
  const N = batch.faceMask.vertexCount;
  for (let i = 0; i < N; i += 1) setMaskWeight(batch, region, i, 0);
  batch.faceMask.maskAttr.needsUpdate = true;
  recolorEditHead();
}

function faceEditRaycast(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  _rayNDC.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  _raycaster.setFromCamera(_rayNDC, camera);
  return _raycaster.intersectObject(faceEditor.editHead, false)[0] || null;
}

function onFaceEditDown(e) {
  if (!faceEditor.active || !faceEditor.editHead) return;
  const hit = faceEditRaycast(e);
  if (!hit) return; // missed head → let OrbitControls handle it (orbit empty space)
  controls.enabled = false; // paint, don't orbit
  if (faceEditor.hingeMode) {
    faceEditor.editHead.worldToLocal(_vLocal.copy(hit.point));
    faceEditor.batch.faceMask.mask.hinge.origin = [_vLocal.x, _vLocal.y, _vLocal.z];
    applyHingeUniforms(faceEditor.batch);
    statusEl.textContent = 'Jaw hinge set.';
    return;
  }
  faceEditor.painting = true;
  faceEditPaint(hit.point, e.button === 2 || faceEditor.erase);
}

function onFaceEditMove(e) {
  if (!faceEditor.painting) return;
  const hit = faceEditRaycast(e);
  if (hit) faceEditPaint(hit.point, faceEditor.erase || (e.buttons === 2));
}

function onFaceEditUp() {
  if (faceEditor.painting || !controls.enabled) { faceEditor.painting = false; controls.enabled = true; }
}

// CPU preview deform for the edit-head (T30 test-drive): apply the SAME jaw-hinge +
// translates as the GPU path (faceDeformNode) to the static edit-head so the sliders
// show the effect WYSIWYG while painting. Cheap (one mesh, recompute on change).
const ZERO_EXPR = { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0 };
const _defAxis = new THREE.Vector3();
function deformEditHead() {
  const mesh = faceEditor.editHead; const batch = faceEditor.batch;
  if (!mesh || !faceEditor.bindPositions || !batch) return;
  const fm = batch.faceMask; const N = fm.vertexCount; const D = fm.mask.data;
  const e = debugFaceOverride || ZERO_EXPR;
  const bind = faceEditor.bindPositions;
  const pos = mesh.geometry.getAttribute('position');
  const h = fm.mask.hinge; const hh = fm.mask.headHeight || 1;
  const ox = h.origin[0]; const oy = h.origin[1]; const oz = h.origin[2];
  _defAxis.set(h.axis[0], h.axis[1], h.axis[2]);
  if (_defAxis.lengthSq() < 1e-8) _defAxis.set(1, 0, 0); else _defAxis.normalize();
  const kx = _defAxis.x; const ky = _defAxis.y; const kz = _defAxis.z;
  const jawA = state.faceJawAngle; const lip = state.faceLipDrop * hh; const corner = state.faceCornerMove * hh;
  const lid = state.faceLidDrop * hh; const brow = state.faceBrowRaise * hh;
  for (let i = 0; i < N; i += 1) {
    let x = bind[i * 3]; let y = bind[i * 3 + 1]; let z = bind[i * 3 + 2];
    const theta = jawA * e.jawOpen * (D[i] / 255);
    if (theta > 1e-5) { // Rodrigues rotate about the hinge axis through the origin
      const vx = x - ox; const vy = y - oy; const vz = z - oz;
      const c = Math.cos(theta); const sN = Math.sin(theta); const oneC = 1 - c;
      const dot = kx * vx + ky * vy + kz * vz;
      const crx = ky * vz - kz * vy; const cry = kz * vx - kx * vz; const crz = kx * vy - ky * vx;
      x = ox + vx * c + crx * sN + kx * dot * oneC;
      y = oy + vy * c + cry * sN + ky * dot * oneC;
      z = oz + vz * c + crz * sN + kz * dot * oneC;
    }
    y -= lip * e.jawOpen * (D[N + i] / 255);
    x += Math.sign(bind[i * 3] - ox) * corner * (e.smile - e.pucker) * (D[2 * N + i] / 255);
    y -= lid * (e.blinkL * (D[3 * N + i] / 255) + e.blinkR * (D[4 * N + i] / 255));
    y += brow * (e.browL * (D[5 * N + i] / 255) + e.browR * (D[6 * N + i] / 255));
    pos.setXYZ(i, x, y, z);
  }
  pos.needsUpdate = true;
}

// Push the current model list + per-region painted-weight sums to the store (badges).
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
    applyHingeUniforms(batch);
    recolorEditHead();
    deformEditHead();
    setEditorState({ status: 'Loaded .bin' });
  } catch (err) {
    setEditorState({ status: `Load failed: ${err.message}` });
  }
}

// Register the editor actions the React UI calls (I.editorUI). main.js owns three;
// React only reads the store + invokes these.
function registerEditorActions() {
  registerActions({
    setOpen(v) { faceEditor.active = v; setFaceEditMode(v); if (v) syncEditorStore(); setEditorState({ open: v }); },
    setModel(i) { faceEditor.batchIndex = i; setFaceEditMode(true); deformEditHead(); setEditorState({ modelIndex: i }); },
    setRegion(r) { faceEditor.region = r; recolorEditHead(); setEditorState({ region: r }); },
    setMode(m) { faceEditor.erase = m === 'erase'; faceEditor.hingeMode = m === 'hinge'; setEditorState({ mode: m }); },
    setRadius(v) { faceEditor.radius = v; setEditorState({ radius: v }); },
    setStrength(v) { faceEditor.strength = v; setEditorState({ strength: v }); },
    setSymmetric(v) { faceEditor.symmetric = v; setEditorState({ symmetric: v }); },
    reseed(flip) { faceEditReseed(flip); recolorEditHead(); deformEditHead(); },
    clearRegion() { faceEditClearRegion(); deformEditHead(); },
    save() { downloadFaceMasks(); },
    loadMask(buf) { faceEditLoadMask(buf); },
    setExpr(expr) { debugFaceOverride = expr; deformEditHead(); setEditorState({ expr }); },
    resetExpr() { debugFaceOverride = null; deformEditHead(); setEditorState({ expr: { ...ZERO_EXPR } }); },
    frameView(v) { faceEditFrameView(v); },
    setOverlay(k, v) {
      const o = { ...editorState.overlays, [k]: v };
      if (k === 'wireframe' && faceEditor.editHead) faceEditor.editHead.material.wireframe = v;
      if (k === 'maskCloud') { state.faceMaskDebug = v; refreshFaceMaskOverlays(); }
      if (k === 'crowd') for (const b of crowdBatches) b.source.traverse((c) => { if (c.name?.endsWith('-gpu-instances')) c.visible = v; });
      setEditorState({ overlays: o });
    }
  });
}

function createPerformerLightRig(index) {
  const target = new THREE.Object3D();
  target.name = `performer-${index}-light-target`;
  scene.add(target);

  const keyLight = new THREE.SpotLight(0xf3fbff, state.performerKey, 7.5, Math.PI * 0.2, 0.68, 1.2);
  keyLight.name = `performer-${index}-key`;
  keyLight.target = target;
  keyLight.castShadow = false;
  scene.add(keyLight);

  const fillLight = new THREE.PointLight(0x8eefff, state.performerFill, 5.5, 1.7);
  fillLight.name = `performer-${index}-fill`;
  scene.add(fillLight);

  const rimLight = new THREE.PointLight(index % 2 === 0 ? 0xff3f7f : 0x51f7ff, state.performerRim, 5.8, 1.6);
  rimLight.name = `performer-${index}-rim`;
  scene.add(rimLight);

  return { target, keyLight, fillLight, rimLight };
}

function applyPerformerLighting(walker) {
  if (!walker.lightRig) return;

  walker.lightRig.keyLight.intensity = state.performerKey;
  walker.lightRig.fillLight.intensity = state.performerFill;
  walker.lightRig.rimLight.intensity = state.performerRim;
  walker.lightRig.keyLight.visible = state.performerKey > 0;
  walker.lightRig.fillLight.visible = state.performerFill > 0;
  walker.lightRig.rimLight.visible = state.performerRim > 0;
}

function updatePerformerLightRig(walker, elapsed) {
  if (!walker.lightRig) return;

  performerWorldPosition.copy(walker.position);

  cameraDirection.subVectors(camera.position, performerWorldPosition).normalize();
  cameraSide.crossVectors(cameraDirection, worldUp).normalize();

  const pulse = 0.92 + Math.sin(elapsed * 1.2 + walker.index * 0.71) * 0.08;
  const torsoY = 1.05 * state.scale;
  const headY = 1.65 * state.scale;

  walker.lightRig.target.position.set(
    performerWorldPosition.x,
    performerWorldPosition.y + torsoY,
    performerWorldPosition.z
  );

  walker.lightRig.keyLight.position.set(
    performerWorldPosition.x + cameraDirection.x * 2.6 - cameraSide.x * 1.35,
    performerWorldPosition.y + 2.65 * state.scale,
    performerWorldPosition.z + cameraDirection.z * 2.6 - cameraSide.z * 1.35
  );

  walker.lightRig.fillLight.position.set(
    performerWorldPosition.x + cameraDirection.x * 1.05 + cameraSide.x * 1.05,
    performerWorldPosition.y + headY,
    performerWorldPosition.z + cameraDirection.z * 1.05 + cameraSide.z * 1.05
  );

  walker.lightRig.rimLight.position.set(
    performerWorldPosition.x - cameraDirection.x * 2.15 + cameraSide.x * 1.15,
    performerWorldPosition.y + 1.45 * state.scale,
    performerWorldPosition.z - cameraDirection.z * 2.15 + cameraSide.z * 1.15
  );

  walker.lightRig.keyLight.intensity = state.performerKey * pulse;
  walker.lightRig.fillLight.intensity = state.performerFill;
  walker.lightRig.rimLight.intensity = state.performerRim * pulse;
}

function applyEmissiveBoost(model) {
  model.traverse((child) => {
    if ((!child.isMesh && !child.isSkinnedMesh) || !child.material) return;

    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      if (!('emissive' in material)) continue;

      if (material.emissiveMap) {
        // The models author their glow as an emissive map (neon veins/caps).
        // Keep the map at full color and only scale how hot it reads so bloom
        // and the MRT emissive pass have real signal. Zeroing emissive here was
        // multiplying the vein map by 0 and killing all glow + bloom.
        material.emissive.setRGB(1, 1, 1);
        material.emissiveIntensity = state.emissiveBoost;
      } else if (material.color) {
        // Untextured fallback: derive a glow from a saturated base color.
        const color = material.color;
        const maxChannel = Math.max(color.r, color.g, color.b);
        const saturation = maxChannel - Math.min(color.r, color.g, color.b);
        if (maxChannel > 0.45 && saturation > 0.14) {
          material.emissive.copy(color).multiplyScalar(state.emissiveBoost * 0.55);
          material.emissiveIntensity = state.emissiveBoost;
        } else {
          material.emissive.setRGB(0, 0, 0);
          material.emissiveIntensity = 0;
        }
      }

      material.needsUpdate = true;
    }
  });
}

function normalizeModel(model) {
  model.position.set(0, 0, 0);
  model.rotation.set(0, 0, 0);
  model.scale.setScalar(1);
  model.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const height = Math.max(size.y, 0.001);
  const targetHeight = 1.8;
  const baseScale = targetHeight / height;

  model.userData.floorNormalization = {
    baseScale,
    center: center.clone(),
    minY: box.min.y
  };
}

function applyModelScale(model) {
  const normalization = model.userData.floorNormalization;
  const scale = normalization.baseScale * state.scale;

  model.scale.setScalar(scale);
  model.position.set(
    -normalization.center.x * scale,
    -normalization.minY * scale,
    -normalization.center.z * scale
  );
}

function arrangeWalkers() {
  const count = walkers.length;
  const center = (count - 1) * 0.5;
  const gridColumns = Math.max(1, Math.ceil(Math.sqrt(count)));
  const gridRows = Math.max(1, Math.ceil(count / gridColumns));

  for (const walker of walkers) {
    const i = walker.index;
    const t = count <= 1 ? 0 : i / (count - 1);
    const offset = (i - center) * state.spacing;
    const jitterX = (hashRandom(i * 19 + 7) - 0.5) * state.spacing * state.disorder;
    const jitterZ = (hashRandom(i * 23 + 11) - 0.5) * state.spacing * state.disorder;

    if (state.arrangement === 'arc') {
      const angle = THREE.MathUtils.lerp(-0.72, 0.72, t);
      const radius = Math.max(5, count * state.spacing * 0.34);
      walker.basePosition.x = Math.sin(angle) * radius + jitterX;
      walker.basePosition.z = -Math.cos(angle) * radius + radius - 1.2 + jitterZ;
      walker.rotationY = angle * 0.72;
      walker.laneOffset = walker.basePosition.z;
    } else if (state.arrangement === 'stagger') {
      walker.basePosition.x = offset + jitterX;
      walker.basePosition.z = (i % 2 ? -1 : 1) * state.spacing * 0.42 + jitterZ;
      walker.rotationY = 0;
      walker.laneOffset = walker.basePosition.z;
    } else if (state.arrangement === 'circle') {
      const angle = (i / count) * Math.PI * 2;
      const radius = Math.max(3.5, count * state.spacing * 0.2);
      walker.basePosition.x = Math.cos(angle) * radius + jitterX;
      walker.basePosition.z = Math.sin(angle) * radius + jitterZ;
      walker.rotationY = -angle + Math.PI * 0.5;
      walker.laneOffset = walker.basePosition.z;
    } else if (state.arrangement === 'grid') {
      const col = i % gridColumns;
      const row = Math.floor(i / gridColumns);
      walker.basePosition.x = (col - (gridColumns - 1) * 0.5) * state.spacing + jitterX;
      walker.basePosition.z = (row - (gridRows - 1) * 0.5) * state.spacing + jitterZ;
      walker.rotationY = 0;
      walker.laneOffset = walker.basePosition.z;
    } else if (state.arrangement === 'random') {
      const radius = Math.max(2, Math.sqrt(count) * state.spacing * 0.72);
      const angle = hashRandom(i * 31 + 3) * Math.PI * 2;
      const distance = Math.sqrt(hashRandom(i * 37 + 5)) * radius;
      walker.basePosition.x = Math.cos(angle) * distance;
      walker.basePosition.z = Math.sin(angle) * distance;
      walker.rotationY = hashRandom(i * 41 + 13) * Math.PI * 2;
      walker.laneOffset = walker.basePosition.z;
    } else if (state.arrangement === 'spiral') {
      const angle = i * 2.399963;
      const radius = Math.sqrt(i + 1) * state.spacing * 0.72;
      walker.basePosition.x = Math.cos(angle) * radius + jitterX;
      walker.basePosition.z = Math.sin(angle) * radius + jitterZ;
      walker.rotationY = -angle + Math.PI * 0.5;
      walker.laneOffset = walker.basePosition.z;
    } else {
      walker.basePosition.x = offset + jitterX;
      walker.basePosition.z = jitterZ;
      walker.rotationY = 0;
      walker.laneOffset = 0;
    }

    walker.basePosition.y = 0;
    walker.position.copy(walker.basePosition);
  }

  updateArmyBounds();
}

function updateArmyBounds() {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (const walker of walkers) {
    if (walker.basePosition.x < minX) minX = walker.basePosition.x;
    if (walker.basePosition.x > maxX) maxX = walker.basePosition.x;
    if (walker.basePosition.z < minZ) minZ = walker.basePosition.z;
    if (walker.basePosition.z > maxZ) maxZ = walker.basePosition.z;
  }

  if (!Number.isFinite(minX)) {
    minX = maxX = minZ = maxZ = 0;
  }

  // Pad by movement range + body footprint so wandering performers stay lit.
  const pad = Math.max(2 * state.scale, state.wanderRadius);
  armyBounds.centerX = (minX + maxX) * 0.5;
  armyBounds.centerZ = (minZ + maxZ) * 0.5;
  armyBounds.halfX = (maxX - minX) * 0.5 + pad;
  armyBounds.halfZ = (maxZ - minZ) * 0.5 + pad;
  armyBounds.radius = Math.hypot(armyBounds.halfX, armyBounds.halfZ);

  // Size the key shadow frustum to the spread, but cap it: a single 2048 map
  // stretched over a giant army gives sub-pixel (invisible) shadows. Capping
  // keeps per-character + self shadows crisp around the centre; performers far
  // past the cap still get floor contact via GTAO.
  const reach = Math.min(Math.max(armyBounds.halfX, armyBounds.halfZ) + 6, 34);
  key.shadow.camera.left = armyBounds.centerX - reach;
  key.shadow.camera.right = armyBounds.centerX + reach;
  key.shadow.camera.top = armyBounds.centerZ + reach;
  key.shadow.camera.bottom = armyBounds.centerZ - reach;
  key.shadow.camera.far = reach * 2.4 + 24;
  key.shadow.camera.updateProjectionMatrix();
}

function hashRandom(value) {
  const x = Math.sin(value * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function updateCrowdMotion(walker, elapsed) {
  walker.position.copy(walker.basePosition);

  if (state.movement === 'lane loop' && state.travel > 0) {
    const loopDistance = Math.max(state.count * state.spacing, 8);
    const travel = ((elapsed * state.speed * 1.25 + walker.index * state.spacing) % loopDistance) - loopDistance * 0.5;
    walker.position.z = walker.laneOffset + travel * state.travel * 0.38;
    return;
  }

  if (state.movement === 'local wander' && state.wanderRadius > 0) {
    const phase = elapsed * state.wanderSpeed + walker.seed * Math.PI * 2;
    walker.position.x += Math.sin(phase * 0.91) * state.wanderRadius * 0.35;
    walker.position.z += Math.cos(phase * 1.13) * state.wanderRadius * 0.35;
    walker.rotationY = Math.atan2(Math.cos(phase * 1.13), Math.sin(phase * 0.91));
    return;
  }

  if (state.movement === 'space wander' && state.wanderRadius > 0) {
    const phase = elapsed * state.wanderSpeed * 0.55 + walker.seed * Math.PI * 2;
    const radius = state.wanderRadius * (0.35 + walker.seed * 0.65);
    walker.position.x += Math.cos(phase) * radius;
    walker.position.z += Math.sin(phase * 0.83) * radius;
    walker.rotationY = -phase + Math.PI * 0.5;
    return;
  }

  if (state.movement === 'orbit drift' && state.wanderRadius > 0) {
    const phase = elapsed * state.wanderSpeed * (0.25 + walker.seed * 0.5) + walker.index;
    walker.position.x += Math.cos(phase) * state.wanderRadius;
    walker.position.z += Math.sin(phase) * state.wanderRadius;
    walker.rotationY = -phase + Math.PI * 0.5;
  }
}

function updateWalkerInstanceMatrix(walker) {
  const normalization = walker.batch.source.userData.floorNormalization;
  const scale = normalization.baseScale * state.scale;

  // Per-instance size + non-uniform proportions (centered on 1). The floor
  // recenter below is multiplied by this scale, so feet stay grounded even
  // when y differs. Proves instances share one skin yet vary independently.
  const sizeMul = 1 + (walker.sizeRand - 0.5) * state.sizeVariance;
  const propX = 1 + (walker.propRand.x - 0.5) * state.proportionVariance;
  const propY = 1 + (walker.propRand.y - 0.5) * state.proportionVariance;
  const propZ = 1 + (walker.propRand.z - 0.5) * state.proportionVariance;

  instancePosition.copy(walker.position);
  instanceScale.set(scale * sizeMul * propX, scale * sizeMul * propY, scale * sizeMul * propZ);
  instanceEuler.set(0, walker.rotationY, 0);
  instanceQuaternion.setFromEuler(instanceEuler);
  transformMatrix.compose(instancePosition, instanceQuaternion, instanceScale);
  normalizationMatrix.makeTranslation(
    -normalization.center.x,
    -normalization.minY,
    -normalization.center.z
  );
  transformMatrix.multiply(normalizationMatrix);
  transformMatrix.toArray(walker.batch.instanceMatrices.array, walker.batchIndex * 16);
}

// T28: push the tracked face expression into a walker's per-instance slot. Pose
// performers get the live scalars (when faceDrive is on); everyone else = neutral.
function writeFaceExpr(walker, batch) {
  if (!batch.faceExpr) return;
  const arr = batch.faceExpr.array;
  const o = walker.batchIndex * 8;
  // debugFaceOverride (dev/CDP): force an expression on EVERY instance, independent of
  // webcam/pose — lets the test harness drive the deform without a live face.
  const e = debugFaceOverride
    || ((walker.boneSource === 'pose' && state.faceDrive && latestFaceExpr) ? latestFaceExpr : null);
  if (e) {
    arr[o] = e.jawOpen; arr[o + 1] = e.smile; arr[o + 2] = e.pucker; arr[o + 3] = e.blinkL;
    arr[o + 4] = e.blinkR; arr[o + 5] = e.browL; arr[o + 6] = e.browR; arr[o + 7] = 0;
  } else {
    for (let k = 0; k < 8; k += 1) arr[o + k] = 0;
  }
}

// Live-tune the deform amounts from state; 0 when faceDrive is off (no rebuild).
function updateFaceUniforms(batch) {
  const u = batch.faceUniforms;
  if (!u) return;
  const on = state.faceDrive;
  u.jawAngle.value = on ? state.faceJawAngle : 0;
  u.lipDrop.value = on ? state.faceLipDrop : 0;
  u.cornerMove.value = on ? state.faceCornerMove : 0;
  u.lidDrop.value = on ? state.faceLidDrop : 0;
  u.browRaise.value = on ? state.faceBrowRaise : 0;
}

function choreoOpts() {
  return {
    choreography: state.choreography,
    choreoDelay: state.choreoDelay,
    count: walkers.length,
    bounds: armyBounds
  };
}

// Clip bone source: advance the shared mixer to this walker's choreographed
// clip time, then snapshot the re-posed skeleton into the walker's slice (V5).
function writeClipBoneSource(walker, batch, elapsed) {
  // idle batch: no mixer — leave the rig at its bind pose (constant rest matrices).
  if (!batch.idle) {
    const animOffset = computeWalkerPhase(walker, batch.duration, choreoOpts());
    batch.mixer.setTime((elapsed * state.speed + animOffset) % batch.duration);
  }
  batch.source.updateMatrixWorld(true);

  for (const skeletonState of batch.skeletonStates) {
    skeletonState.skeleton.update();
    skeletonState.boneMatrices.array.set(
      skeletonState.skeleton.boneMatrices,
      walker.batchIndex * skeletonState.boneCount * 16
    );
  }
}

// Pose bone source (T12/T14): every pose performer shares ONE canonical → ONE
// retarget. Compute the posed bone matrices once per batch, then copy into each
// pose performer's slice (V5/V20). Cost is O(1) retarget regardless of how many
// performers are pose-driven. No canonical (no person / pose off) → rest pose.
// Self-diagnosing: failure logs ONCE and holds rest so the loop survives.
let _poseWriteErr = false;
function computePoseMatrices(batch) {
  const skeleton = batch.skeletonStates[0]?.skeleton;
  if (!skeleton) {
    if (!_poseWriteErr) { console.error('[pose] batch has no skeleton:', batch.mesh?.name); _poseWriteErr = true; }
    return null;
  }

  try {
    if (!batch.retargeter) batch.retargeter = new Retargeter(skeleton, { restQuats: batch.restQuats });
    if (state.poseRetarget && latestCanonical) {
      batch.retargeter.apply(latestCanonical, {
        kptThresh: state.poseKptThresh,
        mirrorX: state.poseMirrorX,
        depthScale: state.poseDepthScale,
        jointLimitDeg: state.poseJointLimit,
        armLimit: state.poseArmLimit,
        follow: state.poseFollow,
        dt: poseRenderDt,
        smoothMs: state.poseSmoothTime,
        swingTwist: state.poseSwingTwist,
        headGain: state.poseHeadGain,
        planeFollow: state.poseTwistSmooth,
        wristTwist: state.poseWristTwist,
        grounding: state.poseGrounding,
        groundFollow: state.poseGroundFollow,
        bodyYaw: state.poseBodyYaw,
        facingDeadzone: state.poseFacingDeadzone,
        facingSmooth: state.poseFacingSmooth,
        facingBodyGain: state.poseFacingBodyGain,
        headPitchGain: state.poseHeadPitchGain,
        flipReject: state.poseFlipReject,
        clavicle: state.poseClavicle
      });
      // V19: surface which bones the joint-limit clamp throttled (e.g. face-touch
      // → forearm wanted 148° capped 110°). Peak-held top 3 into the HUD (stable);
      // 'Dump Clamps' button prints the full table + resets.
      lastPoseRetargeter = batch.retargeter;
      updateFacingArrows(batch.retargeter, batch.walkers?.[0]?.position);
      const df = batch.retargeter.debugFacing;
      if (df) poseStats.facing = `body θ${df.bodyTheta.toFixed(0)}→spine${df.spineYaw.toFixed(0)}  head θ${df.headTheta.toFixed(0)}→neck${df.neckYaw.toFixed(0)}`;
      const rep = batch.retargeter.clampReport();
      poseStats.clamp = rep.length
        ? rep.slice(0, 3).map((r) => `${r.bone} ${Math.round(r.raw)}→${r.max}`).join('  ')
        : 'none';
    } else {
      batch.retargeter.restPose();
    }
  } catch (error) {
    if (!_poseWriteErr) { console.error('[pose] retarget threw — holding rest pose:', error); _poseWriteErr = true; }
    batch.retargeter?.restPose();
  }

  batch.source.updateMatrixWorld(true);
  // Copy each skeleton's matrices (clip walkers re-pose the shared skeleton later
  // in the loop, so snapshot now into stable buffers).
  const snaps = [];
  for (const ss of batch.skeletonStates) {
    ss.skeleton.update();
    const src = ss.skeleton.boneMatrices;
    // Guard the SHARED buffer: a non-finite snapshot (NaN bone from a degenerate
    // frame) would vanish the whole batch. Bail → writePoseSlices keeps the last
    // good slice (holds the previous pose) instead of writing garbage.
    if (!src.every(Number.isFinite)) {
      if (!batch._poseNaNLogged) {
        batch._poseNaNLogged = true;
        console.warn('[pose] non-finite bone matrices — holding last good pose:', batch.mesh?.name);
      }
      return null;
    }
    if (!batch._poseLogged) {
      batch._poseLogged = true;
      console.log('[pose] driving', batch.mesh?.name, { bones: ss.skeleton.bones.length, retarget: state.poseRetarget, hasCanonical: !!latestCanonical });
    }
    snaps.push(Float32Array.from(src));
  }
  return snaps;
}

function writePoseSlices(walker, batch, posed) {
  if (!posed) return;
  batch.skeletonStates.forEach((ss, i) => {
    ss.boneMatrices.array.set(posed[i], walker.batchIndex * ss.boneCount * 16);
  });
}

function updateCrowdBatches(elapsed) {
  for (const batch of crowdBatches) {
    // One retarget per batch, shared by all pose performers.
    const hasPose = batch.walkers.some((w) => w.boneSource === 'pose');
    let posedMatrices = null;
    if (hasPose) {
      const rt0 = performance.now();
      posedMatrices = computePoseMatrices(batch);
      poseStats.retargetMs = ema(poseStats.retargetMs, performance.now() - rt0);
    }

    for (const walker of batch.walkers) {
      if (walker.boneSource === 'pose') {
        writePoseSlices(walker, batch, posedMatrices);
      } else {
        writeClipBoneSource(walker, batch, elapsed);
      }
      updateWalkerInstanceMatrix(walker);
      writeFaceExpr(walker, batch);
    }

    for (const skeletonState of batch.skeletonStates) skeletonState.boneMatrices.needsUpdate = true;
    batch.instanceMatrices.needsUpdate = true;
    if (batch.faceExpr) { updateFaceUniforms(batch); batch.faceExpr.needsUpdate = true; }
    for (const computeNode of batch.computeNodes) renderer.compute(computeNode);
  }
}

function clearWalkers() {
  for (const walker of walkers) {
    if (walker.lightRig) {
      scene.remove(walker.lightRig.target);
      scene.remove(walker.lightRig.keyLight);
      scene.remove(walker.lightRig.fillLight);
      scene.remove(walker.lightRig.rimLight);
    }
  }
  for (const batch of crowdBatches) {
    scene.remove(batch.source);
    batch.mixer?.stopAllAction();
    if (batch.faceMaskOverlay) { // T27: drop the debug vertex cloud with its batch
      scene.remove(batch.faceMaskOverlay);
      batch.faceMaskOverlay.geometry.dispose();
      batch.faceMaskOverlay.material.dispose();
      _faceMaskOverlays.delete(batch.faceMaskOverlay);
    }
  }
  skeletonStates.clear();
  crowdBatches.length = 0;
  walkers.length = 0;
}

async function loadGltf(url) {
  if (!assetCache.has(url)) {
    const promise = new Promise((resolve, reject) => {
      loader.load(
        assetUrl(url),
        resolve,
        (event) => {
          if (!onAssetProgress) return;
          assetProgress.set(url, { loaded: event.loaded, total: event.total || 0 });
          reportAssetProgress();
        },
        reject
      );
    });
    assetCache.set(url, promise);
  }
  return assetCache.get(url);
}

let onAssetProgress = null;
const assetProgress = new Map();

function reportAssetProgress() {
  if (!onAssetProgress) return;
  let loaded = 0;
  let total = 0;
  for (const entry of assetProgress.values()) {
    loaded += entry.loaded;
    total += entry.total;
  }
  onAssetProgress(loaded, total);
}

let poseProvider = null;
let poseOverlayCtx = null;
// Match the PiP box aspect to the actual overlay-canvas dims (native crop = 3:4 model
// aspect, worker webcam = 4:3/16:9). Without this the fixed 16:9 CSS box clips the feed.
function syncPipAspect() {
  if (poseOverlayEl.width && poseOverlayEl.height) {
    posePipEl.style.aspectRatio = `${poseOverlayEl.width} / ${poseOverlayEl.height}`;
  }
}

// --- Facing debug: draw the DETECTED body/head yaw (retargeter's 2D estimators) as
// arrows — a top-down compass in the PiP + 3D arrows on the driven performer. Cyan =
// body, magenta = head. Toggled by state.poseFacingDebug.
const facingArrowBody = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1.1, 0x00e5ff, 0.32, 0.2);
const facingArrowHead = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.8, 0xff4fd8, 0.26, 0.16);
facingArrowBody.visible = false;
facingArrowHead.visible = false;
scene.add(facingArrowBody, facingArrowHead);
// 3D pose skeleton as octahedron bones + joint markers, fed WORLD-aligned joint
// positions by updatePoseDebug3D (same mesh anchoring as the green line skeleton) — so
// it lands on the mesh instead of floating. Replaces the 1px green pose lines.
const poseSkeleton3D = new PoseSkeleton3D();
poseSkeleton3D.visible = false;
scene.add(poseSkeleton3D);
const _skelJoints = new Array(23).fill(null); // reused world-position buffer

const _faceDir = new THREE.Vector3();
function updateFacingArrows(retargeter, anchor) {
  const on = !!(state.poseFacingDebug && retargeter && anchor);
  facingArrowBody.visible = on;
  facingArrowHead.visible = on;
  if (!on) return;
  // θ about vertical: 0 = frontal (+z, toward camera). sin/cos → world horizontal dir.
  _faceDir.set(Math.sin(retargeter.torsoFacing.theta), 0, Math.cos(retargeter.torsoFacing.theta)).normalize();
  facingArrowBody.position.set(anchor.x, anchor.y + 1.0, anchor.z);
  facingArrowBody.setDirection(_faceDir);
  // Head arrow: yaw (horizontal) + PITCH (nod, vertical tilt). pitch is raw ±~0.5; ×2
  // → visible tilt. Nod down (pitch +) → arrow tilts down (−y).
  const hy = retargeter.headFacing.theta;
  const hp = retargeter.headFacing.pitch * 2;
  const cp = Math.cos(hp);
  _faceDir.set(Math.sin(hy) * cp, -Math.sin(hp), Math.cos(hy) * cp).normalize();
  facingArrowHead.position.set(anchor.x, anchor.y + 1.75, anchor.z);
  facingArrowHead.setDirection(_faceDir);
}

// Top-down facing compass drawn in the PiP corner. θ=0 → arrow points DOWN (toward the
// viewer = frontal); it swings as the body/head turn. Sized to the canvas.
function drawFacingCompass(ctx, w, h, bodyYaw, headYaw) {
  const R = Math.max(16, Math.min(w, h) * 0.09);
  const cx = R + 8;
  const cy = R + 8;
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(2,4,10,0.55)'; ctx.fill();
  ctx.strokeStyle = 'rgba(120,255,240,0.35)'; ctx.lineWidth = 1; ctx.stroke();
  const arrow = (ang, color, len) => {
    const dx = Math.sin(ang); const dy = Math.cos(ang);
    const ex = cx + dx * len; const ey = cy + dy * len;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
    const a = Math.atan2(dy, dx); const hd = R * 0.35;
    ctx.beginPath(); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - hd * Math.cos(a - 0.5), ey - hd * Math.sin(a - 0.5));
    ctx.lineTo(ex - hd * Math.cos(a + 0.5), ey - hd * Math.sin(a + 0.5));
    ctx.closePath(); ctx.fill();
  };
  arrow(bodyYaw, '#00e5ff', R - 3); // body cyan
  arrow(headYaw, '#ff4fd8', R - R * 0.42); // head magenta (shorter)
  ctx.restore();
}

let poseLoopActive = false;
let poseLoopId = 0; // generation guard — only the latest loop runs (no stacking)
let poseGen = 0; // start/stop generation — an in-flight startPose superseded by a stop/switch tears down its own provider (no leaked worker/WS)
let poseRenderDt = 1 / 60; // last render-frame delta (s) → retarget's frame-rate-independent rig blend
let latestCanonical = null;
let lastPoseFrame = null; // most recent raw frame (calibration overlay skeleton)
let lastGoodPoseAt = 0;
let lastPoseRetargeter = null; // V19: active pose retargeter, for the clamp dump button
const poseSmoother = new CanonicalSmoother(NUM_KPTS, {
  minCutoff: 2,
  beta: 0.02
});
// T26 (V26): per-performer face expression from landmarks 23-90. One tracked feed →
// one extractor for now (rides the same pose-performer concept, I.faceExpr). faceStats
// is a stable object the Inspector .listen()s (extractor.value is copied in, so a reset
// re-binding can't break the display).
const faceExpr = new FaceExpressionExtractor();
let latestFaceExpr = null;
let debugFaceOverride = null; // dev/CDP: forced expression (see writeFaceExpr / window.__vj.setFace)
let faceExprPrevT = null;
const faceStats = { jawOpen: 0, smile: 0, pucker: 0, blinkL: 0, blinkR: 0, browL: 0, browR: 0, neutral: 'auto…' };
const poseRecorder = new PoseRecorder();
let replayActive = false;
let replayPlayer = null;
let lastPoseFrameAt = 0;

// Per-stage timing breakdown (ms), EMA-smoothed, shown in the Pose Stats panel.
const poseStats = {
  backend: 'worker',
  poseFps: 0,
  detectMs: 0,
  preprocessMs: 0,
  inferenceMs: 0,
  decodeMs: 0,
  captureMs: 0, // sidecar: main-thread frame snapshot (createImageBitmap vs WebCodecs VideoFrame)
  previewEncMs: 0, // native: sidecar JPEG encode of the preview
  previewDecMs: 0, // native: browser decode of the preview
  previewKb: 0, // native: preview JPEG size
  readbackMs: 0, // sidecar: GPU→CPU frame readback (browser side)
  serverMs: 0, // sidecar: full server handler (recv→reply); serverMs−inference = numpy/json overhead
  transportMs: 0, // sidecar: round−server = WS + browser scheduling lag (this is "wire")
  roundMs: 0, // sidecar: send→reply wall-clock
  wireMs: 0, // alias of transport
  frames: '—', // sidecar flow counters: sent/recv drop/stale/timeout @ Hz
  overlayMs: 0,
  retargetMs: 0,
  clamp: '—', // V19: bones the joint-limit clamp throttled this frame (wanted°→cap°)
  facing: '—' // DEBUG: estimator θ (body/head) → yaw actually applied to spine/neck bones
};
const ema = (prev, next, a = 0.2) => prev + a * (next - prev);

// Minimal robustness gate (V11/V12, ahead of full T9): only drive from a frame
// whose CORE joints (hips + shoulders) are confident — a garbage/partial frame
// would otherwise fling the rig and the mesh vanishes. Brief dropout → hold last
// good; sustained loss (>500ms) → release to bind. A real outlier/teleport layer
// (T9) replaces this.
const POSE_CORE_JOINTS = [KPT.leftHip, KPT.rightHip, KPT.leftShoulder, KPT.rightShoulder];

function frameCoreConfident(canon) {
  return POSE_CORE_JOINTS.every((i) => canon.joints[i] && canon.joints[i].confidence >= state.poseKptThresh);
}

// Guided calibration (§6): capture a neutral A-pose ~2s → median limb lengths.
// Direction-based retarget doesn't need them to STRETCH-fix, but they gate bad
// joints (a bone gone implausibly long/short = a bad measurement → reject, §16).
const CALIB_BONES = [
  ['lThigh', KPT.leftHip, KPT.leftKnee],
  ['lShin', KPT.leftKnee, KPT.leftAnkle],
  ['rThigh', KPT.rightHip, KPT.rightKnee],
  ['rShin', KPT.rightKnee, KPT.rightAnkle],
  ['lUarm', KPT.leftShoulder, KPT.leftElbow],
  ['lFarm', KPT.leftElbow, KPT.leftWrist],
  ['rUarm', KPT.rightShoulder, KPT.rightElbow],
  ['rFarm', KPT.rightElbow, KPT.rightWrist]
];
// Guided calibration phases: idle → ready (3s get-into-pose) → capturing (2s
// sample) → done (1.5s result). Drives the on-screen overlay.
const CALIB_READY_MS = 5000;
const CALIB_CAPTURE_MS = 2000;
const CALIB_DONE_MS = 1600;
let calibPhase = 'idle';
let calibPhaseStart = 0;
const calibSamples = {};
const poseCalibration = { lengths: null, ready: false };

function jointDist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Bone-length plausibility gate: any driven bone wildly off its calibrated length
// → a bad joint this frame → treat the frame as an outlier.
function boneLengthBad(canon) {
  if (!poseCalibration.ready) return false;
  for (const [k, a, b] of CALIB_BONES) {
    const ja = canon.joints[a];
    const jb = canon.joints[b];
    if (!ja || !jb || ja.confidence < state.poseKptThresh || jb.confidence < state.poseKptThresh) continue;
    const ratio = jointDist(ja, jb) / poseCalibration.lengths[k];
    if (ratio < 0.55 || ratio > 1.6) return true;
  }
  return false;
}

// Outlier reject (§14): did a core joint teleport vs the last accepted frame?
let lastAcceptedRaw = null;
let lastAcceptedAt = 0;
let consecutiveRejects = 0;
const MAX_REJECTS = 6; // after this many, accept (genuine fast motion, not a glitch)

function coreJumped(raw, now) {
  if (!lastAcceptedRaw) return false;
  // VELOCITY-based, not per-frame displacement: scale the jump budget by elapsed
  // time so a low pose-fps backend (sidecar at ~7fps) doesn't false-reject normal
  // motion (big between sparse frames) → that was the periodic "glitch". 33ms ≈ the
  // 30fps reference at which poseMaxJump was tuned; clamp so a long stall still gates.
  const dt = lastAcceptedAt ? Math.max(16, now - lastAcceptedAt) : 33;
  const budget = state.poseMaxJump * Math.min(6, dt / 33);
  const max2 = budget * budget;
  for (const i of POSE_CORE_JOINTS) {
    const a = raw.joints[i];
    const b = lastAcceptedRaw.joints[i];
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    if (dx * dx + dy * dy + dz * dz > max2) return true;
  }
  return false;
}

function updateCanonical(frame) {
  // Adapter boundary: MediaPipe frames carry their own topology/axis convention → their
  // own canonical builder. RTMW (worker/sidecar/native) uses toCanonical.
  const raw = frame ? (frame.source === 'mediapipe' ? toCanonicalFromMediaPipe(frame) : toCanonical(frame)) : null;
  const now = performance.now();

  if (raw) {
    // Drive whatever is confident — do NOT require all core joints (that forced an
    // A-pose fallback when e.g. legs/hips dropped). Each bone self-gates on its own
    // joints' confidence in the retargeter; un-tracked regions rest, tracked drive.
    // Reject before smooth (§54): a sudden teleport OR an implausible bone length
    // is an outlier → hold the last good pose (unless it persists = real motion).
    const outlier = coreJumped(raw, now) || (state.poseBoneGate && boneLengthBad(raw));
    if (state.poseRejectOutliers && outlier && consecutiveRejects < MAX_REJECTS) {
      consecutiveRejects += 1;
      return;
    }
    consecutiveRejects = 0;
    lastAcceptedRaw = raw;
    lastAcceptedAt = now;
    let canon = raw;
    if (state.poseSmoothing) {
      poseSmoother.setParams({ minCutoff: state.poseSmoothMinCutoff, beta: state.poseSmoothBeta });
      canon = poseSmoother.smooth(raw);
    }
    // Native capture owns the RAW camera (not a selfie); the browser worker feeds a
    // mirrored selfie. Opposite handedness → auto-invert the mirror for native so ONE
    // "Mirror Pose" setting stays aligned across both backends.
    const isNative = poseProvider instanceof NativeCapturePoseProvider;
    const doMirror = isNative ? !state.poseMirror : state.poseMirror;
    latestCanonical = doMirror ? mirrorCanonical(canon) : canon;
    lastGoodPoseAt = now;

    // T26: extract face expression scalars from the same canonical frame. dt from the
    // frame timestamp (pose-rate); the extractor's own One Euro smooths per scalar.
    const faceDt = faceExprPrevT === null ? 0 : (latestCanonical.timestampMs - faceExprPrevT) / 1000;
    faceExprPrevT = latestCanonical.timestampMs;
    // MediaPipe gives ARKit blendshapes directly (richer + already neutral-relative);
    // RTMW extracts the 7 scalars from the sparse face landmarks.
    if (frame.faceBlendshapes) {
      latestFaceExpr = blendshapesToExpr(frame.faceBlendshapes);
      faceStats.neutral = 'blendshapes';
    } else {
      latestFaceExpr = faceExpr.update(latestCanonical, faceDt, { thresh: state.poseKptThresh });
      faceStats.neutral = faceExpr.neutral ? 'set' : 'capturing…';
    }
    Object.assign(faceStats, latestFaceExpr);

    // Calibration: collect bone-length samples during the capture phase (finalize
    // + countdown are driven by updateCalibration in the render loop).
    if (calibPhase === 'capturing') {
      for (const [k, a, b] of CALIB_BONES) {
        const ja = raw.joints[a];
        const jb = raw.joints[b];
        if (ja && jb && ja.confidence >= state.poseKptThresh && jb.confidence >= state.poseKptThresh) {
          calibSamples[k].push(jointDist(ja, jb));
        }
      }
    }
  } else if (now - lastGoodPoseAt > state.poseHoldMs) {
    latestCanonical = null; // sustained loss → bind (hold last-known until then)
    lastAcceptedRaw = null;
  }
  // else: brief dropout → keep holding the last good pose
}

// Ask the sidecar which camera indices are live (separate one-shot WS). Best run
// while NOT in native mode (enumerating opens cameras → would clash with an active
// native capture). Logs the full list; shows live indices in the status line.
function scanCameras() {
  statusEl.textContent = 'Scanning cameras…';
  let ws;
  try { ws = new WebSocket('ws://127.0.0.1:8787'); } catch { statusEl.textContent = 'Scan failed (no sidecar?)'; return; }
  ws.onopen = () => ws.send(JSON.stringify({ type: 'config', mode: 'scan' }));
  ws.onerror = () => { statusEl.textContent = 'Scan failed — is `npm run sidecar` running?'; };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type !== 'cameras') return;
    const rows = msg.list.map((c) => `  device ${c.index} (${c.backend}): ${c.live ? 'LIVE ✓' : 'black/none ✗'}  ${c.w}x${c.h}`);
    console.log('[cameras]\n' + rows.join('\n'));
    const live = msg.list.filter((c) => c.live).map((c) => c.index);
    statusEl.textContent = live.length
      ? `Cameras — LIVE indices: [${live.join(', ')}] (full list in console). Set "Native Device".`
      : 'No live cameras found — check the device is free + privacy settings (full list in console).';
    try { ws.close(); } catch { /* already closing */ }
  };
}

async function startPose() {
  if (poseProvider) return;
  const myGen = ++poseGen; // claim this start; a later stop/switch bumps poseGen → we bail + tear down
  posePipEl.hidden = false;
  // Sidecar backends load the model NATIVELY (nothing downloads in the browser); only
  // the in-browser worker compiles the GLB/onnx here.
  const nativeBackend = state.poseBackend.startsWith('sidecar');
  posePipLabel.textContent = nativeBackend ? 'pose: connecting sidecar…' : 'pose: loading model…';
  statusEl.textContent = nativeBackend
    ? 'Connecting to native sidecar (ws://127.0.0.1:8787)…'
    : 'Loading pose model (rtmw3d-l ~219MB) — first load compiles GPU shaders, may briefly stutter…';
  const t0 = performance.now();
  // Build into a LOCAL — only adopt into module `poseProvider` AFTER start() succeeds and
  // only if still current. Otherwise a superseded start would leak its worker/WS/camera.
  let provider = null;
  try {
    poseSmoother.reset();
    if (state.poseBackend === 'sidecar') {
      provider = new SidecarPoseProvider({ kptThresh: state.poseKptThresh, sendMaxSide: state.poseSendMaxSide, readback: state.poseReadback });
      provider.debug = state.poseSidecarDebug;
    } else if (state.poseBackend === 'sidecar-native') {
      provider = new NativeCapturePoseProvider({ kptThresh: state.poseKptThresh, width: 640, height: 480, preview: state.posePreview, device: state.poseNativeDevice });
      provider.debug = state.poseSidecarDebug;
    } else if (state.poseBackend === 'mediapipe') {
      provider = new MediaPipeProvider({ kptThresh: state.poseKptThresh, face: true, hands: state.poseMediaPipeHands });
    } else {
      const Provider = state.poseWorker ? WorkerPoseProvider : RTMWPoseProvider;
      provider = new Provider({ ep: state.poseEP, kptThresh: state.poseKptThresh, yoloRes: state.poseYoloRes, rtmwVariant: state.poseRtmwVariant });
    }
    poseStats.backend = state.poseBackend;
    await provider.start();
    if (myGen !== poseGen) { provider.stop(); return; } // superseded during start → tear down, no leak
    poseProvider = provider;
    const label = state.poseBackend.startsWith('sidecar') ? `${state.poseBackend}/${poseProvider.ep}` : `${state.poseWorker ? 'worker' : 'main'}/${state.poseEP}`;
    console.info(`[pose] ready in ${((performance.now() - t0) / 1000).toFixed(1)}s (${label})`);
    statusEl.textContent = '';
    // Native mode has no browser camera → no stream to preview; the overlay+mesh are the feedback.
    if (poseProvider.stream) { poseVideoEl.srcObject = poseProvider.stream; await poseVideoEl.play(); }
    else { poseVideoEl.srcObject = null; }
    poseOverlayEl.width = poseProvider.video.videoWidth || 1280;
    poseOverlayEl.height = poseProvider.video.videoHeight || 720;
    syncPipAspect();
    poseOverlayCtx = poseOverlayEl.getContext('2d');
    poseLoopActive = true;
    posePipLabel.textContent = `pose: ${label}`;
    poseLoop(++poseLoopId);
  } catch (error) {
    try { provider?.stop(); } catch { /* partial start — best-effort cleanup */ }
    if (myGen === poseGen) { // only report/reset if WE are still the current attempt
      console.error('pose start failed:', error);
      posePipLabel.textContent = `pose: ${error.message}`;
      statusEl.textContent = `Pose start failed: ${error.message}`;
      state.poseEnabled = false;
      poseProvider = null;
    }
  }
}

// Draw overlay + drive canonical from one frame. Shared by live + replay.
function consumePoseFrame(frame) {
  if (poseOverlayCtx) {
    const ov0 = performance.now();
    if (state.poseOverlay) {
      drawOverlay(poseOverlayCtx, frame, poseOverlayEl.width, poseOverlayEl.height, {
        kptThresh: state.poseKptThresh,
        mirror: true,
        bg: poseProvider?.preview ?? null // native-capture preview backdrop (no <video> there)
      });
    } else {
      poseOverlayCtx.clearRect(0, 0, poseOverlayEl.width, poseOverlayEl.height);
    }
    if (state.poseFacingDebug && lastPoseRetargeter) {
      drawFacingCompass(poseOverlayCtx, poseOverlayEl.width, poseOverlayEl.height,
        lastPoseRetargeter.torsoFacing.theta, lastPoseRetargeter.headFacing.theta);
    }
    poseStats.overlayMs = ema(poseStats.overlayMs, performance.now() - ov0);
  }
  lastPoseFrame = frame; // for the calibration-overlay skeleton view
  updateCanonical(frame);
  window.__poseCanonical = latestCanonical;
}

// Latest-frame loop: always infers the current video frame, so no backlog builds
// (V17 worker offload comes in T20; this keeps it newest-only on the main thread).
async function poseLoop(myId) {
  while (poseLoopActive && poseProvider && myId === poseLoopId) {
    try {
      poseProvider.detectEveryN = state.poseDetectEveryN;
      const frame = await poseProvider.infer();
      const vw = poseProvider.video?.videoWidth;
      const vh = poseProvider.video?.videoHeight;
      if (vw && poseOverlayEl.width !== vw) {
        poseOverlayEl.width = vw;
        poseOverlayEl.height = vh;
        syncPipAspect(); // match the pip box to the feed/crop aspect so nothing is clipped
      }
      poseRecorder.capture(frame);
      consumePoseFrame(frame);

      // Stage breakdown from the provider + actual pose throughput (fps).
      const tm = poseProvider.timings;
      poseStats.detectMs = ema(poseStats.detectMs, tm.detect);
      poseStats.preprocessMs = ema(poseStats.preprocessMs, tm.preprocess);
      poseStats.inferenceMs = ema(poseStats.inferenceMs, tm.inference);
      poseStats.decodeMs = ema(poseStats.decodeMs, tm.decode);
      poseStats.captureMs = ema(poseStats.captureMs, tm.capture ?? 0);
      poseStats.previewEncMs = ema(poseStats.previewEncMs, tm.previewEncode ?? 0);
      poseStats.previewDecMs = ema(poseStats.previewDecMs, tm.previewDecode ?? 0);
      poseStats.previewKb = ema(poseStats.previewKb, (tm.previewBytes ?? 0) / 1024);
      poseStats.readbackMs = ema(poseStats.readbackMs, tm.readback ?? 0);
      poseStats.serverMs = ema(poseStats.serverMs, tm.serverTotal ?? 0);
      poseStats.transportMs = ema(poseStats.transportMs, tm.transport ?? tm.wire ?? 0);
      poseStats.roundMs = ema(poseStats.roundMs, tm.round ?? 0);
      poseStats.wireMs = ema(poseStats.wireMs, tm.transport ?? tm.wire ?? 0);
      const st = poseProvider.stats;
      if (st) poseStats.frames = `${st.sent}/${st.recv} drop${st.dropped} stale${st.stale} TO${st.timedOut} @${(st.sendHz || 0).toFixed(0)}Hz`;
      const now = performance.now();
      if (lastPoseFrameAt) poseStats.poseFps = ema(poseStats.poseFps, 1000 / Math.max(1, now - lastPoseFrameAt));
      lastPoseFrameAt = now;

      // Yield a render frame between inferences so three's GPU work + the webgpu
      // command queue drain instead of being hammered back-to-back (eases the
      // shared-GPU contention that makes webgpu inference climb under load).
      await new Promise((resolve) => requestAnimationFrame(resolve));
    } catch (error) {
      console.error('pose loop error:', error);
      poseLoopActive = false;
      posePipLabel.textContent = `pose err: ${error.message}`;
    }
  }
}

// Replay a recording through the pipeline (no webcam) — iterate retarget/smooth/
// twist against a fixed motion. Pauses live inference while active.
function startReplay() {
  if (!poseRecorder.length) {
    statusEl.textContent = 'No recording to replay.';
    state.poseReplaying = false;
    return;
  }
  poseLoopActive = false; // pause live
  poseLoopId += 1; // kill any in-flight live loop
  poseSmoother.reset();
  const meta = poseRecorder.metadata ?? {};
  poseOverlayEl.width = meta.inputWidth || poseOverlayEl.width || 1280;
  poseOverlayEl.height = meta.inputHeight || poseOverlayEl.height || 720;
  syncPipAspect();
  poseOverlayCtx = poseOverlayEl.getContext('2d');
  posePipEl.hidden = false;
  posePipLabel.textContent = `replay: ${poseRecorder.length}f`;
  replayPlayer = new PosePlayer(poseRecorder.frames);
  replayActive = true;
  const t0 = performance.now();
  const step = () => {
    if (!replayActive) return;
    consumePoseFrame(replayPlayer.frameAt(performance.now() - t0));
    requestAnimationFrame(step);
  };
  step();
}

function stopReplay() {
  replayActive = false;
  replayPlayer = null;
  if (!state.poseEnabled) {
    if (poseOverlayCtx) poseOverlayCtx.clearRect(0, 0, poseOverlayEl.width, poseOverlayEl.height);
    posePipEl.hidden = true;
    posePipLabel.textContent = 'pose: off';
    latestCanonical = null;
  }
}

function stopPose() {
  poseGen += 1; // invalidate any in-flight startPose → it stops the provider it built (no leak)
  poseLoopActive = false;
  poseLoopId += 1; // invalidate any in-flight loop so it can't resume on restart
  if (poseProvider) {
    poseProvider.stop();
    poseProvider = null;
  }
  poseVideoEl.srcObject = null;
  facingArrowBody.visible = false;
  facingArrowHead.visible = false;
  poseSkeleton3D.visible = false;
  if (poseOverlayCtx) poseOverlayCtx.clearRect(0, 0, poseOverlayEl.width, poseOverlayEl.height);
  posePipEl.hidden = true;
  posePipLabel.textContent = 'pose: off';
  latestCanonical = null;
  calibPhase = 'idle';
  calibOverlay.hidden = true;
  poseCalibration.ready = false; // re-calibrate per person/session
}

// 3D debug overlay: magenta = actual driven rig bones (on the mesh via the
// instance matrix); green = the canonical pose that drove them, pelvis-aligned to
// the mesh and scaled to match. Compare → see what the retarget did with the pose.
function updatePoseDebug3D() {
  const show = state.poseDebug3D && !!latestCanonical;
  // Green pose is now drawn as octahedron bones (poseSkeleton3D), not 1px lines → hide
  // the old green line/points; keep the magenta rig lines as the interpretation reference.
  poseDebugLines.visible = false;
  poseDebugPoints.visible = false;
  poseSkeleton3D.visible = show;
  poseRigLines.visible = show;
  poseRigPoints.visible = show;
  if (!show) return;
  const walker = walkers.find((w) => w.boneSource === 'pose') ?? walkers[0];
  const skeleton = walker?.batch?.skeletonStates?.[0]?.skeleton;
  if (!walker || !skeleton) {
    poseSkeleton3D.visible = poseRigLines.visible = poseRigPoints.visible = false;
    return;
  }

  // Walker's instance matrix (source-space bone pos → world, lands on the mesh).
  _instMat.fromArray(walker.batch.instanceMatrices.array, walker.batchIndex * 16);
  const byName = new Map(skeleton.bones.map((b) => [b.name, b]));
  const worldOf = (name) => {
    const bone = byName.get(name);
    if (!bone) return null;
    return _vA.setFromMatrixPosition(bone.matrixWorld).applyMatrix4(_instMat).clone();
  };

  // Magenta rig bones.
  const rp = poseRigGeom.attributes.position.array;
  let m = 0;
  for (const [a, b] of RIG_EDGE_NAMES) {
    const pa = worldOf(a);
    const pb = worldOf(b);
    if (!pa || !pb) continue;
    rp[m++] = pa.x; rp[m++] = pa.y; rp[m++] = pa.z;
    rp[m++] = pb.x; rp[m++] = pb.y; rp[m++] = pb.z;
  }
  poseRigGeom.setDrawRange(0, m / 3);
  poseRigGeom.attributes.position.needsUpdate = true;

  // Green canonical → pelvis anchored at the mesh Hips, scaled to the MESH HEIGHT: match
  // the pose's Hips→top-of-head span to the rig's, using the vertical (y) extent only.
  // (The old 3D-torso ratio pulsed with the jittery z → "weirdly scaling"; y is stable.)
  const hipsW = worldOf('Hips');
  const headW = worldOf('Head');
  const J = latestCanonical.joints;
  let canonTop = 0.2; // highest confident body joint above the pelvis (≈ head)
  for (let i = 0; i <= 16; i += 1) { const j = J[i]; if (j && j.confidence >= state.poseKptThresh && j.y > canonTop) canonTop = j.y; }
  const rigUp = (headW && hipsW) ? Math.max(0.1, headW.y - hipsW.y) : 0.6; // rig Hips→Head height
  const s = (rigUp / canonTop) * state.poseDebugScale;
  const ox = hipsW ? hipsW.x : walker.position.x;
  const oy = hipsW ? hipsW.y : state.poseDebugHeight;
  const oz = hipsW ? hipsW.z : walker.position.z;
  // Per-keypoint world positions → octahedron skeleton. z damped ×0.4 (z is Z_RANGE-
  // scaled, ~larger than x/y — undamped it blows the depth out).
  for (let i = 0; i < 23; i += 1) {
    const j = J[i];
    _skelJoints[i] = (j && j.confidence >= state.poseKptThresh)
      ? { x: ox + j.x * s, y: oy + j.y * s, z: oz + j.z * s * 0.4 }
      : null;
  }
  poseSkeleton3D.update(_skelJoints);
}

// Guided calibration phase machine + on-screen overlay (countdown a single
// performer can read from across the room).
const CALIB_FIGURE = `<svg width="110" height="170" viewBox="0 0 110 170" fill="none" stroke="#7df9ee" stroke-width="5" stroke-linecap="round" style="display:block;margin:0 auto 16px">
  <circle cx="55" cy="22" r="14"/>
  <line x1="55" y1="36" x2="55" y2="100"/>
  <line x1="55" y1="50" x2="18" y2="94"/>
  <line x1="55" y1="50" x2="92" y2="94"/>
  <line x1="55" y1="100" x2="38" y2="162"/>
  <line x1="55" y1="100" x2="72" y2="162"/>
</svg>`;
function setCalibText(line, big, figure = false) {
  const fig = figure ? CALIB_FIGURE : '';
  const num = big != null ? `<span class="big">${big}</span>` : '';
  calibTextEl.innerHTML = `${fig}${line}${num}`;
}
function updateCalibration() {
  if (calibPhase === 'idle') { calibOverlay.hidden = true; return; }
  calibOverlay.hidden = false;
  // Live detected skeleton so you SEE the tracking while calibrating (not just a guide).
  if (lastPoseFrame && poseProvider) {
    const vw = poseProvider.video?.videoWidth || 320;
    const vh = poseProvider.video?.videoHeight || 427;
    if (calibSkeletonEl.width !== vw) { calibSkeletonEl.width = vw; calibSkeletonEl.height = vh; }
    drawOverlay(calibSkeletonCtx, lastPoseFrame, vw, vh, { kptThresh: state.poseKptThresh, mirror: true, bg: poseProvider.preview ?? null });
    calibSkeletonEl.hidden = false;
  } else {
    calibSkeletonEl.hidden = true;
  }
  const t = performance.now() - calibPhaseStart;

  if (calibPhase === 'ready') {
    setCalibText('Stand like this — full body in frame', Math.ceil((CALIB_READY_MS - t) / 1000), true);
    if (t >= CALIB_READY_MS) {
      for (const [k] of CALIB_BONES) calibSamples[k] = [];
      // Reset the facing frontal reference: the "stand like this" pose IS frontal, so the
      // running-max relearns THIS person/distance's frontal width during the hold.
      for (const b of crowdBatches) b.retargeter?.recalibrateFacing();
      faceExpr.recalibrateNeutral(); // T26: the frontal "stand like this" hold = neutral face
      calibPhase = 'capturing';
      calibPhaseStart = performance.now();
    }
  } else if (calibPhase === 'capturing') {
    setCalibText('Hold still — calibrating', Math.ceil((CALIB_CAPTURE_MS - t) / 1000), true);
    if (t >= CALIB_CAPTURE_MS) {
      const lengths = {};
      let ok = true;
      for (const [k] of CALIB_BONES) {
        const s = calibSamples[k];
        if (s.length < 5) { ok = false; break; }
        s.sort((x, y) => x - y);
        lengths[k] = s[s.length >> 1];
      }
      if (ok) {
        poseCalibration.lengths = lengths;
        poseCalibration.ready = true;
        setCalibText('✓ Calibrated');
      } else {
        setCalibText('✗ Full body not visible — try again');
      }
      calibPhase = 'done';
      calibPhaseStart = performance.now();
    }
  } else if (calibPhase === 'done') {
    if (t >= CALIB_DONE_MS) { calibPhase = 'idle'; calibOverlay.hidden = true; }
  }
}

function render(timestamp) {
  timer.update(timestamp);
  const delta = Math.min(timer.getDelta(), 0.05);
  const elapsed = timer.getElapsed();
  poseRenderDt = delta; // render frame time → the retarget's frame-rate-independent blend

  for (const walker of walkers) {
    updateCrowdMotion(walker, elapsed);
    updatePerformerLightRig(walker, elapsed);
  }
  updateCrowdBatches(elapsed);
  updatePoseDebug3D();
  updateCalibration();

  const bx = armyBounds.centerX;
  const bz = armyBounds.centerZ;
  const hx = armyBounds.halfX;
  const hz = armyBounds.halfZ;
  const coverage = armyBounds.radius + 10;

  // lightMotion=false zeroes the oscillation so lights freeze in place but stay
  // lit; glowLights toggles the decorative point glows entirely.
  const m = state.lightMotion ? 1 : 0;

  floorGlow.visible = backGlow.visible = sideGlow.visible = state.glowLights;
  if (state.glowLights) {
    floorGlow.intensity = state.floorLightBase + Math.sin(elapsed * 1.7) * state.floorLightBase * 0.16 * m;
    floorGlow.position.x = bx + Math.sin(elapsed * 0.55) * hx * m;
    floorGlow.position.z = bz + Math.cos(elapsed * 0.43) * hz * m;
    floorGlow.distance = coverage;
    backGlow.position.x = bx + Math.sin(elapsed * 0.4) * hx * m;
    backGlow.position.z = bz - hz - 2;
    backGlow.intensity = state.backLightBase;
    backGlow.distance = coverage * 1.25;
    sideGlow.position.x = bx + Math.cos(elapsed * 0.31) * hx * m;
    sideGlow.position.z = bz + Math.sin(elapsed * 0.29) * hz * m;
    sideGlow.distance = coverage;
  }
  heroSpot.position.x = bx + Math.sin(elapsed * 0.18) * hx * m;
  heroSpot.position.y = 9 + armyBounds.radius * 0.35;
  heroSpot.position.z = bz + hz + 6;
  heroSpot.target.position.x = bx + Math.sin(elapsed * 0.22) * hx * 0.5 * m;
  heroSpot.target.position.z = bz;
  heroSpot.distance = Math.max(34, coverage * 2);
  heroSpot.angle = Math.min(Math.PI * 0.48, Math.PI * 0.3 + armyBounds.radius * 0.015);
  cameraFill.position.copy(camera.position);
  cameraFill.position.y += 0.75;
  cameraFill.target.position.copy(controls.target);

  controls.update();
  renderPipeline.render();
  updateRenderStats(delta);
}

function updateRenderStats(delta) {
  const renderInfo = renderer.info.render;
  const memoryInfo = renderer.info.memory;

  statsState.fps = delta > 0 ? Math.round(1 / delta) : 0;
  statsState.calls = renderInfo.calls;
  statsState.triangles = renderInfo.triangles;
  statsState.lines = renderInfo.lines;
  statsState.points = renderInfo.points;
  statsState.geometries = memoryInfo.geometries;
  statsState.textures = memoryInfo.textures;
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// Dev-only debug bridge (T28/T29): lets the CDP test harness read live face state and
// flip inspector toggles from outside the module. Stripped from production builds.
if (import.meta.env.DEV) {
  window.__vj = {
    state,
    get batches() { return crowdBatches; },
    faceExpr,
    get face() { return latestFaceExpr; },
    camera,
    controls,
    refreshFaceMaskOverlays,
    setFace(o) { debugFaceOverride = o; },
    clearFace() { debugFaceOverride = null; },
    faceEditor,
    setFaceEditMode,
    faceEditReseed,
    faceEditPaint,
    faceEditClearRegion,
    editorActions,
    editorState,
    THREE
  };
}
