import './styles.css';
import * as THREE from 'three';
import { RenderPipeline, StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import {
  Fn,
  add,
  attributeArray,
  emissive,
  instanceIndex,
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

const canvas = document.querySelector('#scene');
const statusEl = document.querySelector('#status');

// Resolve manifest/model URLs against Vite's base so the app works both at the
// dev root and under a GitHub Pages project subpath (e.g. /mesh-test/).
const BASE_URL = import.meta.env.BASE_URL;
const assetUrl = (path) => `${BASE_URL}${String(path).replace(/^\//, '')}`;

const preloaderEl = document.querySelector('#preloader');
const preloaderBar = document.querySelector('#preloader-bar');
const preloaderLabel = document.querySelector('#preloader-label');
const preloaderPct = document.querySelector('#preloader-pct');

const renderer = new WebGPURenderer({
  canvas,
  antialias: true,
  alpha: true
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
camera.position.set(0, 3.4, 20);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.055;
controls.target.set(0, 0.95, 0);
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 3.5;
controls.maxDistance = 38;

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
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 40;
key.shadow.camera.left = -18;
key.shadow.camera.right = 18;
key.shadow.camera.top = 18;
key.shadow.camera.bottom = -18;
key.shadow.bias = -0.00035;
key.shadow.normalBias = 0.035;
scene.add(key);

const rim = new THREE.DirectionalLight(0xff3f81, 8.2);
rim.position.set(8, 5.5, -8);
scene.add(rim);

const crossFill = new THREE.DirectionalLight(0x8fbfff, 2.4);
crossFill.position.set(6, 4.8, 8);
scene.add(crossFill);

const heroSpot = new THREE.SpotLight(0xe8f7ff, 24, 34, Math.PI * 0.17, 0.55, 1.25);
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

const floorGlow = new THREE.PointLight(0x38fff0, 12, 22, 2.15);
floorGlow.position.set(-5, 0.75, 3.2);
scene.add(floorGlow);

const backGlow = new THREE.PointLight(0x7d2dff, 16, 32, 1.9);
backGlow.position.set(0, 2.4, -8);
scene.add(backGlow);

const sideGlow = new THREE.PointLight(0xff315e, 14, 22, 2);
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

const bloomPass = bloom(scenePassEmissive, 0.7, 0.45, 0.2);
const renderPipeline = new RenderPipeline(renderer);
renderPipeline.outputNode = scenePassColor
  .mul(vec4(vec3(aoPass.getTextureNode().r), 1))
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

const state = {
  selectedMeshIds: [],
  animationName: 'Walking',
  arrangement: 'line',
  movement: 'in place',
  count: 7,
  speed: 1,
  animDesync: 0.6,
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
  rimLight: 8.2,
  crossFill: 2.4,
  cameraFill: 10,
  performerKey: 3.2,
  performerFill: 0,
  performerRim: 0,
  performerLightCount: 0,
  exposure: 1.15,
  fog: 0.034,
  ao: 0.85,
  bloom: 0.7,
  quality: 'performance',
  emissiveBoost: 1.3,
  cameraDrift: true,
  floorGrid: true,
  shufflePhase() {
    for (const walker of walkers) walker.seed = Math.random();
  },
  applyQualityPreset() {
    applyQualityPreset(state.quality);
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

  setPreloader(0.1, 'Loading manifest…');
  const response = await fetch(assetUrl('bipeds-manifest.json'));
  if (!response.ok) {
    throw new Error('Missing biped manifest. Run npm run optimize:bipeds.');
  }

  manifest = await response.json();
  if (!manifest.meshes?.length) throw new Error('Biped manifest is empty.');

  buildInspectorControls();

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
  state.selectedMeshIds = manifest.meshes.map((mesh) => mesh.id);

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
  performanceGroup.add(state, 'animDesync', 0, 1, 0.01).name('Anim Desync');
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
  performanceGroup.add(state, 'cameraDrift').name('Camera Drift');
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
  });
  lightingGroup.add(state, 'cameraFill', 0, 24, 0.1).name('Camera Fill').onChange((value) => {
    cameraFill.intensity = value;
  });
  lightingGroup.add(state, 'performerKey', 0, 10, 0.1).name('Performer Key').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  });
  lightingGroup.add(state, 'performerFill', 0, 6, 0.05).name('Performer Fill').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  });
  lightingGroup.add(state, 'performerRim', 0, 8, 0.05).name('Performer Rim').onChange(() => {
    for (const walker of walkers) applyPerformerLighting(walker);
  });
  lightingGroup.add(state, 'performerLightCount', 0, 500).name('Lit Performers').onChange((value) => {
    state.performerLightCount = Math.round(value);
    rebuildWalkers();
  });
  lightingGroup.add(state, 'floorLightBase', 0, 24, 0.1).name('Floor Glow');
  lightingGroup.add(state, 'backLightBase', 0, 28, 0.1).name('Back Glow');
  lightingGroup.add(state, 'ao', 0, 3, 0.01).name('GTAO').onChange((value) => {
    aoPass.scale.value = value;
  });
  lightingGroup.add(state, 'bloom', 0, 3, 0.01).name('Bloom').onChange((value) => {
    bloomPass.strength.value = value;
  });
  lightingGroup.add(state, 'emissiveBoost', 0, 5, 0.01).name('Emissive').onChange(() => {
    for (const batch of crowdBatches) applyEmissiveBoost(batch.source);
  });

  const meshGroup = renderer.inspector.createParameters('VJ Layer / Meshes');
  for (const mesh of manifest.meshes) {
    const meshState = { enabled: true };
    meshGroup.add(meshState, 'enabled').name(mesh.name).onChange((enabled) => {
      if (enabled) {
        state.selectedMeshIds = [...new Set([...state.selectedMeshIds, mesh.id])];
      } else {
        state.selectedMeshIds = state.selectedMeshIds.filter((id) => id !== mesh.id);
      }
      rebuildWalkers();
    });
  }

  const statsGroup = renderer.inspector.createParameters('VJ Layer / Render Stats');
  statsGroup.add(statsState, 'fps').name('FPS').listen();
  statsGroup.add(statsState, 'calls').name('Draw Calls').listen();
  statsGroup.add(statsState, 'triangles').name('Triangles').listen();
  statsGroup.add(statsState, 'lines').name('Lines').listen();
  statsGroup.add(statsState, 'points').name('Points').listen();
  statsGroup.add(statsState, 'geometries').name('Geometries').listen();
  statsGroup.add(statsState, 'textures').name('Textures').listen();
}

function applyQualityPreset(preset) {
  if (preset === 'cinematic') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    key.shadow.mapSize.set(2048, 2048);
    heroSpot.shadow.mapSize.set(2048, 2048);
    aoPass.samples.value = 24;
    aoPass.resolutionScale = 0.75;
    aoPass.scale.value = state.ao = 1.45;
    bloomPass.strength.value = state.bloom = 0.9;
    crossFill.intensity = state.crossFill = 3.2;
    cameraFill.intensity = state.cameraFill = 14;
    state.performerKey = 4.2;
    state.performerFill = 1.2;
    state.performerRim = 2.2;
  } else if (preset === 'balanced') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    key.shadow.mapSize.set(1024, 1024);
    heroSpot.shadow.mapSize.set(1024, 1024);
    aoPass.samples.value = 12;
    aoPass.resolutionScale = 0.6;
    aoPass.scale.value = state.ao = 1.05;
    bloomPass.strength.value = state.bloom = 0.78;
    crossFill.intensity = state.crossFill = 2.7;
    cameraFill.intensity = state.cameraFill = 12;
    state.performerKey = 3.6;
    state.performerFill = 0.45;
    state.performerRim = 0.8;
  } else {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    key.shadow.mapSize.set(1024, 1024);
    heroSpot.shadow.mapSize.set(1024, 1024);
    aoPass.samples.value = 8;
    aoPass.resolutionScale = 0.5;
    aoPass.scale.value = state.ao = 0.85;
    bloomPass.strength.value = state.bloom = 0.7;
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
  return preferred;
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
  if (mesh.animations.includes(name)) return { name, url: mesh.url };

  throw new Error(`${mesh.name} does not include ${name}. Select meshes with a shared animation.`);
}

function createWalker(mesh, animationAsset, index) {
  return {
    index,
    mesh,
    animationAsset,
    batchKey: `${mesh.id}:${animationAsset.name}`,
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
    lightRig: index < state.performerLightCount ? createPerformerLightRig(index) : null
  };
}

async function loadCrowdBatch(mesh, animationAsset, batchWalkers) {
  const gltf = await loadGltf(animationAsset.url);
  const clip = gltf.animations.find((candidate) => candidate.name === animationAsset.name) ?? gltf.animations[0];

  if (!clip) {
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

  const mixer = new THREE.AnimationMixer(source);
  const action = mixer.clipAction(clip);
  action.play();

  const batch = {
    mesh,
    animationAsset,
    source,
    sourceScale: 1,
    sourceOffset: new THREE.Vector3(),
    walkers: batchWalkers,
    mixer,
    action,
    duration: clip.duration,
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
    walker.duration = clip.duration;
  }

  source.updateMatrixWorld(true);
  for (const skinnedMesh of skinnedMeshes) createComputedSkinnedMesh(skinnedMesh, batch);
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
  const vertices = attributeArray(batch.walkers.length * vertexCount * 2, 'vec4');

  const computeNode = Fn(() => {
    const sourceVertex = instanceIndex.mod(uint(vertexCount));
    const meshInstance = instanceIndex.div(uint(vertexCount));
    const sourceOffset = sourceVertex.mul(uint(2));
    const targetOffset = instanceIndex.mul(uint(2));
    const boneOffset = meshInstance.mul(uint(skeletonState.boneCount));
    const indices = skinIndices.element(sourceVertex);
    const weights = skinWeights.element(sourceVertex);
    const skinVertex = bindMatrix.mul(vec4(sourceVertices.element(sourceOffset).xyz, 1));
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

  // Resize the key + hero shadow frusta to cover the whole spread.
  const reach = Math.max(armyBounds.halfX, armyBounds.halfZ) + 6;
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

function updateCrowdBatches(elapsed) {
  for (const batch of crowdBatches) {
    for (const walker of batch.walkers) {
      // animDesync offsets each instance's clip time by its own seed so the
      // crowd doesn't march in lockstep (0 = synced, 1 = fully independent).
      const animOffset = walker.seed * batch.duration * state.animDesync;
      batch.mixer.setTime((elapsed * state.speed + animOffset) % batch.duration);
      batch.source.updateMatrixWorld(true);

      for (const skeletonState of batch.skeletonStates) {
        skeletonState.skeleton.update();
        skeletonState.boneMatrices.array.set(
          skeletonState.skeleton.boneMatrices,
          walker.batchIndex * skeletonState.boneCount * 16
        );
      }

      updateWalkerInstanceMatrix(walker);
    }

    for (const skeletonState of batch.skeletonStates) skeletonState.boneMatrices.needsUpdate = true;
    batch.instanceMatrices.needsUpdate = true;
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
    batch.mixer.stopAllAction();
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

function render(timestamp) {
  timer.update(timestamp);
  const delta = Math.min(timer.getDelta(), 0.05);
  const elapsed = timer.getElapsed();

  for (const walker of walkers) {
    updateCrowdMotion(walker, elapsed);
    updatePerformerLightRig(walker, elapsed);
  }
  updateCrowdBatches(elapsed);

  if (state.cameraDrift) {
    camera.position.x = Math.sin(elapsed * 0.08) * 18;
    camera.position.z = Math.cos(elapsed * 0.08) * 20;
  }

  const bx = armyBounds.centerX;
  const bz = armyBounds.centerZ;
  const hx = armyBounds.halfX;
  const hz = armyBounds.halfZ;
  const coverage = armyBounds.radius + 10;

  floorGlow.intensity = state.floorLightBase + Math.sin(elapsed * 1.7) * state.floorLightBase * 0.16;
  floorGlow.position.x = bx + Math.sin(elapsed * 0.55) * hx;
  floorGlow.position.z = bz + Math.cos(elapsed * 0.43) * hz;
  floorGlow.distance = coverage;
  backGlow.position.x = bx + Math.sin(elapsed * 0.4) * hx;
  backGlow.position.z = bz - hz - 2;
  backGlow.intensity = state.backLightBase;
  backGlow.distance = coverage * 1.25;
  sideGlow.position.x = bx + Math.cos(elapsed * 0.31) * hx;
  sideGlow.position.z = bz + Math.sin(elapsed * 0.29) * hz;
  sideGlow.distance = coverage;
  heroSpot.position.x = bx + Math.sin(elapsed * 0.18) * hx;
  heroSpot.position.y = 9 + armyBounds.radius * 0.35;
  heroSpot.position.z = bz + hz + 6;
  heroSpot.target.position.x = bx + Math.sin(elapsed * 0.22) * hx * 0.5;
  heroSpot.target.position.z = bz;
  heroSpot.distance = Math.max(34, coverage * 2);
  heroSpot.angle = Math.min(Math.PI * 0.33, Math.PI * 0.17 + armyBounds.radius * 0.012);
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
