// Merges each character's per-animation GLBs (which each re-embed the full model
// + textures) into ONE compact GLB per character: a single mesh/skeleton/texture
// set plus every kept animation clip. Cuts deployed asset size ~3x and removes
// the duplicate-geometry / unused-Character files.
//
// Reads the unzipped source under public/bipeds/, writes public/models/<id>.glb
// and rewrites public/bipeds-manifest.json. Run: npm run optimize:bipeds
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// gltf-transform is NOT a project dependency — it pulls native deps (sharp) that
// break cross-platform CI lockfiles, and it's only needed for this one-off asset
// step. Install it on demand:  npm i -D @gltf-transform/core @gltf-transform/functions
let NodeIO;
let mergeDocuments;
let prune;
let dedup;
try {
  ({ NodeIO } = await import('@gltf-transform/core'));
  ({ mergeDocuments, prune, dedup } = await import('@gltf-transform/functions'));
} catch {
  console.error('Missing optimizer deps. Run:\n  npm i -D @gltf-transform/core @gltf-transform/functions');
  process.exit(1);
}

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceDir = join(root, 'public', 'bipeds');
const outDir = join(root, 'public', 'models');
const manifestPath = join(root, 'public', 'bipeds-manifest.json');

// Only these animations are kept — every character has them, so animation
// selection never fails on a missing clip.
const KEEP = ['Walking', 'Running'];

function cleanName(folder) {
  return folder.replace(/^Meshy_AI_/, '').replace(/_biped$/, '').replaceAll('_', ' ');
}

function animationName(fileName) {
  const match = fileName.match(/_Animation_(.+)_withSkin\.glb$/);
  return match ? match[1].replaceAll('_', ' ') : null;
}

// Transplant one animation GLB's clip onto `base`, retargeting its channels to
// base's same-named bones, then drop everything else the merge pulled in.
function addClip(io, base, root, baseNodes, clipName, sourceDoc) {
  const keep = {
    node: new Set(root.listNodes()),
    skin: new Set(root.listSkins()),
    mesh: new Set(root.listMeshes()),
    scene: new Set(root.listScenes()),
    material: new Set(root.listMaterials()),
    texture: new Set(root.listTextures())
  };

  mergeDocuments(base, sourceDoc);

  const clip = root.listAnimations().at(-1);
  clip.setName(clipName);
  for (const channel of clip.listChannels()) {
    const target = channel.getTargetNode();
    const baseNode = baseNodes.get(target.getName());
    if (baseNode && baseNode !== target) channel.setTargetNode(baseNode);
  }

  // The merge duplicated the whole rig + textures; the clip now drives base's
  // bones, so discard the duplicates and let prune/dedup reclaim the bytes.
  for (const scene of root.listScenes()) if (!keep.scene.has(scene)) scene.dispose();
  for (const mesh of root.listMeshes()) if (!keep.mesh.has(mesh)) mesh.dispose();
  for (const skin of root.listSkins()) if (!keep.skin.has(skin)) skin.dispose();
  for (const node of root.listNodes()) if (!keep.node.has(node)) node.dispose();
  for (const material of root.listMaterials()) if (!keep.material.has(material)) material.dispose();
  for (const texture of root.listTextures()) if (!keep.texture.has(texture)) texture.dispose();
}

const io = new NodeIO();
await rm(outDir, { recursive: true, force: true });
await mkdir(outDir, { recursive: true });

const folders = (await readdir(sourceDir, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();

const meshes = [];
for (const folder of folders) {
  const files = await readdir(join(sourceDir, folder));
  const byAnimation = new Map();
  for (const file of files) {
    const name = animationName(file);
    if (name && KEEP.includes(name)) byAnimation.set(name, join(sourceDir, folder, file));
  }

  const present = KEEP.filter((name) => byAnimation.has(name));
  if (present.length === 0) {
    console.warn(`skip ${folder} — no Walking/Running`);
    continue;
  }

  const base = await io.read(byAnimation.get(present[0]));
  const baseRoot = base.getRoot();
  baseRoot.listAnimations()[0].setName(present[0]);
  const baseNodes = new Map(baseRoot.listNodes().map((node) => [node.getName(), node]));

  for (let i = 1; i < present.length; i += 1) {
    addClip(io, base, baseRoot, baseNodes, present[i], await io.read(byAnimation.get(present[i])));
  }

  baseRoot.setDefaultScene(baseRoot.listScenes()[0]);
  await prune(base);
  await dedup(base);

  // GLB allows a single buffer — fold every accessor into the first.
  const buffers = baseRoot.listBuffers();
  const main = buffers[0];
  for (const accessor of baseRoot.listAccessors()) accessor.setBuffer(main);
  for (let i = 1; i < buffers.length; i += 1) buffers[i].dispose();

  await io.write(join(outDir, `${folder}.glb`), base);
  meshes.push({ id: folder, name: cleanName(folder), url: `models/${folder}.glb`, animations: present });
  console.log(`built ${folder} (${present.join(' + ')})`);
}

if (meshes.length === 0) throw new Error(`No characters built from ${sourceDir}`);

const manifest = { generatedAt: new Date().toISOString(), animations: KEEP, meshes };
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`done — ${meshes.length} characters -> public/models/`);
