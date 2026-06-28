# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WebGPU "VJ layer" that renders a crowd of GPU-skinned biped characters (Meshy AI GLB exports) with live-tweakable arrangement, motion, and lighting. Three.js r0.185 + TSL (Three.js Shading Language) + Vite. Pure ESM, no framework. WebGPU only — there is no WebGL fallback, and `renderer.init()` must resolve before anything renders.

## Commands

```bash
npm run dev              # vite dev server on 127.0.0.1
npm run build            # vite production build -> dist/
npm run preview          # serve the built dist/
npm run prepare:bipeds   # regenerate public/bipeds/ + manifest from assets/bipeds/*.zip
```

No test runner, linter, or typechecker is configured. There is no IDE UI: all runtime controls live in the three.js **Inspector** panels (`renderer.inspector`), not custom DOM. The only DOM is the canvas + a status line.

## Asset pipeline

The Meshy exports are wildly redundant: every `_Animation_<name>_withSkin.glb`
re-embeds the **entire** model + 4K textures (only the clip differs), plus an
unused `_Character_output.glb`. Two-stage pipeline collapses that:

1. `npm run prepare:bipeds` (`scripts/prepare-bipeds.mjs`) — unzips
   `assets/bipeds/*.zip` (Meshy source) into `public/bipeds/` via the system
   `unzip`. Local-only step; needs the zips.
2. `npm run optimize:bipeds` (`scripts/optimize-bipeds.mjs`) — for each
   character merges the **Walking + Running** GLBs (only those two — every
   character has them, so animation selection never fails) into ONE compact
   `public/models/<id>.glb`: a single mesh/skeleton/texture set + both clips.
   Uses `gltf-transform` (`mergeDocuments` → repoint each clip's channels onto
   the base rig by bone name → dispose the duplicate rig/material/texture →
   `prune`/`dedup` → fold to one buffer). ~3× smaller (688M → ~206M). Rewrites
   `public/bipeds-manifest.json`.

**`public/models/<id>.glb` + the manifest are the shipped, committed artifacts.**
`assets/*.zip` and `public/bipeds/` are gitignored, large, regenerable, and
**never deployed** (3 zips exceed GitHub's 100 MB limit — keep them out of git
history / use a release or LFS if they must be versioned).

Manifest shape: `{ animations: ["Walking","Running"], meshes: [{ id, name, url:
"models/<id>.glb", animations: ["Walking","Running"] }] }`. One GLB per
character; the clip is chosen by name at load time. The app throws if the
manifest is missing or empty.

## Deploy (GitHub Pages)

`.github/workflows/deploy.yml` builds on push to `main` and publishes `dist/`.
`vite.config.js` reads `base` from `VITE_BASE`; the workflow sets it to
`/<repo>/` (project-site subpath). In code, **all runtime asset URLs go through
`assetUrl()`** (prefixes `import.meta.env.BASE_URL`) so the manifest + GLB fetches
resolve under the subpath — never hardcode a leading-`/` asset path. Enable Pages
with Source = GitHub Actions. Deployed site ≈ 206 MB (under the 1 GB Pages limit).

## Architecture — everything is in `src/main.js`

One ~1080-line module. Flow: `init()` → fetch manifest → `buildInspectorControls()` → `rebuildWalkers()` → `renderer.setAnimationLoop(render)`.

**Core GPU technique — per-instance skinning via compute (the whole point of the project).** Modeled on `docs/webgpu_skinning_instancing_individual.html` (the upstream three.js example — read it before touching the compute path). Standard instancing can't give each instance its own animation phase; this does:

- **Walkers** are logical agents (position, rotation, per-instance random seeds, optional light rig). **Batches** (`crowdBatches`) group walkers sharing a `batchKey` = `meshId:animationName`.
- For each batch: every instance's bone matrices are packed into a `StorageBufferAttribute` (`boneMatrices`, sized `walkers × boneCount × 16`). A TSL compute node (`createComputedSkinnedMesh`) skins **all instances' vertices in one dispatch** into a `vertices` `attributeArray`, indexing by `instanceIndex / vertexCount` and `instanceIndex % vertexCount`.
- A single `THREE.InstancedMesh(geometry, material, walkers.length)` (identity instance matrices — transforms are baked into the computed buffer) then draws the whole batch, reading skinned positions/normals back through `positionNode` / `normalNode`. The original source meshes are hidden (`-gpu-instances` suffix marks the live draw mesh). **Must be `InstancedMesh`, not a plain `Mesh` with `.count`** — see gotcha below.
- Per frame, `updateCrowdBatches()` advances each walker's mixer to its own `(elapsed*speed + seed*duration*animDesync) % duration`, copies skeleton bone matrices into the storage buffer, updates the instance matrix, then calls `renderer.compute(computeNode)`.
- Per-instance individuality (the technique's showcase) is driven entirely from the instance transform + clip time, never the shared skin: `animDesync` (random clip-time offset), `sizeVariance` (random uniform scale), `proportionVariance` (random non-uniform x/y/z) — applied in `updateWalkerInstanceMatrix`; the floor recenter is scaled too so feet stay grounded under non-uniform scale.

**Render pipeline** (`RenderPipeline`, not the default render): one MRT `pass` emits `output` / `normal` / `emissive`; GTAO (`ao`) and `bloom` TSL nodes composite into `renderPipeline.outputNode`. Quality presets (`performance` / `balanced` / `cinematic`) scale pixel ratio, shadow map size, AO samples, bloom.

**`state` object** holds all live params and is the bridge to Inspector controls. `onChange` handlers decide cost: cheap params (speed, light intensities) mutate in place; structural ones (`count`, `animationName`, mesh selection, `performerLightCount`) call `rebuildWalkers()`; layout ones (`arrangement`, `spacing`, `scale`, …) call `arrangeWalkers()`. `rebuildId` guards against overlapping async rebuilds.

**Normalization:** each source model is measured once (`normalizeModel`) to a target height of 1.8 and recentered on the floor; the per-instance transform applies this via `normalizationMatrix` so mixed-scale GLBs line up.

## Conventions / gotchas

- `selectAnimation` **throws** if a selected mesh lacks the chosen animation — by design (see the no-fallbacks rule below). Selecting meshes with no shared animation is a user error, not something to paper over.
- `three` is `"latest"` in package.json but pinned to **0.185.0** via the lockfile. TSL/compute APIs move fast across versions; check the installed version before assuming an API exists.
- Frustum culling is disabled on instanced meshes (one giant bounding volume would cull wrong).
- **Use `InstancedMesh`, never `Mesh` + `.count`, for the GPU-skinned batches.** The renderer only gives an object its own pipeline bindings when `isInstancedMesh || count > 1` (see `RenderObject.js` cache key). A plain `Mesh` with `count === 1` shares bindings with every other single-instance batch → they all collapse onto one buffer and render the same bind-pose at the origin (looks like one mesh wearing a collage of every texture). It only "works" once batches reach ≥2 instances. `InstancedMesh` binds per-object and keeps `instanceIndex` active even for one instance.
- Geometry stays **indexed**. The render reads the computed buffer by `instanceIndex * vertexCount + vertexIndex`; with correct per-object bindings, `vertexIndex` dereferences the source vertex id, so indexing is 1:1 (no `toNonIndexed()` needed — it would ~3-4× the vertex/compute cost).
- Per-instance world transforms are baked into the computed vertices buffer, so the `InstancedMesh.instanceMatrix` stays identity (otherwise it double-transforms).
- **Scaling is fragment/light-bound, not draw-call-bound.** Each batch is one instanced draw, so a 65-strong army is still ~17 draw calls/frame and tens of millions of tris run at 100 fps. What tanks FPS is forward lighting: every active light is evaluated per fragment. The per-performer light rigs (`performerLightCount`, 1–3 lights each) do **not** scale — they default to `0`. Light the army with the handful of scene lights + emissive glow; reserve performer rigs for small hero scenes. (`renderer.info.render.calls` is **cumulative** under `RenderPipeline`, not per-frame — don't read it as a per-frame count.)
- **Emissive glow comes from the GLB's `emissiveMap`** (neon veins/caps), with `emissive` factor white. `applyEmissiveBoost` must preserve that map and only scale `emissiveIntensity` — deriving emissive from base color (the old heuristic) zeroed it and killed all glow + bloom. The MRT `emissive` channel feeds `bloomPass`, so no emissive output = no bloom.
- `cameraFill` is a camera-following spotlight (the "light what you're looking at" fill). It's the lever for under-lit fronts, since the strong `key`/`rim` come from behind. Accent lights + key/hero shadow frusta are fanned across the army each frame via `armyBounds` / `updateArmyBounds`.
- `hashRandom` is a deterministic sin-hash, not `Math.random` — arrangements are reproducible across rebuilds; `shufflePhase` is the only intentional randomness.
- Project-specific rule (also global): do **not** add convenience fallbacks for core functionality. Fix the real cause or throw a clear error.
