// Pure per-walker animation phase (clip-time offset). No three/DOM deps so it
// unit-tests in node. Choreography = how each instance's playhead is offset from
// the shared clip: lockstep, desynced, or spatial patterns (ripple/wave). Output
// must be reproducible — only deterministic inputs (seed, index, basePosition).

export const CHOREOGRAPHIES = ['synced', 'desync', 'ripple', 'wave', 'index', 'random'];

// Deterministic sin-hash, NOT Math.random — mirrors main.js hashRandom so
// arrangements/phases reproduce across rebuilds (V7).
export function hashRandom(value) {
  const x = Math.sin(value * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

// Clip-time offset (seconds) for one walker. Added to (elapsed*speed) by caller.
// walker: { seed, index, basePosition:{x,z} }
// opts:   { choreography, choreoDelay, count, bounds:{centerX,centerZ,halfX,halfZ,radius} }
export function computeWalkerPhase(walker, duration, opts) {
  const delay = opts.choreoDelay;
  if (delay <= 0) return 0;

  const span = duration * delay;

  switch (opts.choreography) {
    case 'synced':
      // Whole crowd shares one playhead — instant, all at once.
      return 0;

    case 'ripple': {
      // Radial delay from crowd center outward (inside → outside). 0 at center,
      // → span at the rim. bounds.radius is the padded half-diagonal.
      const b = opts.bounds;
      const r = Math.hypot(walker.basePosition.x - b.centerX, walker.basePosition.z - b.centerZ);
      const t = b.radius > 1e-6 ? Math.min(r / b.radius, 1) : 0;
      return t * span;
    }

    case 'wave': {
      // Linear delay along +x across the crowd width — a marching wavefront.
      const b = opts.bounds;
      const t = b.halfX > 1e-6 ? (walker.basePosition.x - b.centerX) / b.halfX * 0.5 + 0.5 : 0;
      return clamp01(t) * span;
    }

    case 'index': {
      // Sequential delay by spawn order — cascade down the roster.
      const denom = Math.max(opts.count - 1, 1);
      return (walker.index / denom) * span;
    }

    case 'random':
      // Per-walker pseudo-random offset, distinct stream from seed.
      return hashRandom(walker.index * 31 + 7) * span;

    case 'desync':
    default:
      return walker.seed * span;
  }
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
