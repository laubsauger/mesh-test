// Synthetic inference timing. Runs a session with fixed feeds — zeros input, no
// webcam needed, so the EP probe is measurable headless / over CDP (V23).
export async function benchSession(session, feeds, { runs = 20, warmup = 3 } = {}) {
  for (let i = 0; i < warmup; i += 1) disposeOutputs(await session.run(feeds));

  const times = [];
  for (let i = 0; i < runs; i += 1) {
    const t0 = performance.now();
    disposeOutputs(await session.run(feeds));
    times.push(performance.now() - t0);
  }

  times.sort((a, b) => a - b);
  return {
    runs,
    min: times[0],
    median: times[(times.length / 2) | 0],
    mean: times.reduce((s, t) => s + t, 0) / times.length,
    max: times[times.length - 1]
  };
}

function disposeOutputs(out) {
  for (const k in out) out[k]?.dispose?.();
}
