// Face-mask editor UI (T30 / I.editorUI). React + Tailwind overlay on the VJ app.
// Reads the shared editor store; every mutation goes through `actions.*` (registered by
// main.js) so React never touches three directly. The root is pointer-events-none so
// clicks in the empty center reach the canvas (painting); panels re-enable pointers.
import { useSyncExternalStore, useRef } from 'react';
import { editorState, actions, subscribe, getVersion } from './editor-store.js';
import { FACE_REGIONS } from '../pose/face-mask.js';

const REGION_COLOR = {
  jaw: '#ff5c5c', lowerLip: '#ff9f43', mouthCorner: '#feca57',
  upperLidL: '#54a0ff', upperLidR: '#5f27cd', browL: '#1dd1a1', browR: '#00d2d3'
};
const EXPR = [
  ['jawOpen', 'Jaw'], ['smile', 'Smile'], ['pucker', 'Pucker'],
  ['blinkL', 'Blink L'], ['blinkR', 'Blink R'], ['browL', 'Brow L'], ['browR', 'Brow R']
];

function useEditor() {
  useSyncExternalStore(subscribe, getVersion, getVersion);
  return editorState;
}

function Section({ title, children }) {
  return (
    <div className="mb-3">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300/70 mb-1.5">{title}</div>
      {children}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, color }) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-200 mb-1.5">
      <span className="w-16 shrink-0 text-slate-400">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-cyan-400" style={color ? { accentColor: color } : undefined} />
      <span className="w-9 text-right tabular-nums text-slate-400">{(+value).toFixed(2)}</span>
    </label>
  );
}

function Btn({ active, onClick, children, title }) {
  return (
    <button title={title} onClick={onClick}
      className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${active
        ? 'bg-cyan-500 text-black' : 'bg-white/5 text-slate-200 hover:bg-white/10'}`}>
      {children}
    </button>
  );
}

export default function EditorUI() {
  const s = useEditor();
  const fileRef = useRef(null);
  if (!s.open) return null;

  const onLoadFile = (e) => {
    const f = e.target.files?.[0]; if (!f) return;
    f.arrayBuffer().then((buf) => actions.loadMask?.(buf));
    e.target.value = '';
  };

  return (
    <div className="fixed inset-0 z-50 pointer-events-none select-none font-sans text-slate-100">
      {/* Header */}
      <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-auto
        bg-black/70 backdrop-blur rounded-lg px-3 py-2 border border-white/10">
        <span className="text-sm font-bold text-cyan-300">Face Mask Editor</span>
        <select value={s.modelIndex} onChange={(e) => actions.setModel(+e.target.value)}
          className="bg-white/5 text-xs rounded px-2 py-1 border border-white/10">
          {s.models.map((m) => <option key={m.index} value={m.index}>{m.name}</option>)}
        </select>
        <Btn onClick={() => actions.save()} title="Download .bin (commit into public/models/)">Save .bin</Btn>
        <Btn onClick={() => fileRef.current?.click()} title="Load a .bin override">Load</Btn>
        <input ref={fileRef} type="file" accept=".bin" className="hidden" onChange={onLoadFile} />
        <Btn onClick={() => actions.setOpen(false)} title="Close editor">✕</Btn>
      </div>

      {/* Left column: brush + regions + view */}
      <div className="absolute top-16 left-3 w-60 pointer-events-auto max-h-[calc(100vh-6rem)] overflow-y-auto
        bg-black/70 backdrop-blur rounded-lg p-3 border border-white/10">
        <Section title="Tool">
          <div className="flex gap-1.5">
            {['paint', 'erase', 'hinge'].map((m) => (
              <Btn key={m} active={s.mode === m} onClick={() => actions.setMode(m)}>{m}</Btn>
            ))}
          </div>
        </Section>

        <Section title="Region">
          <div className="grid grid-cols-2 gap-1.5">
            {FACE_REGIONS.map((r) => (
              <button key={r} onClick={() => actions.setRegion(r)}
                className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] transition-all ${s.region === r
                  ? 'bg-white/15 ring-1 ring-cyan-400' : 'bg-white/5 hover:bg-white/10'}`}>
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: REGION_COLOR[r] }} />
                <span className="truncate">{r}</span>
              </button>
            ))}
          </div>
        </Section>

        <Section title="Brush">
          <Slider label="Radius" value={s.radius} min={0.005} max={0.3} step={0.005} onChange={actions.setRadius} />
          <Slider label="Strength" value={s.strength} min={0.05} max={1} step={0.05} onChange={actions.setStrength} />
          <label className="flex items-center gap-2 text-xs text-slate-300 mt-1">
            <input type="checkbox" checked={s.symmetric} onChange={(e) => actions.setSymmetric(e.target.checked)}
              className="accent-cyan-400" />
            Symmetric (mirror X)
          </label>
        </Section>

        <Section title="Auto-gen fixes">
          <div className="flex flex-wrap gap-1.5">
            <Btn onClick={() => actions.reseed(false)} title="Re-run auto-gen">Re-seed</Btn>
            <Btn onClick={() => actions.reseed(true)} title="Flip assumed forward axis (fixes back-of-head)">Flip Forward</Btn>
            <Btn onClick={() => actions.clearRegion()} title="Zero the current region">Clear</Btn>
          </div>
        </Section>

        <Section title="View">
          <div className="flex flex-wrap gap-1.5">
            <Btn active={s.overlays.wireframe} onClick={() => actions.setOverlay('wireframe', !s.overlays.wireframe)}>Wire</Btn>
            <Btn active={s.overlays.crowd} onClick={() => actions.setOverlay('crowd', !s.overlays.crowd)}>Crowd</Btn>
            <Btn active={s.overlays.maskCloud} onClick={() => actions.setOverlay('maskCloud', !s.overlays.maskCloud)}>Mask pts</Btn>
          </div>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {['front', 'left', 'right', '3q'].map((v) => (
              <Btn key={v} onClick={() => actions.frameView(v)} title={`Camera: ${v}`}>{v}</Btn>
            ))}
          </div>
        </Section>
      </div>

      {/* Bottom: expression test-drive */}
      <div className="absolute bottom-3 left-3 right-3 md:left-1/2 md:right-auto md:-translate-x-1/2 md:w-[640px]
        pointer-events-auto bg-black/70 backdrop-blur rounded-lg px-4 py-3 border border-white/10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300/70">
            Test-drive (live deform)</span>
          <Btn onClick={() => actions.resetExpr()}>Reset</Btn>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4">
          {EXPR.map(([k, label]) => (
            <Slider key={k} label={label} value={s.expr[k]} min={0} max={1} step={0.02}
              onChange={(v) => actions.setExpr({ ...s.expr, [k]: v })} />
          ))}
        </div>
      </div>

      {/* Status toast */}
      {s.status ? (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 pointer-events-none
          bg-cyan-500/90 text-black text-xs font-medium px-3 py-1.5 rounded-full">{s.status}</div>
      ) : null}
    </div>
  );
}
