// Face-mask editor UI (T30 / I.editorUI). React + Tailwind overlay on the VJ app.
// Reads the shared editor store; every mutation goes through `actions.*` (registered by
// main.js) so React never touches three directly. The root is pointer-events-none so
// clicks in the empty center reach the canvas (painting); panels re-enable pointers.
import { useSyncExternalStore, useRef } from 'react';
import { editorState, actions, subscribe, getVersion } from './editor-store.js';
import { FACE_REGIONS, EXPR_KEYS, DEFORM_TYPES } from '../pose/face-mask.js';

const REGION_COLOR = {
  jaw: '#ff5c5c', lowerLip: '#ff9f43', mouthCorner: '#feca57',
  upperLidL: '#54a0ff', upperLidR: '#5f27cd', browL: '#1dd1a1', browR: '#00d2d3'
};
const EXPR = [
  ['jawOpen', 'Jaw'], ['smile', 'Smile'], ['pucker', 'Pucker'],
  ['blinkL', 'Blink L'], ['blinkR', 'Blink R'], ['browL', 'Brow L'], ['browR', 'Brow R']
];
const DIRS = [['+X', [1, 0, 0]], ['-X', [-1, 0, 0]], ['+Y', [0, 1, 0]], ['-Y', [0, -1, 0]], ['+Z', [0, 0, 1]], ['-Z', [0, 0, -1]]];
const sameDir = (a, b) => a && b && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

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

// Inline label · slider · value. min-w-0 lets the range shrink instead of overflowing.
function Slider({ label, value, min, max, step, onChange }) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-200 mb-1.5 min-w-0">
      <span className="w-14 shrink-0 text-slate-400 truncate">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 min-w-0 accent-cyan-400" />
      <span className="w-9 shrink-0 text-right tabular-nums text-slate-400">{(+value).toFixed(2)}</span>
    </label>
  );
}

// Compact stacked control for the test-drive bar (label+value on top, full-width slider).
function MiniSlider({ label, value, onChange }) {
  return (
    <div className="min-w-0">
      <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
        <span className="truncate">{label}</span><span className="tabular-nums">{(+value).toFixed(2)}</span>
      </div>
      <input type="range" min={0} max={1} step={0.02} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
    </div>
  );
}

function Btn({ active, disabled, onClick, children, title }) {
  return (
    <button title={title} disabled={disabled} onClick={onClick}
      className={`px-2.5 py-1 rounded text-xs font-medium transition-colors disabled:opacity-30 ${active
        ? 'bg-cyan-500 text-black' : 'bg-white/5 text-slate-200 hover:bg-white/10'}`}>
      {children}
    </button>
  );
}

export default function EditorUI() {
  const s = useEditor();
  const fileRef = useRef(null);
  const rc = s.regionConfig || { driver: 0, type: 0, amount: 0, dir: [0, -1, 0], mirrorX: false };
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
          className="bg-white/5 text-xs rounded px-2 py-1 border border-white/10 max-w-[9rem]">
          {s.models.map((m) => <option key={m.index} value={m.index}>{m.name}</option>)}
        </select>
        <div className="flex gap-1">
          <Btn disabled={!s.canUndo} onClick={() => actions.undo()} title="Undo (⌘/Ctrl+Z, mouse back)">↶</Btn>
          <Btn disabled={!s.canRedo} onClick={() => actions.redo()} title="Redo (⌘/Ctrl+Shift+Z, mouse fwd)">↷</Btn>
        </div>
        <Btn onClick={() => actions.save()} title="Download .bin (commit into public/models/)">Save</Btn>
        <Btn onClick={() => fileRef.current?.click()} title="Load a .bin override">Load</Btn>
        <input ref={fileRef} type="file" accept=".bin" className="hidden" onChange={onLoadFile} />
        <Btn onClick={() => actions.setOpen(false)} title="Close editor">✕</Btn>
      </div>

      {/* Left column */}
      <div className="absolute top-16 left-3 w-64 pointer-events-auto max-h-[calc(100vh-6rem)]
        overflow-y-auto overflow-x-hidden bg-black/70 backdrop-blur rounded-lg p-3 border border-white/10">
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
                className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] min-w-0 transition-all ${s.region === r
                  ? 'bg-white/15 ring-1 ring-cyan-400' : 'bg-white/5 hover:bg-white/10'}`}>
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: REGION_COLOR[r] }} />
                <span className="truncate">{r}</span>
              </button>
            ))}
          </div>
        </Section>

        <Section title={`Marker: ${s.region}`}>
          <label className="flex items-center gap-2 text-xs text-slate-300 mb-1.5 min-w-0">
            <span className="w-14 shrink-0 text-slate-400">Driver</span>
            <select value={rc.driver} onChange={(e) => actions.setRegionConfig({ driver: +e.target.value })}
              className="flex-1 min-w-0 bg-white/5 rounded px-1.5 py-0.5 border border-white/10">
              {EXPR_KEYS.map((k, i) => <option key={k} value={i}>{k}</option>)}
            </select>
          </label>
          <div className="flex gap-1.5 mb-1.5">
            {DEFORM_TYPES.map((t, i) => (
              <Btn key={t} active={rc.type === i} onClick={() => actions.setRegionConfig({ type: i })}>{t}</Btn>
            ))}
          </div>
          <Slider label="Amount" value={rc.amount} min={0} max={1.2} step={0.01} onChange={(v) => actions.setRegionConfig({ amount: v })} />
          {rc.type === 0 ? (
            <>
              <div className="text-[10px] text-slate-400 mt-1 mb-1">Direction</div>
              <div className="grid grid-cols-6 gap-1">
                {DIRS.map(([lbl, d]) => (
                  <Btn key={lbl} active={sameDir(rc.dir, d)} onClick={() => actions.setRegionConfig({ dir: d })}>{lbl}</Btn>
                ))}
              </div>
              <label className="flex items-center gap-2 text-xs text-slate-300 mt-1.5">
                <input type="checkbox" checked={rc.mirrorX} onChange={(e) => actions.setRegionConfig({ mirrorX: e.target.checked })}
                  className="accent-cyan-400" />
                Mirror X by side (spread)
              </label>
            </>
          ) : (
            <div className="text-[10px] text-cyan-300/80 mt-1 leading-relaxed">
              Hinge: use the <b>hinge</b> tool → click the head for origin, then a 2nd point for the axis. Amount = radians.
            </div>
          )}
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
            <Btn onClick={() => actions.reseed(true)} title="Flip assumed forward axis">Flip Forward</Btn>
            <Btn onClick={() => actions.clearRegion()} title="Zero the current region">Clear</Btn>
          </div>
        </Section>

        <Section title="View">
          <div className="flex flex-wrap gap-1.5">
            <Btn active={s.overlays.wireframe} onClick={() => actions.setOverlay('wireframe', !s.overlays.wireframe)}>Wire</Btn>
            <Btn active={s.overlays.crowd} onClick={() => actions.setOverlay('crowd', !s.overlays.crowd)}>Crowd</Btn>
            <Btn active={s.overlays.maskCloud} onClick={() => actions.setOverlay('maskCloud', !s.overlays.maskCloud)}>Mask pts</Btn>
          </div>
          <div className="mt-2"><Slider label="Texture" value={s.overlays.texAlpha ?? 0} min={0} max={1} step={0.05} onChange={actions.setTexAlpha} /></div>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {['front', 'left', 'right', '3q'].map((v) => (
              <Btn key={v} onClick={() => actions.frameView(v)} title={`Camera: ${v}`}>{v}</Btn>
            ))}
          </div>
        </Section>
      </div>

      {/* Bottom: expression test-drive */}
      <div className="absolute bottom-3 left-3 right-3 md:left-1/2 md:right-auto md:-translate-x-1/2 md:w-[680px]
        pointer-events-auto bg-black/70 backdrop-blur rounded-lg px-4 py-3 border border-white/10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300/70">
            Test-drive (live deform)</span>
          <Btn onClick={() => actions.resetExpr()}>Reset</Btn>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-5 gap-y-2">
          {EXPR.map(([k, label]) => (
            <MiniSlider key={k} label={label} value={s.expr[k]} onChange={(v) => actions.setExpr({ ...s.expr, [k]: v })} />
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
