// Shared editor UI primitives + store hook (T32). Presentational only — no three, no
// store mutation logic beyond the passed onChange. Panels compose these.
import { useSyncExternalStore } from 'react';
import { editorState, subscribe, getVersion } from '../editor-store.js';

// Subscribe React to the mutable store via a monotonic version snapshot.
export function useEditor() {
  useSyncExternalStore(subscribe, getVersion, getVersion);
  return editorState;
}

export function Section({ title, children }) {
  return (
    <div className="mb-3">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300/70 mb-1.5">{title}</div>
      {children}
    </div>
  );
}

// Inline label · slider · value. min-w-0 lets the range shrink instead of overflowing.
export function Slider({ label, value, min, max, step, onChange }) {
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
export function MiniSlider({ label, value, onChange }) {
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

export function Btn({ active, disabled, onClick, children, title }) {
  return (
    <button title={title} disabled={disabled} onClick={onClick}
      className={`px-2.5 py-1 rounded text-xs font-medium transition-colors disabled:opacity-30 ${active
        ? 'bg-cyan-500 text-black' : 'bg-white/5 text-slate-200 hover:bg-white/10'}`}>
      {children}
    </button>
  );
}
