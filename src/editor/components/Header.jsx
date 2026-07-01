// Editor header (T32): title, model select, undo/redo, save/load .bin, close.
import { useRef } from 'react';
import { actions } from '../editor-store.js';
import { HOTKEYS } from '../keymap.js';
import { Btn } from './primitives.jsx';

export default function Header({ s }) {
  const fileRef = useRef(null);
  const onLoadFile = (e) => {
    const f = e.target.files?.[0]; if (!f) return;
    f.arrayBuffer().then((buf) => actions.loadMask?.(buf));
    e.target.value = '';
  };
  return (
    <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-auto
      bg-black/70 backdrop-blur rounded-lg px-3 py-2 border border-white/10">
      <span className="text-sm font-bold text-cyan-300">Face Mask Editor</span>
      <select value={s.modelIndex} onChange={(e) => actions.setModel(+e.target.value)}
        className="bg-white/5 text-xs rounded px-2 py-1 border border-white/10 max-w-[9rem]">
        {s.models.map((m) => <option key={m.index} value={m.index}>{m.name}</option>)}
      </select>
      <div className="flex gap-1">
        <Btn disabled={!s.canUndo} onClick={() => actions.undo()} title={`Undo (${HOTKEYS.undo.label}, mouse back)`}>↶</Btn>
        <Btn disabled={!s.canRedo} onClick={() => actions.redo()} title={`Redo (${HOTKEYS.redo.label}, mouse fwd)`}>↷</Btn>
      </div>
      <Btn onClick={() => actions.save()} kbd={HOTKEYS.save.label} title="Download .bin (commit into public/models/)">Save</Btn>
      <Btn onClick={() => fileRef.current?.click()} title="Load a .bin override">Load</Btn>
      <input ref={fileRef} type="file" accept=".bin" className="hidden" onChange={onLoadFile} />
      <Btn onClick={() => actions.setOpen(false)} title="Close editor">✕</Btn>
    </div>
  );
}
