// Tool select (T32): Move (orbit) vs Brush (paint), + brush sub-modes.
import { actions } from '../editor-store.js';
import { Section, Btn } from './primitives.jsx';

export default function ToolPanel({ s }) {
  return (
    <Section title="Tool">
      <div className="flex gap-1.5 mb-1.5">
        <Btn active={s.tool === 'camera'} onClick={() => actions.setTool('camera')} title="Orbit the camera (V)">✋ Move</Btn>
        <Btn active={s.tool === 'brush'} onClick={() => actions.setTool('brush')} title="Paint on the head (B)">🖌 Brush</Btn>
      </div>
      {s.tool === 'brush' ? (
        <div className="flex gap-1.5">
          {['paint', 'erase', 'hinge'].map((m) => (
            <Btn key={m} active={s.mode === m} onClick={() => actions.setMode(m)}>{m}</Btn>
          ))}
        </div>
      ) : <div className="text-[10px] text-slate-400">Drag to orbit · scroll to zoom.</div>}
    </Section>
  );
}
