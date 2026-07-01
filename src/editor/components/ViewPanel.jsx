// View toggles (T32): wireframe / crowd / mask-cloud overlays, texture-blend alpha,
// camera presets.
import { actions } from '../editor-store.js';
import { Section, Slider, Btn } from './primitives.jsx';

export default function ViewPanel({ s }) {
  return (
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
  );
}
