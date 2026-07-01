// Mounts the React editor overlay into the VJ app (T30). Called once from main.js.
// The root div is created here so main.js never touches JSX/DOM UI directly.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import EditorUI from './EditorUI.jsx';
import './editor.css';

export function mountEditor() {
  let el = document.getElementById('face-editor-root');
  if (!el) {
    el = document.createElement('div');
    el.id = 'face-editor-root';
    document.body.appendChild(el);
  }
  createRoot(el).render(<StrictMode><EditorUI /></StrictMode>);
}
