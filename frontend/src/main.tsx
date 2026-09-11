import { createRoot } from 'react-dom/client'
// Orbitron Black — the studio overlay's "by MakerMods" chip. Self-hosted so
// the UI renders identically offline.
import '@fontsource/orbitron/900.css'
import App from './App.tsx'
import './index.css'
// Side-effect import: boots i18next before the first render so no
// component can call t() against an uninitialized instance.
import '@/i18n'

createRoot(document.getElementById("root")!).render(<App />);
