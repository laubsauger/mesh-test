// Resolve runtime asset URLs against Vite's base so the app works at the dev
// root and under a GitHub Pages project subpath (e.g. /mesh-test/). Never
// hardcode a leading-`/` asset path — always route through assetUrl().
const BASE_URL = import.meta.env.BASE_URL;

export const assetUrl = (path) => `${BASE_URL}${String(path).replace(/^\//, '')}`;
