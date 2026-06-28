import { defineConfig } from 'vite';

// `base` is the path the site is served from. GitHub Pages project sites live at
// https://<user>.github.io/<repo>/, so CI passes VITE_BASE=/<repo>/. Locally it
// defaults to '/'. import.meta.env.BASE_URL in the app resolves asset URLs.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/'
});
