import { defineConfig } from 'vite';

export default defineConfig({
  base: '/local-recommender-zkp/',
  root: '.',
  publicDir: 'public',
  server: {
    port: 5173,
    open: true
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
