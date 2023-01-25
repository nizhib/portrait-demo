/* eslint-disable import/no-extraneous-dependencies,import/no-unresolved */
import { defineConfig } from 'vite';
// import eslint from 'vite-plugin-eslint';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
