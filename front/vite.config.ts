import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  publicDir: 'static',
});
