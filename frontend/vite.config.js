import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      "/predict": {
        target: "http://localhost:8000",
        secure: false,
      },
    },
  },
  plugins: [react()],
});