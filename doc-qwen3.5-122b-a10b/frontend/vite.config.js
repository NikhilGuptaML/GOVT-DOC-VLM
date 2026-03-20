import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/process': 'http://localhost:8002',
      '/images': 'http://localhost:8002',
    }
  }
})
