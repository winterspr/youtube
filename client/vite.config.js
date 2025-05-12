import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/recommend': {
        target: 'http://127.0.0.1:5000', // Địa chỉ backend Flask
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/recommend/, ''),
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        //rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
