import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Khi chạy `npm run dev`, các yêu cầu tới FastAPI (cổng 8000) được chuyển tiếp để không cần cấu hình CORS.
// Đổi địa chỉ backend bằng biến môi trường VITE_BACKEND_URL (xem .env.example).
const backend = process.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/health': backend,
      '/openapi.json': backend,
    },
  },
  build: {
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: (id: string) => (id.includes('recharts') || id.includes('d3-') ? 'bieu-do' : undefined),
      },
    },
  },
})
