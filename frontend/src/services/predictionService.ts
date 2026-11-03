import type { HealthResponse, MetaResponse, PredictResponse, PredictValues } from '../types/api'
import { request } from './http'

export const getMeta = () => request<MetaResponse>('/api/meta')

export const getHealth = () => request<HealthResponse>('/health')

/** Gọi POST /api/predict. Giá do backend tính; frontend không tự ước lượng. */
export const predictPrice = (values: PredictValues, model?: string) =>
  request<PredictResponse>('/api/predict', {
    method: 'POST',
    body: JSON.stringify(model ? { values, model } : { values }),
  })

/** Gửi giá thực tế để backend theo dõi sai số (POST /api/feedback). */
export const sendFeedback = (predictionId: string, actualPrice: number) =>
  request<{ status: string }>('/api/feedback', {
    method: 'POST',
    body: JSON.stringify({ prediction_id: predictionId, actual_price: actualPrice }),
  })
