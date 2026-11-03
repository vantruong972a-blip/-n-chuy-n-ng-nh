import type { CarListing, RecommendFilters, RecommendResponse } from '../types/api'
import { hasEndpoint } from './capabilityService'
import { request } from './http'

/** Backend hiện CHƯA có API gợi ý xe / chi tiết xe. Các hàm dưới đây chỉ gọi khi endpoint thực sự xuất hiện trong /openapi.json. */
export class NotIntegratedError extends Error {
  constructor(what: string) {
    super(`${what} chưa được tích hợp ở máy chủ.`)
  }
}

export const isRecommendationAvailable = () => hasEndpoint('post', '/api/recommend')
export const isCarDetailAvailable = () => hasEndpoint('get', '/api/cars/')

/** Hợp đồng đề xuất: POST /api/recommend  { ...bộ lọc, method, limit }  ->  { items: CarListing[], method } */
export async function recommendCars(filters: RecommendFilters): Promise<RecommendResponse> {
  if (!(await isRecommendationAvailable())) throw new NotIntegratedError('Chức năng gợi ý xe')
  return request<RecommendResponse>('/api/recommend', { method: 'POST', body: JSON.stringify(filters) })
}

/** Hợp đồng đề xuất: GET /api/cars/{id} -> CarListing */
export async function getCarDetail(id: string): Promise<CarListing> {
  if (!(await isCarDetailAvailable())) throw new NotIntegratedError('Chức năng xem chi tiết xe')
  return request<CarListing>(`/api/cars/${encodeURIComponent(id)}`)
}
