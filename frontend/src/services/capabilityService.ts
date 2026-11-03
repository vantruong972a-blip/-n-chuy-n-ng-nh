import { request } from './http'

interface OpenApi {
  paths: Record<string, Record<string, unknown>>
}

let cache: Promise<OpenApi> | null = null

/** Đọc /openapi.json để biết backend thực sự có endpoint nào (không đoán). */
export function getOpenApi(): Promise<OpenApi> {
  cache ??= request<OpenApi>('/openapi.json').catch((e) => {
    cache = null
    throw e
  })
  return cache
}

export async function hasEndpoint(method: 'get' | 'post', pathPrefix: string): Promise<boolean> {
  const doc = await getOpenApi()
  return Object.entries(doc.paths).some(([p, ops]) => p.startsWith(pathPrefix) && method in ops)
}
