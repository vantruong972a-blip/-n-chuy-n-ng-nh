import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../services/http'

export function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e
  return new ApiError(-1, [e instanceof Error ? e.message : 'Đã xảy ra lỗi không xác định.'])
}

interface Settled<T> {
  key: string
  token: number
  data?: T
  error?: ApiError
}

/** Tải dữ liệu bất đồng bộ một lần theo `key` (đổi key hoặc gọi reload() để tải lại). */
export function useAsync<T>(fn: () => Promise<T>, key = 'default') {
  const [token, setToken] = useState(0)
  const [settled, setSettled] = useState<Settled<T> | null>(null)
  const fnRef = useRef(fn)
  useEffect(() => {
    fnRef.current = fn
  })
  useEffect(() => {
    let alive = true
    fnRef
      .current()
      .then((data) => alive && setSettled({ key, token, data }))
      .catch((e) => alive && setSettled({ key, token, error: toApiError(e) }))
    return () => {
      alive = false
    }
  }, [key, token])
  const current = settled && settled.key === key && settled.token === token ? settled : null
  const reload = useCallback(() => setToken((t) => t + 1), [])
  return { data: current?.data, error: current?.error, loading: current === null, reload }
}
