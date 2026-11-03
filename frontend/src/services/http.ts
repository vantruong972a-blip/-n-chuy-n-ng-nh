/** Lớp gọi HTTP dùng chung. Mọi lỗi được chuyển thành ApiError với thông báo tiếng Việt. */

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? ''

export class ApiError extends Error {
  status: number
  messages: string[]
  constructor(status: number, messages: string[]) {
    super(messages.join(' '))
    this.status = status
    this.messages = messages
  }
  get isNetwork() {
    return this.status === 0
  }
}

function normalizeDetail(detail: unknown): string[] {
  if (typeof detail === 'string') return [detail]
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      if (typeof d === 'string') return d
      if (d && typeof d === 'object' && 'msg' in d) {
        const loc = Array.isArray((d as { loc?: unknown[] }).loc) ? (d as { loc: unknown[] }).loc.slice(1).join('.') : ''
        return `${loc ? loc + ': ' : ''}${String((d as { msg: unknown }).msg)}`
      }
      return JSON.stringify(d)
    })
  }
  return []
}

export async function request<T>(path: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<T> {
  const { timeoutMs = 25000, ...rest } = init
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  let res: Response
  try {
    res = await fetch(BASE + path, {
      ...rest,
      signal: ctrl.signal,
      headers: { Accept: 'application/json', ...(rest.body ? { 'Content-Type': 'application/json' } : {}), ...rest.headers },
    })
  } catch (e) {
    const timedOut = e instanceof DOMException && e.name === 'AbortError'
    throw new ApiError(0, [
      timedOut
        ? 'Máy chủ phản hồi quá lâu. Vui lòng thử lại sau ít phút.'
        : 'Không kết nối được máy chủ. Hãy kiểm tra máy chủ FastAPI đã chạy chưa (cổng 8000) rồi thử lại.',
    ])
  } finally {
    clearTimeout(timer)
  }
  let body: unknown
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    const detail = normalizeDetail((body as { detail?: unknown } | null)?.detail)
    const fallback =
      res.status === 404
        ? 'Không tìm thấy nội dung được yêu cầu.'
        : res.status >= 500
          ? 'Máy chủ gặp lỗi khi xử lý yêu cầu. Vui lòng thử lại sau.'
          : 'Yêu cầu không hợp lệ.'
    throw new ApiError(res.status, detail.length ? detail : [fallback])
  }
  return body as T
}
