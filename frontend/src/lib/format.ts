const nf = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 0 })
const nf1 = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 })
const nf2 = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 })

export const formatInt = (n: number) => nf.format(n)
export const formatDecimal = (n: number, digits: 1 | 2 = 1) => (digits === 1 ? nf1 : nf2).format(n)

/** 480_000_000 -> "480 triệu"; 1_250_000_000 -> "1,25 tỷ". */
export function formatVnd(v: number): string {
  if (!Number.isFinite(v)) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e9) return `${nf2.format(v / 1e9)} tỷ`
  if (abs >= 1e6) return `${nf.format(v / 1e6)} triệu`
  return `${nf.format(v)} đ`
}

/** 480_000_000 -> "480.000.000 ₫" */
export const formatVndFull = (v: number) => `${nf.format(Math.round(v))} ₫`

/** Giá trị theo triệu đồng (dùng cho biểu đồ và bảng). */
export const toMillions = (v: number) => v / 1e6
export const formatMillions = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '—' : nf1.format(v / 1e6))

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('vi-VN', { dateStyle: 'medium', timeStyle: 'short' })
}

export const formatPercent = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v) ? '—' : `${new Intl.NumberFormat('vi-VN', { maximumFractionDigits: digits }).format(v)}%`
