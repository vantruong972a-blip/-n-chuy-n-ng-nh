import type { FeatureMeta, MetaResponse, PredictValues } from '../types/api'

export type FormValues = Record<string, string>
export type FormErrors = Record<string, string>

/** Chuẩn hóa số kiểu Việt ("1,5") về số máy tính. */
export const parseNumber = (s: string): number => Number(s.trim().replace(',', '.'))

/** Chỉ cho phép gõ ký tự đúng định dạng số; trả về chuỗi đã lọc hoặc null nếu gõ sai. */
export function sanitizeNumberInput(raw: string, integer: boolean): string | null {
  if (raw === '') return ''
  return (integer ? /^\d{0,7}$/ : /^\d{0,7}([.,]\d{0,2})?$/).test(raw) ? raw : null
}

export function num(f: FeatureMeta, x: number) {
  return f.name === 'year' ? String(x) : x.toLocaleString('vi-VN')
}

function labelOf(f: FeatureMeta) {
  return f.label || f.name
}

export function validateField(f: FeatureMeta, raw: string): string | null {
  const v = raw.trim()
  if (v === '') return f.required ? `Vui lòng nhập ${labelOf(f).toLowerCase()}.` : null
  if (f.type === 'number') {
    const x = parseNumber(v)
    if (!Number.isFinite(x)) return `${labelOf(f)} phải là một số.`
    if (f.integer && !Number.isInteger(x)) return `${labelOf(f)} phải là số nguyên.`
    if (f.min != null && x < f.min) return `${labelOf(f)} không được nhỏ hơn ${num(f, f.min)}.`
    if (f.max != null && x > f.max) return `${labelOf(f)} không được lớn hơn ${num(f, f.max)}.`
    return null
  }
  if (!f.free_text && f.name !== 'model' && !f.options.includes(v)) return `${labelOf(f)} không nằm trong danh sách hỗ trợ.`
  if (v.length > 60) return `${labelOf(f)} quá dài (tối đa 60 ký tự).`
  return null
}

export function validateForm(meta: MetaResponse, values: FormValues): FormErrors {
  const errors: FormErrors = {}
  for (const f of meta.features) {
    const e = validateField(f, values[f.name] ?? '')
    if (e) errors[f.name] = e
  }
  return errors
}

/** Chỉ gửi các trường có giá trị; số được gửi dưới dạng số, chữ được chuẩn hóa về chữ thường như dữ liệu huấn luyện. */
export function toPayload(meta: MetaResponse, values: FormValues): PredictValues {
  const out: PredictValues = {}
  for (const f of meta.features) {
    const v = (values[f.name] ?? '').trim()
    if (v === '') continue
    out[f.name] = f.type === 'number' ? parseNumber(v) : v.toLowerCase()
  }
  return out
}
