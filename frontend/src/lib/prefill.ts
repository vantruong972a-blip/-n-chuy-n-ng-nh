import { formatBrand, formatModel } from './labels'
import type { CarListing } from '../types/api'
import type { FormValues } from './validation'

/** Các trường biểu mẫu định giá có thể điền sẵn từ đường dẫn (khớp tên trường của /api/meta). */
const FIELDS = ['brand', 'model', 'variant', 'body_type', 'year', 'mileage_km', 'engine_capacity', 'transmission', 'fuel_type', 'drivetrain', 'seats', 'origin', 'province'] as const

/** Tạo liên kết sang trang Định giá, điền sẵn thông tin của tin rao và mang theo giá niêm yết để so sánh. */
export function valuationLink(car: CarListing): string {
  const q = new URLSearchParams()
  for (const f of FIELDS) {
    const v = (car as unknown as Record<string, unknown>)[f]
    if (v !== null && v !== undefined && String(v).trim() !== '') q.set(f, String(v))
  }
  if (Number.isFinite(car.price) && car.price > 0) q.set('gia_niem_yet', String(Math.round(car.price)))
  return `/dinh-gia?${q.toString()}`
}

/** Đọc lại các tham số trên; bỏ giá trị lạ hoặc quá dài. Việc kiểm tra hợp lệ vẫn do validateForm làm khi gửi. */
export function parsePrefill(search: URLSearchParams): { values: FormValues; listedPrice: number | null; label: string | null } {
  const values: FormValues = {}
  for (const f of FIELDS) {
    const v = search.get(f)
    if (v && v.length <= 60) values[f] = v
  }
  const raw = Number(search.get('gia_niem_yet'))
  const listedPrice = Number.isFinite(raw) && raw > 0 && raw < 1e12 ? raw : null
  const label = values.brand && values.model && values.year ? `${formatBrand(String(values.brand))} ${formatModel(String(values.model))} ${values.year}` : null
  return { values, listedPrice: Object.keys(values).length ? listedPrice : null, label: Object.keys(values).length ? label : null }
}
