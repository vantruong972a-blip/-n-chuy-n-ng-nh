import { formatDecimal, formatInt } from '../lib/format'
import { FIELD_HELP, valueLabel } from '../lib/labels'
import type { FeatureMeta, PredictValues } from '../types/api'

function formatFieldValue(name: string, value: string | number): string {
  if (typeof value === 'number') {
    const unit = FIELD_HELP[name]?.unit
    const text = name === 'year' ? String(value) : Number.isInteger(value) ? formatInt(value) : formatDecimal(value, 2)
    return unit ? `${text} ${unit}` : text
  }
  return valueLabel(name, value)
}

/** Bảng thông tin xe đã gửi lên API (đúng giá trị người dùng nhập, đã chuẩn hóa). */
export function InputSummary({ features, payload }: { features: FeatureMeta[] | undefined; payload: PredictValues }) {
  const order = features?.map((f) => f.name) ?? Object.keys(payload)
  const rows = order.filter((n) => payload[n] !== undefined)
  const labelOf = (n: string) => features?.find((f) => f.name === n)?.label ?? n
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-sm">
      {rows.map((n) => (
        <div key={n} className="min-w-0">
          <dt className="text-mute">{labelOf(n)}</dt>
          <dd className="truncate font-semibold text-ink">{formatFieldValue(n, payload[n])}</dd>
        </div>
      ))}
    </dl>
  )
}
