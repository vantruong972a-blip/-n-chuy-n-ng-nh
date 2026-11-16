import type { MetaResponse, StrengthSegment, StrengthsResponse } from '../types/api'

const TYPE_LABEL: Record<string, string> = { 'giá dự đoán': 'Dải giá', 'tuổi xe': 'Tuổi xe', 'số km': 'Số km' }
const TYPE_ORDER = ['giá dự đoán', 'tuổi xe', 'số km']

/** Các nhóm xe mà mô hình `key` có sai số log thấp nhất, gom theo loại thông tin (giá, tuổi, km). */
function leadingGroups(segments: StrengthSegment[], key: string) {
  const out: { type: string; items: { text: string; close: boolean }[] }[] = []
  for (const t of TYPE_ORDER) {
    const items = segments.filter((s) => s.type === t && s.best === key).map((s) => ({ text: s.segment, close: s.close }))
    if (items.length) out.push({ type: t, items })
  }
  return out
}

interface Props {
  meta: MetaResponse
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  strengths: StrengthsResponse | null
}

/** Chọn mô hình dự đoán dưới dạng thẻ; mỗi thẻ nói rõ mô hình tốt ở nhóm xe nào, lấy từ sai số đo thật trên tập kiểm định. */
export function ModelPicker({ meta, value, onChange, disabled, strengths }: Props) {
  const segs = strengths?.available ? strengths.segments : []
  const detail = segs.filter((s) => TYPE_ORDER.includes(s.type))
  const defaultLabel = meta.models[meta.default_model] ?? meta.default_model
  const options: { key: string; title: string; role?: string | null; text: string }[] = [
    { key: '', title: 'Tự động (khuyên dùng)', role: null, text: `Dùng mô hình có sai số thấp nhất trên tập kiểm định: ${defaultLabel}.` },
    ...Object.entries(meta.models).map(([k, label]) => ({ key: k, title: label, role: meta.roles?.[k]?.title ?? null, text: meta.roles?.[k]?.summary ?? '' })),
  ]
  const cell = (s: StrengthSegment, k: string) => {
    const v = s.models[k]?.rmsle
    const best = s.best === k
    return <td key={k} className={`px-3 py-2 text-right tabular-nums ${best ? 'bg-azure-soft font-bold text-navy' : ''}`}>{v == null ? '—' : v.toFixed(3).replace('.', ',')}</td>
  }

  return (
    <fieldset className="card p-5 sm:p-6" disabled={disabled}>
      <legend className="label px-0">Chọn mô hình dự đoán</legend>
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        {options.map((o) => {
          const groups = o.key ? leadingGroups(segs, o.key) : []
          return (
            <label key={o.key || 'auto'} className={`flex cursor-pointer gap-3 rounded-xl border p-4 transition-colors ${value === o.key ? 'border-navy bg-azure-soft' : 'border-line hover:border-ink/30'}`}>
              <input type="radio" name="mo-hinh" className="mt-1 size-4 shrink-0 accent-navy" checked={value === o.key} onChange={() => onChange(o.key)} />
              <span className="min-w-0">
                <span className="block text-sm font-semibold text-ink">{o.title}{o.role ? <span className="font-normal text-mute"> – {o.role}</span> : null}</span>
                <span className="mt-0.5 block text-xs text-mute">{o.text}</span>
                {o.key && strengths?.available && (
                  <span className="mt-2 block text-xs text-ink-soft">
                    {groups.length === 0 ? (
                      <>Không dẫn đầu độ chính xác ở nhóm xe nào trong dữ liệu kiểm định.</>
                    ) : (
                      <>
                        <strong className="font-semibold">Chính xác nhất khi:</strong>
                        {groups.map((g) => (
                          <span key={g.type} className="block">
                            {TYPE_LABEL[g.type] ?? g.type}: {g.items.map((i) => (i.close ? `${i.text} (sát nút)` : i.text)).join('; ')}
                          </span>
                        ))}
                      </>
                    )}
                  </span>
                )}
              </span>
            </label>
          )
        })}
      </div>
      <p className="mt-3 text-xs text-mute">Dù chọn mô hình nào, kết quả vẫn hiển thị giá của cả ba mô hình để so sánh.</p>
      {detail.length > 0 && (
        <details className="mt-3 rounded-lg border border-line p-3">
          <summary className="cursor-pointer text-sm font-semibold text-navy">Xem sai số từng mô hình theo thông tin của xe</summary>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[480px] text-left text-sm">
              <caption className="sr-only">Sai số log (RMSLE) của từng mô hình theo nhóm xe trên tập kiểm định; nhỏ hơn là tốt hơn.</caption>
              <thead className="text-xs text-mute">
                <tr>
                  <th scope="col" className="px-3 py-2 font-semibold">Nhóm xe</th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">Số xe</th>
                  {Object.entries(meta.models).map(([k, label]) => <th key={k} scope="col" className="px-3 py-2 text-right font-semibold">{label}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {TYPE_ORDER.flatMap((t) => {
                  const rows = detail.filter((s) => s.type === t)
                  if (!rows.length) return []
                  return [
                    <tr key={`h-${t}`}><th colSpan={2 + Object.keys(meta.models).length} scope="colgroup" className="bg-haze px-3 py-1.5 text-xs font-semibold text-mute">{TYPE_LABEL[t] ?? t}</th></tr>,
                    ...rows.map((s) => (
                      <tr key={`${t}-${s.segment}`}>
                        <th scope="row" className="px-3 py-2 font-medium text-ink">{s.segment}</th>
                        <td className="px-3 py-2 text-right tabular-nums text-mute">{s.n.toLocaleString('vi-VN')}</td>
                        {Object.keys(meta.models).map((k) => cell(s, k))}
                      </tr>
                    )),
                  ]
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-mute">Chỉ số là sai số log (RMSLE), nhỏ hơn là tốt hơn; ô nền xanh, chữ đậm là mô hình thấp nhất trong nhóm. Đo trên các xe kiểm định không dùng để huấn luyện. Chênh lệch giữa các mô hình thường nhỏ vì dữ liệu giá rao có nhiễu, nên đây là tham khảo, không phải cam kết.</p>
        </details>
      )}
    </fieldset>
  )
}
