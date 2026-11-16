import { useRef, useState, type FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { CheckCircle2, History, RotateCcw, Sparkles, TriangleAlert } from 'lucide-react'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { FormField } from '../components/FormField'
import { InputSummary } from '../components/InputSummary'
import { ModelPriceChart } from '../components/ModelPriceChart'
import { useToast } from '../components/toastContext'
import { Alert, Button, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { addHistory } from '../hooks/usePredictionHistory'
import { formatVnd, formatVndFull } from '../lib/format'
import { FIELD_GROUPS, MODEL_NOTE, humanizeMessage } from '../lib/labels'
import { parsePrefill } from '../lib/prefill'
import { toPayload, validateForm, type FormErrors, type FormValues } from '../lib/validation'
import { ApiError } from '../services/http'
import { getMeta, predictPrice, sendFeedback } from '../services/predictionService'
import type { MetaResponse, PredictResponse, PredictValues } from '../types/api'
import { toApiError } from '../hooks/useAsync'

const EXAMPLE: FormValues = { brand: 'toyota', model: 'vios', year: '2018', mileage_km: '60000', transmission: 'automatic', fuel_type: 'gasoline' }

function FeedbackBox({ result }: { result: PredictResponse }) {
  const toast = useToast()
  const [value, setValue] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState(false)
  const [sent, setSent] = useState<number | null>(null)
  const [err, setErr] = useState('')
  const actual = Number(value.replace(',', '.'))
  const valid = Number.isFinite(actual) && actual > 0 && actual < 100_000

  async function send() {
    setBusy(true)
    try {
      await sendFeedback(result.prediction_id, Math.round(actual * 1e6))
      setSent(actual * 1e6)
      toast.notify('success', 'Đã ghi nhận giá thực tế. Cảm ơn bạn!')
    } catch (e) {
      toast.notify('error', toApiError(e).message)
    } finally {
      setBusy(false)
      setConfirm(false)
    }
  }
  if (sent !== null) return <Alert kind="success" title="Đã gửi giá thực tế">Giá bạn cung cấp: {formatVndFull(sent)}. Hệ thống dùng nó để theo dõi sai số ở trang Quản trị.</Alert>
  return (
    <form className="rounded-xl bg-haze p-4" onSubmit={(e) => { e.preventDefault(); if (!valid) setErr('Nhập giá thực tế lớn hơn 0 (đơn vị: triệu đồng).'); else { setErr(''); setConfirm(true) } }}>
      <label htmlFor="gia-thuc-te" className="text-sm font-semibold text-ink">Bạn biết giá thực tế của xe này?</label>
      <p className="mt-0.5 text-xs text-mute">Gửi để hệ thống theo dõi độ lệch giữa giá dự đoán và thực tế.</p>
      <div className="mt-2 flex gap-2">
        <div className="relative flex-1">
          <input id="gia-thuc-te" inputMode="decimal" className="input pr-16" placeholder="Ví dụ: 430" value={value} aria-invalid={err ? true : undefined}
            onChange={(e) => /^\d{0,6}([.,]\d{0,2})?$/.test(e.target.value) && setValue(e.target.value)} />
          <span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">triệu</span>
        </div>
        <Button type="submit" variant="secondary">Gửi</Button>
      </div>
      {err && <p role="alert" className="mt-1.5 text-sm font-medium text-bad">{err}</p>}
      <ConfirmDialog open={confirm} title="Gửi giá thực tế?" confirmLabel="Xác nhận gửi" loading={busy} onConfirm={send} onCancel={() => setConfirm(false)}>
        Bạn sắp gửi giá thực tế <strong>{formatVndFull(actual * 1e6)}</strong> cho dự đoán này. Dữ liệu được ghi vào nhật ký giám sát và không thể sửa ở giao diện.
      </ConfirmDialog>
    </form>
  )
}

function ListedPriceCompare({ listed, predicted, label }: { listed: number; predicted: number; label: string }) {
  const diff = (listed - predicted) / predicted
  const pct = Math.abs(Math.round(diff * 100))
  const text = pct < 3 ? 'gần bằng' : diff > 0 ? `cao hơn khoảng ${pct}% so với` : `thấp hơn khoảng ${pct}% so với`
  return (
    <div className="rounded-xl border border-line bg-white p-4 text-sm">
      <p className="font-semibold text-ink">So với giá niêm yết của tin rao</p>
      <p className="mt-1 text-ink-soft">Giá niêm yết {formatVndFull(listed)} <strong className="text-navy-deep">{text}</strong> giá mô hình ({label}).</p>
      <p className="mt-1 text-xs leading-relaxed text-mute">Chỉ để tham khảo: sai số của mô hình thường khoảng ±10–20% (lớn hơn với xe giá rất cao), và giá niêm yết thường cao hơn giá chốt.</p>
    </div>
  )
}

function Result({ meta, payload, result, listedPrice }: { meta: MetaResponse; payload: PredictValues; result: PredictResponse; listedPrice: number | null }) {
  const chosen = result.all.find((a) => a.model === result.model)
  const prices = result.all.map((a) => a.price)
  const spread = Math.max(...prices) / Math.min(...prices)
  return (
    <div className="pop-in space-y-6">
      <div>
        <p className="text-sm text-mute">Giá tham khảo theo {result.model_label}</p>
        <p className="mt-1 text-5xl font-extrabold tracking-tight text-navy-deep sm:text-6xl">{formatVnd(chosen?.price ?? result.price)}</p>
        <p className="mt-1 text-sm text-mute">{formatVndFull(result.price)}</p>
        {result.range && (
          <p className="mt-3 text-sm text-ink-soft">
            Khoảng hợp lý (~{Math.round(result.range.level * 100)}%): <strong className="text-navy-deep">{formatVnd(result.range.low)} – {formatVnd(result.range.high)}</strong>
            <span className="block text-xs leading-relaxed text-mute">Đo trên {result.range.n.toLocaleString('vi-VN')} xe thuộc dải giá {result.range.band} mà mô hình chưa từng học. Khoảng này nói về sai số mô hình so với giá niêm yết, không phải giá giao dịch.</span>
          </p>
        )}
        <p className="mt-3 text-xs leading-relaxed text-mute">{result.unit_note}</p>
      </div>
      {spread > 1.3 && (
        <Alert kind="warn" title="Các mô hình đưa ra giá khá khác nhau">
          Giá cao nhất gấp khoảng {spread.toLocaleString('vi-VN', { maximumFractionDigits: 1 })} lần giá thấp nhất. Xe này có thể hiếm hoặc ít mẫu trong dữ liệu, hãy xem kết quả như một khoảng tham khảo rộng.
        </Alert>
      )}
      {result.warnings.length > 0 && (
        <Alert kind="warn" title="Lưu ý về dữ liệu bạn nhập">
          <ul className="list-disc space-y-1 pl-5">{result.warnings.map((w) => <li key={w}>{humanizeMessage(w, meta.features)}</li>)}</ul>
        </Alert>
      )}
      <section aria-labelledby="so-sanh-mo-hinh">
        <h3 id="so-sanh-mo-hinh" className="text-sm font-bold text-ink">So sánh ba mô hình trên cùng một xe</h3>
        <ModelPriceChart items={result.all} selected={result.model} />
        <ul className="mt-1 space-y-1 text-xs text-mute">
          {result.all.map((a) => (
            <li key={a.model}>
              <strong className="text-ink-soft">{a.label}{a.role?.title ? ` – ${a.role.title}` : ''}</strong>: {a.role?.summary ?? MODEL_NOTE[a.model] ?? ''}
            </li>
          ))}
        </ul>
        {result.recommended && (
          <p className="mt-2 text-xs leading-relaxed text-mute">
            {result.recommended.is_default
              ? `Với xe thuộc dải giá ${result.recommended.band}, ${result.recommended.label} cũng là mô hình có sai số tương đối thấp nhất trên tập kiểm định.`
              : `Với xe thuộc dải giá ${result.recommended.band}, ${result.recommended.label} có sai số tương đối thấp hơn mô hình mặc định trên tập kiểm định (${result.recommended.n.toLocaleString('vi-VN')} xe); nên đối chiếu thêm giá của mô hình này.`}
          </p>
        )}
      </section>
      <section aria-labelledby="xe-da-nhap">
        <h3 id="xe-da-nhap" className="mb-3 text-sm font-bold text-ink">Thông tin xe đã nhập</h3>
        <InputSummary features={meta.features} payload={payload} />
        {result.defaults_used.length > 0 && (
          <p className="mt-3 text-xs leading-relaxed text-mute">
            Để trống: {result.defaults_used.map((n) => meta.features.find((f) => f.name === n)?.label ?? n).join(', ')}. Mô hình xử lý như dữ liệu thiếu (dung tích và số chỗ được điền theo dòng xe).
          </p>
        )}
      </section>
      {listedPrice !== null && <ListedPriceCompare listed={listedPrice} predicted={chosen?.price ?? result.price} label={result.model_label} />}
      <FeedbackBox key={result.prediction_id} result={result} />
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-xs text-mute">
        <span>Mô hình: <strong className="text-ink-soft">{result.model_label}</strong></span>
        <span>Phiên bản: <strong className="text-ink-soft">{result.model_version}</strong></span>
        <span>Mã dự đoán: <strong className="text-ink-soft">{result.prediction_id}</strong></span>
      </p>
    </div>
  )
}

export default function ValuationPage() {
  const meta = useAsync(getMeta, 'meta')
  const toast = useToast()
  const [search] = useSearchParams()
  const [prefill] = useState(() => parsePrefill(search))
  const [values, setValues] = useState<FormValues>(prefill.values)
  const [errors, setErrors] = useState<FormErrors>({})
  const [modelChoice, setModelChoice] = useState('')
  const [busy, setBusy] = useState(false)
  const [apiErr, setApiErr] = useState<ApiError | null>(null)
  const [done, setDone] = useState<{ payload: PredictValues; result: PredictResponse } | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)
  const m = meta.data

  const setField = (name: string, v: string) => {
    setValues((s) => ({ ...s, [name]: v, ...(name === 'brand' ? { model: '' } : {}) }))
    setErrors((e) => { if (!(name in e)) return e; const { [name]: _drop, ...rest } = e; void _drop; return rest })
  }

  async function submit(ev: FormEvent) {
    ev.preventDefault()
    if (!m) return
    const errs = validateForm(m, values)
    setErrors(errs)
    const first = Object.keys(errs)[0]
    if (first) {
      document.getElementById(`f-${first}`)?.focus()
      toast.notify('error', 'Vui lòng sửa các trường được đánh dấu đỏ.')
      return
    }
    setBusy(true)
    setApiErr(null)
    const payload = toPayload(m, values)
    try {
      const result = await predictPrice(payload, modelChoice || undefined)
      setDone({ payload, result })
      addHistory(payload, result)
      toast.notify('success', 'Đã có kết quả định giá.')
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    } catch (e) {
      setApiErr(toApiError(e))
    } finally {
      setBusy(false)
    }
  }

  const reset = () => { setValues({}); setErrors({}); setApiErr(null); setDone(null) }
  const fill = () => { setValues(EXAMPLE); setErrors({}) }

  const grouped = m && FIELD_GROUPS.map((g) => ({ ...g, features: g.fields.map((n) => m.features.find((f) => f.name === n)).filter((f) => !!f) }))
  const known = new Set(FIELD_GROUPS.flatMap((g) => g.fields))
  const extra = m?.features.filter((f) => !known.has(f.name)) ?? []
  if (grouped && extra.length) grouped[grouped.length - 1].features.push(...extra)

  return (
    <>
      <PageHeader title="Định giá ô tô">
        Nhập thông tin xe để nhận giá tham khảo từ ba mô hình học máy. Chỉ cần hãng, dòng xe và năm sản xuất; càng nhiều thông tin, dự đoán càng sát.
      </PageHeader>
      <div className="container-page grid gap-8 py-10 lg:grid-cols-[1.15fr_1fr] lg:items-start">
        {prefill.label && (
          <div className="lg:col-span-2">
            <Alert kind="info" title="Đã điền sẵn từ tin rao">
              Thông tin của xe <strong>{prefill.label}</strong> đã được điền sẵn. Kiểm tra lại rồi bấm “Dự đoán giá xe”{prefill.listedPrice !== null ? '; kết quả sẽ so với giá niêm yết của tin đó' : ''}.
            </Alert>
          </div>
        )}
        <form onSubmit={submit} noValidate className="space-y-6" aria-label="Biểu mẫu thông tin xe">
          {meta.loading && <div className="space-y-4"><Skeleton className="h-48" /><Skeleton className="h-64" /></div>}
          {meta.error && (
            <Alert kind="error" title="Chưa tải được biểu mẫu" action={<Button variant="secondary" onClick={meta.reload}>Thử lại</Button>}>
              {meta.error.messages.join(' ')}
            </Alert>
          )}
          {m && grouped?.map((g) => (
            <fieldset key={g.title} className="card p-5 sm:p-6">
              <legend className="float-left mb-1 w-full px-0 text-lg font-bold text-navy-deep">{g.title}</legend>
              <p className="clear-both mb-4 text-sm text-mute">{g.description}</p>
              <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2">
                {g.features.map((f) => (
                  <FormField key={f.name} feature={f} meta={m} value={values[f.name] ?? ''} error={errors[f.name]} disabled={busy} brand={values.brand ?? ''} onChange={setField} />
                ))}
              </div>
            </fieldset>
          ))}
          {m && (
            <div className="card p-5 sm:p-6">
              <label htmlFor="mo-hinh" className="label">Mô hình dự đoán</label>
              <select id="mo-hinh" className="input" value={modelChoice} onChange={(e) => setModelChoice(e.target.value)} disabled={busy}>
                <option value="">Tự động: mô hình tốt nhất ({m.models[m.default_model] ?? m.default_model})</option>
                {Object.entries(m.models).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
              </select>
              <p className="mt-1 text-xs text-mute">Dù chọn mô hình nào, kết quả vẫn hiển thị giá của cả ba mô hình để so sánh.</p>
            </div>
          )}
          {apiErr && (
            <Alert kind="error" title={apiErr.isNetwork ? 'Không kết nối được máy chủ' : 'Không thể dự đoán giá'}>
              {apiErr.messages.length > 1
                ? <ul className="list-disc space-y-1 pl-5">{apiErr.messages.map((x) => <li key={x}>{humanizeMessage(x, m?.features)}</li>)}</ul>
                : humanizeMessage(apiErr.messages[0] ?? '', m?.features)}
            </Alert>
          )}
          <div className="flex flex-col-reverse gap-3 sm:flex-row sm:items-center">
            <Button type="submit" loading={busy} disabled={!m} className="h-12 px-8 text-base">{busy ? 'Đang xử lý…' : 'Dự đoán giá xe'}</Button>
            <Button variant="secondary" onClick={fill} disabled={!m || busy}><Sparkles className="size-4" aria-hidden />Điền ví dụ</Button>
            <Button variant="ghost" onClick={reset} disabled={busy}><RotateCcw className="size-4" aria-hidden />Nhập lại</Button>
          </div>
          <p className="text-xs text-mute"><span className="text-bad">*</span> Trường bắt buộc.</p>
        </form>

        <aside ref={resultRef} className="card scroll-mt-24 p-5 sm:p-7 lg:sticky lg:top-24" aria-live="polite" aria-label="Kết quả định giá">
          {busy && <div className="space-y-4" aria-busy><Skeleton className="h-6 w-1/2" /><Skeleton className="h-16 w-3/4" /><Skeleton className="h-36" /></div>}
          {!busy && done && m && <Result meta={m} payload={done.payload} result={done.result} listedPrice={prefill.listedPrice} />}
          {!busy && !done && (
            <div className="py-8 text-center">
              <CheckCircle2 className="mx-auto size-10 text-navy/30" aria-hidden />
              <h2 className="mt-4 text-lg font-bold text-navy-deep">Kết quả sẽ hiện ở đây</h2>
              <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-mute">Điền thông tin xe rồi bấm “Dự đoán giá xe”. Giá do máy chủ tính, không phải ước lượng trên trình duyệt.</p>
              <p className="mx-auto mt-4 flex max-w-xs items-start gap-2 text-left text-xs leading-relaxed text-mute">
                <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />Đây là giá rao khoảng năm 2022, chỉ để tham khảo.
              </p>
              <Link to="/lich-su" className="mt-4 inline-flex min-h-11 items-center gap-1.5 text-sm font-semibold text-navy hover:underline"><History className="size-4" aria-hidden />Xem lịch sử dự đoán</Link>
            </div>
          )}
        </aside>
      </div>
    </>
  )
}
