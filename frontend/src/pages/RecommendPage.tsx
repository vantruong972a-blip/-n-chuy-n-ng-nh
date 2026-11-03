import { useState, type FormEvent } from 'react'
import { SearchX } from 'lucide-react'
import { CarCard } from '../components/CarCard'
import { NotIntegrated } from '../components/NotIntegrated'
import { Alert, Button, PageHeader, Skeleton } from '../components/ui'
import { useAsync, toApiError } from '../hooks/useAsync'
import { formatBrand, valueLabel } from '../lib/labels'
import { getMeta } from '../services/predictionService'
import { isRecommendationAvailable, recommendCars } from '../services/recommendationService'
import type { CarListing, CategoryFeature, MetaResponse, RecommendFilters } from '../types/api'

interface Form { min: string; max: string; brand: string; model: string; year: string; fuel_type: string; transmission: string; body_type: string; method: 'knn' | 'content_based' }
const EMPTY: Form = { min: '', max: '', brand: '', model: '', year: '', fuel_type: '', transmission: '', body_type: '', method: 'content_based' }

const cat = (m: MetaResponse | undefined, name: string) => m?.features.find((f): f is CategoryFeature => f.name === name && f.type === 'category')

function Select({ id, label, meta, name, value, onChange }: { id: string; label: string; meta?: MetaResponse; name: string; value: string; onChange: (v: string) => void }) {
  const f = cat(meta, name)
  const opts = (f?.options ?? []).map((o) => ({ value: o, label: name === 'brand' ? formatBrand(o) : valueLabel(name, o) })).sort((a, b) => a.label.localeCompare(b.label, 'vi'))
  return (
    <div>
      <label htmlFor={id} className="label">{label}</label>
      <select id={id} className="input" value={value} onChange={(e) => onChange(e.target.value)} disabled={!f}>
        <option value="">Tất cả</option>
        {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}

export default function RecommendPage() {
  const meta = useAsync(getMeta, 'meta')
  const cap = useAsync(isRecommendationAvailable, 'cap-recommend')
  const [f, setF] = useState<Form>(EMPTY)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [items, setItems] = useState<CarListing[] | null>(null)
  const [msg, setMsg] = useState<{ kind: 'notready' | 'error'; text: string } | null>(null)
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((s) => ({ ...s, [k]: v, ...(k === 'brand' ? { model: '' } : {}) }))
  const num = (k: 'min' | 'max' | 'year') => (e: string) => /^\d{0,7}$/.test(e) && set(k, e)
  const yearMeta = meta.data?.features.find((x) => x.name === 'year')
  const yMin = yearMeta?.type === 'number' ? yearMeta.min : null
  const yMax = yearMeta?.type === 'number' ? yearMeta.max : null

  async function submit(ev: FormEvent) {
    ev.preventDefault()
    const e: Record<string, string> = {}
    const lo = f.min ? Number(f.min) : undefined
    const hi = f.max ? Number(f.max) : undefined
    if (lo != null && hi != null && lo > hi) e.max = 'Ngân sách tối đa phải lớn hơn hoặc bằng ngân sách tối thiểu.'
    if (f.year && yMin != null && yMax != null && (Number(f.year) < yMin || Number(f.year) > yMax)) e.year = `Năm sản xuất từ ${yMin} đến ${yMax}.`
    setErrors(e)
    if (Object.keys(e).length) return
    const filters: RecommendFilters = {
      method: f.method, limit: 12,
      ...(lo != null && { budget_min: lo * 1e6 }), ...(hi != null && { budget_max: hi * 1e6 }),
      ...(f.brand && { brand: f.brand }), ...(f.model.trim() && { model: f.model.trim().toLowerCase() }),
      ...(f.year && { year_from: Number(f.year) }), ...(f.fuel_type && { fuel_type: f.fuel_type }),
      ...(f.transmission && { transmission: f.transmission }), ...(f.body_type && { body_type: f.body_type }),
    }
    setBusy(true); setMsg(null); setItems(null)
    try {
      if (!(await isRecommendationAvailable())) {
        setMsg({ kind: 'notready', text: 'Máy chủ chưa có API gợi ý xe (POST /api/recommend) nên chưa thể tìm xe. Bộ lọc của bạn không bị mất.' })
      } else {
        setItems((await recommendCars(filters)).items)
      }
    } catch (x) {
      setMsg({ kind: 'error', text: toApiError(x).message })
    } finally { setBusy(false) }
  }

  const available = cap.data === true
  return (
    <>
      <PageHeader title="Gợi ý xe phù hợp">Chọn ngân sách và nhu cầu, hệ thống sẽ tìm những xe có đặc điểm và mức giá tương đồng.</PageHeader>
      <div className="container-page space-y-8 py-10">
        {cap.loading ? <Skeleton className="h-24" /> : available
          ? <Alert kind="success" title="Máy chủ đã có API gợi ý xe">Bạn có thể tìm xe bằng bộ lọc bên dưới.</Alert>
          : (
            <NotIntegrated title="Chức năng gợi ý xe chưa được tích hợp">
              <p>Backend hiện chỉ có API định giá và giám sát. Chưa có nguồn dữ liệu danh sách xe, ảnh xe hay API gợi ý, nên trang này <strong>không hiển thị xe nào</strong> để tránh đưa ra dữ liệu giả.</p>
              <p>Giao diện và lớp gọi API đã sẵn sàng. Khi máy chủ có <code className="rounded bg-white px-1.5 py-0.5 text-xs">POST /api/recommend</code> (KNN tìm xe tương đồng, hoặc lọc theo nội dung) và <code className="rounded bg-white px-1.5 py-0.5 text-xs">GET /api/cars/&#123;id&#125;</code>, trang sẽ tự nhận ra và hiển thị kết quả. Hợp đồng đề xuất nằm trong README.</p>
              {cap.error && <p className="font-medium text-bad">Không đọc được danh sách API của máy chủ: {cap.error.message}</p>}
            </NotIntegrated>
          )}

        <form onSubmit={submit} noValidate className="card p-5 sm:p-7" aria-label="Bộ lọc tìm xe">
          <h2 className="text-lg font-bold text-navy-deep">Tiêu chí của bạn</h2>
          <div className="mt-5 grid gap-x-5 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label htmlFor="ns-min" className="label">Ngân sách tối thiểu</label>
              <div className="relative"><input id="ns-min" inputMode="numeric" className="input pr-16" placeholder="Ví dụ: 300" value={f.min} onChange={(e) => num('min')(e.target.value)} /><span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">triệu</span></div>
            </div>
            <div>
              <label htmlFor="ns-max" className="label">Ngân sách tối đa</label>
              <div className="relative"><input id="ns-max" inputMode="numeric" className="input pr-16" placeholder="Ví dụ: 600" value={f.max} aria-invalid={errors.max ? true : undefined} onChange={(e) => num('max')(e.target.value)} /><span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">triệu</span></div>
              {errors.max && <p role="alert" className="mt-1 text-sm font-medium text-bad">{errors.max}</p>}
            </div>
            <Select id="hang" label="Hãng xe" meta={meta.data} name="brand" value={f.brand} onChange={(v) => set('brand', v)} />
            <div>
              <label htmlFor="dong" className="label">Dòng xe</label>
              <input id="dong" className="input" list="ds-dong" maxLength={60} placeholder="Ví dụ: vios" value={f.model} onChange={(e) => set('model', e.target.value)} />
              <datalist id="ds-dong">{(meta.data?.brand_models[f.brand] ?? []).slice(0, 200).map((o) => <option key={o} value={o} />)}</datalist>
            </div>
            <div>
              <label htmlFor="nam" className="label">Năm sản xuất từ</label>
              <input id="nam" inputMode="numeric" className="input" placeholder={yMin ? `Từ ${yMin}` : 'Ví dụ: 2016'} value={f.year} aria-invalid={errors.year ? true : undefined} onChange={(e) => num('year')(e.target.value)} />
              {errors.year && <p role="alert" className="mt-1 text-sm font-medium text-bad">{errors.year}</p>}
            </div>
            <Select id="nhien-lieu" label="Loại nhiên liệu" meta={meta.data} name="fuel_type" value={f.fuel_type} onChange={(v) => set('fuel_type', v)} />
            <Select id="hop-so" label="Hộp số" meta={meta.data} name="transmission" value={f.transmission} onChange={(v) => set('transmission', v)} />
            <Select id="kieu-dang" label="Kiểu dáng xe" meta={meta.data} name="body_type" value={f.body_type} onChange={(v) => set('body_type', v)} />
          </div>
          <fieldset className="mt-6">
            <legend className="label">Cách gợi ý</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {([['content_based', 'Theo nhu cầu của bạn', 'Content-Based Filtering: so khớp tiêu chí bạn chọn với đặc điểm xe.'], ['knn', 'Xe tương đồng về giá và đặc điểm', 'KNN: tìm các xe gần nhất trong không gian đặc trưng.']] as const).map(([v, t, d]) => (
                <label key={v} className={`flex cursor-pointer gap-3 rounded-xl border p-4 transition-colors ${f.method === v ? 'border-navy bg-azure-soft' : 'border-line hover:border-ink/30'}`}>
                  <input type="radio" name="cach" className="mt-1 size-4 accent-navy" checked={f.method === v} onChange={() => set('method', v)} />
                  <span><span className="block text-sm font-semibold text-ink">{t}</span><span className="block text-xs text-mute">{d}</span></span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="mt-6 flex flex-col gap-3 sm:flex-row">
            <Button type="submit" loading={busy} className="h-12 px-8 text-base">Tìm xe phù hợp</Button>
            <Button variant="ghost" onClick={() => { setF(EMPTY); setErrors({}); setItems(null); setMsg(null) }}>Xóa bộ lọc</Button>
          </div>
        </form>

        {msg && (msg.kind === 'notready'
          ? <Alert kind="info" title="Chưa thể tìm xe">{msg.text}</Alert>
          : <Alert kind="error" title="Không tìm được xe">{msg.text}</Alert>)}
        {items && items.length === 0 && (
          <div className="card p-10 text-center"><SearchX className="mx-auto size-9 text-mute" aria-hidden /><h2 className="mt-3 text-lg font-bold text-navy-deep">Không có xe nào khớp</h2><p className="mt-1 text-sm text-mute">Hãy nới rộng ngân sách hoặc bỏ bớt bộ lọc.</p></div>
        )}
        {items && items.length > 0 && (
          <section aria-label="Kết quả gợi ý"><h2 className="mb-4 text-xl font-bold text-navy-deep">{items.length} xe phù hợp</h2>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">{items.map((c) => <CarCard key={c.id} car={c} />)}</div></section>
        )}
      </div>
    </>
  )
}
