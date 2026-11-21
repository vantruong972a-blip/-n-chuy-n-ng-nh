import { useEffect, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { SearchX, SlidersHorizontal, X } from 'lucide-react'
import { CarCard } from '../components/CarCard'
import { Combobox, type ComboOption } from '../components/Combobox'
import { NotIntegrated } from '../components/NotIntegrated'
import { Alert, Button, PageHeader, Skeleton } from '../components/ui'
import { useAsync, toApiError } from '../hooks/useAsync'
import { formatBrand, formatModel, valueLabel } from '../lib/labels'
import { getMeta } from '../services/predictionService'
import { getCarDetail, isRecommendationAvailable, recommendCars } from '../services/recommendationService'
import type { CarListing, CategoryFeature, MetaResponse, RecommendFilters } from '../types/api'

interface Form { min: string; max: string; brand: string; model: string; year: string; fuel_type: string; transmission: string; body_type: string; method: 'knn' | 'content_based' | 'rule_based' }
const EMPTY: Form = { min: '', max: '', brand: '', model: '', year: '', fuel_type: '', transmission: '', body_type: '', method: 'content_based' }

const STORE = 'goi-y-state-v1'
interface Saved { f: Form; items: CarListing[] | null; total: number; info: { notes: string[]; suggestions: string[]; field: 'brand' | 'model' | null } }
// Giữ tiêu chí và kết quả khi người dùng sang trang chi tiết xe rồi quay lại (chỉ trong phiên trình duyệt này).
function loadSaved(): Partial<Saved> {
  try { const raw = sessionStorage.getItem(STORE); return raw ? (JSON.parse(raw) as Partial<Saved>) : {} } catch { return {} }
}

const BUDGETS = [
  { label: 'Dưới 300 triệu', min: '', max: '300' },
  { label: '300 – 600 triệu', min: '300', max: '600' },
  { label: '600 triệu – 1 tỷ', min: '600', max: '1000' },
  { label: 'Trên 1 tỷ', min: '1000', max: '' },
]

const UNKNOWN = ['other', 'khac', 'khác']
// Hãng chỉ có dòng "other" thì trang gợi ý không có xe nào để hiện nên không đưa vào danh sách chọn.
const hasListableModels = (m: MetaResponse | undefined, brand: string) => {
  const models = m?.brand_models[brand]
  return !models || models.some((x) => !UNKNOWN.includes(x.toLowerCase()))
}

const cat = (m: MetaResponse | undefined, name: string) => m?.features.find((f): f is CategoryFeature => f.name === name && f.type === 'category')

function Select({ id, label, meta, name, value, onChange, keep }: { id: string; label: string; meta?: MetaResponse; name: string; value: string; onChange: (v: string) => void; keep?: (o: string) => boolean }) {
  const f = cat(meta, name)
  const opts = (f?.options ?? []).filter((o) => !keep || keep(o)).map((o) => ({ value: o, label: name === 'brand' ? formatBrand(o) : valueLabel(name, o) })).sort((a, b) => a.label.localeCompare(b.label, 'vi'))
  return (
    <div>
      <label htmlFor={id} className="label">{label}</label>
      <Combobox id={id} options={opts} value={value} onChange={onChange} emptyLabel="Tất cả" placeholder="Tất cả" disabled={!f} noResults="Không có lựa chọn nào khớp" />
    </div>
  )
}

export default function RecommendPage() {
  const meta = useAsync(getMeta, 'meta')
  const cap = useAsync(isRecommendationAvailable, 'cap-recommend')
  const [search, setSearch] = useSearchParams()
  const likedId = (search.get('like') ?? '').slice(0, 60)
  const liked = useAsync(async () => (likedId ? await getCarDetail(likedId) : null), `like-${likedId}`)
  const [saved] = useState(loadSaved)
  const [f, setF] = useState<Form>({ ...EMPTY, ...(saved.f ?? {}) })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [items, setItems] = useState<CarListing[] | null>(saved.items ?? null)
  const [total, setTotal] = useState(saved.total ?? 0)
  const [info, setInfo] = useState<{ notes: string[]; suggestions: string[]; field: 'brand' | 'model' | null }>(saved.info ?? { notes: [], suggestions: [], field: null })
  const [msg, setMsg] = useState<{ kind: 'notready' | 'error'; text: string } | null>(null)
  useEffect(() => {
    try { sessionStorage.setItem(STORE, JSON.stringify({ f, items, total, info })) } catch { /* trình duyệt chặn lưu: bỏ qua */ }
  }, [f, items, total, info])
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((s) => ({ ...s, [k]: v, ...(k === 'brand' ? { model: '' } : {}) }))
  const num = (k: 'min' | 'max' | 'year') => (e: string) => /^\d{0,7}$/.test(e) && set(k, e)
  const modelOptions = (() => {
    const list = (meta.data?.brand_models[f.brand] ?? []).filter((m) => !['other', 'khac', 'khác'].includes(m.toLowerCase()))
    if (f.model && !list.includes(f.model)) list.push(f.model)   // giữ giá trị đã lưu từ lần tìm trước
    return list.sort((a, b) => formatModel(a).localeCompare(formatModel(b), 'vi'))
  })()
  const brandOptions: ComboOption[] = (cat(meta.data, 'brand')?.options ?? [])
    .filter((o) => hasListableModels(meta.data, o))
    .map((o) => {
      const n = (meta.data?.brand_models[o] ?? []).filter((m) => !UNKNOWN.includes(m.toLowerCase())).length
      return { value: o, label: formatBrand(o), hint: n ? `${n} dòng xe` : undefined }
    })
    .sort((a, b) => a.label.localeCompare(b.label, 'vi'))
  const modelComboOptions: ComboOption[] = modelOptions.map((o) => ({ value: o, label: formatModel(o) }))
  const yearMeta = meta.data?.features.find((x) => x.name === 'year')
  const yMin = yearMeta?.type === 'number' ? yearMeta.min : null
  const yMax = yearMeta?.type === 'number' ? yearMeta.max : null

  async function submit(ev?: FormEvent) {
    ev?.preventDefault()
    const e: Record<string, string> = {}
    const lo = f.min ? Number(f.min) : undefined
    const hi = f.max ? Number(f.max) : undefined
    if (lo != null && hi != null && lo > hi) e.max = 'Ngân sách tối đa phải lớn hơn hoặc bằng ngân sách tối thiểu.'
    if (f.year && yMin != null && yMax != null && (Number(f.year) < yMin || Number(f.year) > yMax)) e.year = `Năm sản xuất từ ${yMin} đến ${yMax}.`
    setErrors(e)
    if (Object.keys(e).length) return
    const filters: RecommendFilters = {
      method: f.method, limit: 12,
      ...(likedId && f.method === 'content_based' && { liked_ids: [likedId] }),
      ...(lo != null && { budget_min: lo * 1e6 }), ...(hi != null && { budget_max: hi * 1e6 }),
      ...(f.brand && { brand: f.brand }), ...(f.model.trim() && { model: f.model.trim().toLowerCase() }),
      ...(f.year && { year_from: Number(f.year) }), ...(f.fuel_type && { fuel_type: f.fuel_type }),
      ...(f.transmission && { transmission: f.transmission }), ...(f.body_type && { body_type: f.body_type }),
    }
    setBusy(true); setMsg(null); setItems(null); setInfo({ notes: [], suggestions: [], field: null })
    try {
      if (!(await isRecommendationAvailable())) {
        setMsg({ kind: 'notready', text: 'Máy chủ chưa có API gợi ý xe (POST /api/recommend) nên chưa thể tìm xe. Bộ lọc của bạn không bị mất.' })
      } else {
        const res = await recommendCars(filters)
        setItems(res.items)
        setTotal(res.total ?? res.items.length)
        setInfo({ notes: res.notes ?? [], suggestions: res.suggestions ?? [], field: res.suggestion_for ?? null })
      }
    } catch (x) {
      setMsg({ kind: 'error', text: toApiError(x).message })
    } finally { setBusy(false) }
  }

  // Quay lại trang với kết quả đã lưu: tìm lại bằng đúng bộ lọc đó để ảnh, giá, số lượng luôn là dữ liệu mới nhất từ máy chủ
  useEffect(() => {
    if (!saved.items) return
    const t = setTimeout(() => { void submit() }, 0)
    return () => clearTimeout(t)
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const available = cap.data === true
  return (
    <>
      <PageHeader title="Gợi ý xe phù hợp">Chọn ngân sách và nhu cầu, hệ thống sẽ tìm những xe có đặc điểm và mức giá tương đồng.</PageHeader>
      <div className="container-page space-y-8 py-10">
        {cap.loading ? <Skeleton className="h-24" /> : available
          ? null
          : (
            <NotIntegrated title="Chức năng gợi ý xe chưa được tích hợp">
              <p>Backend hiện chỉ có API định giá và giám sát. Chưa có nguồn dữ liệu danh sách xe, ảnh xe hay API gợi ý, nên trang này <strong>không hiển thị xe nào</strong> để tránh đưa ra dữ liệu giả.</p>
              <p>Giao diện và lớp gọi API đã sẵn sàng. Khi máy chủ có <code className="rounded bg-white px-1.5 py-0.5 text-xs">POST /api/recommend</code> (KNN tìm xe tương đồng, hoặc lọc theo nội dung) và <code className="rounded bg-white px-1.5 py-0.5 text-xs">GET /api/cars/&#123;id&#125;</code>, trang sẽ tự nhận ra và hiển thị kết quả. Hợp đồng đề xuất nằm trong README.</p>
              {cap.error && <p className="font-medium text-bad">Không đọc được danh sách API của máy chủ: {cap.error.message}</p>}
            </NotIntegrated>
          )}

        <form onSubmit={submit} noValidate className="card p-5 sm:p-7" aria-label="Bộ lọc tìm xe">
          <div className="flex items-center gap-3">
            <span className="icon-tile"><SlidersHorizontal className="size-5" aria-hidden /></span>
            <div><h2 className="text-lg font-bold leading-tight text-navy-deep">Tiêu chí của bạn</h2><p className="text-sm text-mute">Bỏ trống những gì bạn chưa quan tâm.</p></div>
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <span className="text-sm text-mute">Chọn nhanh ngân sách:</span>
            {BUDGETS.map((b) => (
              <button key={b.label} type="button" className={`chip ${f.min === b.min && f.max === b.max ? '!border-navy !bg-azure-soft !text-navy' : ''}`}
                aria-pressed={f.min === b.min && f.max === b.max} onClick={() => setF((x) => ({ ...x, min: b.min, max: b.max }))}>{b.label}</button>
            ))}
          </div>
          <div className="mt-4 grid gap-x-5 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label htmlFor="ns-min" className="label">Ngân sách tối thiểu</label>
              <div className="relative"><input id="ns-min" inputMode="numeric" className="input pr-16" placeholder="Ví dụ: 300" value={f.min} onChange={(e) => num('min')(e.target.value)} /><span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">triệu</span></div>
            </div>
            <div>
              <label htmlFor="ns-max" className="label">Ngân sách tối đa</label>
              <div className="relative"><input id="ns-max" inputMode="numeric" className="input pr-16" placeholder="Ví dụ: 600" value={f.max} aria-invalid={errors.max ? true : undefined} onChange={(e) => num('max')(e.target.value)} /><span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">triệu</span></div>
              {errors.max && <p role="alert" className="mt-1 text-sm font-medium text-bad">{errors.max}</p>}
            </div>
            <div>
              <label htmlFor="hang" className="label">Hãng xe</label>
              <Combobox id="hang" options={brandOptions} value={f.brand} onChange={(v) => set('brand', v)} emptyLabel="Tất cả hãng xe" placeholder="Chọn hoặc gõ tên hãng" disabled={!meta.data} noResults="Không có hãng nào khớp" />
            </div>
            <div>
              <label htmlFor="dong" className="label">Dòng xe</label>
              <Combobox id="dong" options={modelComboOptions} value={f.model} onChange={(v) => set('model', v)} emptyLabel="Tất cả dòng xe" placeholder={f.brand ? 'Chọn hoặc gõ tên dòng xe' : 'Chọn hãng xe trước'} disabled={!f.brand} noResults="Không có dòng xe nào khớp" />
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
          {likedId && (
            <div className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-navy bg-azure-soft p-4 text-sm">
              <span className="font-semibold text-ink">Gợi ý các xe giống xe bạn thích:</span>
              {liked.loading ? <span className="text-mute">đang tải…</span> : liked.data
                ? <span className="font-semibold text-navy-deep">{formatBrand(liked.data.brand)} {formatModel(liked.data.model)} {liked.data.year}</span>
                : <span className="text-bad">không tìm thấy xe mã {likedId}</span>}
              <button type="button" className="inline-flex min-h-9 items-center gap-1 rounded-full border border-line bg-white px-3 text-xs font-semibold text-navy hover:border-navy"
                onClick={() => { const n = new URLSearchParams(search); n.delete('like'); setSearch(n, { replace: true }) }}><X className="size-3.5" aria-hidden />Bỏ xe này</button>
              {f.method !== 'content_based' && <span className="basis-full text-xs text-mute">Xe yêu thích chỉ dùng với cách “Theo hồ sơ nội dung (Content-Based)”.</span>}
            </div>
          )}
          <fieldset className="mt-6">
            <legend className="label">Cách gợi ý</legend>
            <div className="grid gap-3 lg:grid-cols-3">
              {([
                ['content_based', 'Theo hồ sơ nội dung (Content-Based)', 'Dựng hồ sơ từ tiêu chí hoặc xe bạn thích (hãng, dòng, nhiên liệu, hộp số, kiểu dáng, giá, năm, km), rồi xếp xe theo độ tương đồng.'],
                ['rule_based', 'Lọc theo tiêu chí và chấm điểm', 'Chỉ giữ xe khớp mọi tiêu chí bạn chọn, xếp theo độ mới, số km và mức giá.'],
                ['knn', 'Xe tương đồng về giá và đặc điểm (KNN)', 'Tìm các xe gần nhất trong không gian đặc trưng.'],
              ] as const).map(([v, t, d]) => (
                <label key={v} className={`flex cursor-pointer gap-3 rounded-xl border p-4 transition-colors ${f.method === v ? 'border-navy bg-azure-soft' : 'border-line hover:border-ink/30'}`}>
                  <input type="radio" name="cach" className="mt-1 size-4 accent-navy" checked={f.method === v} onChange={() => set('method', v)} />
                  <span><span className="block text-sm font-semibold text-ink">{t}</span><span className="block text-xs text-mute">{d}</span></span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="mt-6 flex flex-col gap-3 sm:flex-row">
            <Button type="submit" loading={busy} className="h-12 px-8 text-base">Tìm xe phù hợp</Button>
            <Button variant="ghost" onClick={() => { setF(EMPTY); setErrors({}); setItems(null); setMsg(null); setInfo({ notes: [], suggestions: [], field: null }) }}>Xóa bộ lọc</Button>
          </div>
        </form>

        {msg && (msg.kind === 'notready'
          ? <Alert kind="info" title="Chưa thể tìm xe">{msg.text}</Alert>
          : <Alert kind="error" title="Không tìm được xe">{msg.text}</Alert>)}
        {info.notes.length > 0 && (
          <Alert kind="info" title={items && items.length === 0 ? 'Không tìm thấy tên bạn nhập' : 'Đã khớp tên gần đúng'}>
            <ul className="list-disc space-y-1 pl-5">{info.notes.map((n) => <li key={n}>{n}</li>)}</ul>
            {info.suggestions.length > 0 && info.field && (
              <p className="mt-3 flex flex-wrap items-center gap-2">
                <span>Thử:</span>
                {info.suggestions.map((sg) => (
                  <button key={sg} type="button" className="min-h-9 rounded-full border border-line bg-white px-3 text-sm font-semibold text-navy hover:border-navy hover:bg-azure-soft"
                    onClick={() => { set(info.field as 'brand' | 'model', sg); setInfo({ notes: [], suggestions: [], field: null }); setItems(null) }}>{sg}</button>
                ))}
                <span className="text-xs text-mute">(bấm để điền vào bộ lọc, rồi bấm “Tìm xe phù hợp”)</span>
              </p>
            )}
          </Alert>
        )}
        {busy && (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3" aria-busy aria-label="Đang tìm xe">
            {[0, 1, 2].map((i) => <div key={i} className="card overflow-hidden"><Skeleton className="aspect-[16/10] rounded-none" /><div className="space-y-3 p-5"><Skeleton className="h-6 w-2/3" /><Skeleton className="h-8 w-1/2" /><Skeleton className="h-9" /></div></div>)}
          </div>
        )}
        {!busy && !items && !msg && available && info.notes.length === 0 && (
          <div className="rounded-[var(--radius-card)] border border-dashed border-line bg-white/60 p-10 text-center">
            <SearchX className="mx-auto size-9 text-navy/30" aria-hidden />
            <h2 className="mt-3 text-lg font-bold text-navy-deep">Chưa có kết quả</h2>
            <p className="mx-auto mt-1 max-w-md text-sm text-mute">Chọn ngân sách hoặc hãng xe ở trên rồi bấm “Tìm xe phù hợp”. Bạn có thể gõ tên hãng để lọc nhanh danh sách.</p>
          </div>
        )}
        {items && items.length === 0 && (
          <div className="card p-10 text-center"><SearchX className="mx-auto size-9 text-mute" aria-hidden /><h2 className="mt-3 text-lg font-bold text-navy-deep">Không có xe nào khớp</h2><p className="mt-1 text-sm text-mute">Hãy nới rộng ngân sách hoặc bỏ bớt bộ lọc.</p></div>
        )}
        {items && items.length > 0 && (
          <section aria-label="Kết quả gợi ý"><h2 className="mb-4 text-xl font-bold text-navy-deep">{total > items.length ? `${items.length} xe phù hợp nhất trong ${total.toLocaleString('vi-VN')} xe khớp bộ lọc` : `${items.length} xe phù hợp`}</h2>
            <p className="-mt-2 mb-4 text-sm text-mute">Giá là giá niêm yết trên Bonbanh/Chợ Tốt, thu thập khoảng năm 2022; mỗi xe chỉ hiện một lần dù được đăng lại nhiều tin.</p>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">{items.map((c) => <CarCard key={c.id} car={c} />)}</div></section>
        )}
      </div>
    </>
  )
}
