import { useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { RefreshCw } from 'lucide-react'
import { useToast } from '../components/toastContext'
import { Alert, Button, PageHeader, Skeleton, Stat } from '../components/ui'
import { toApiError, useAsync } from '../hooks/useAsync'
import { formatDateTime, formatDecimal, formatInt, formatMillions, formatPercent, toMillions } from '../lib/format'
import { fieldLabelByName } from '../lib/labels'
import { getHealth, getMeta } from '../services/predictionService'
import { getExperiments, getMetrics, getModelInfo, getMonitoring } from '../services/monitoringService'

const C = { navy: '#0a2463', azure: '#2f6bff', soft: '#a9bde8', bad: '#b42318', grid: '#e1e5eb' }
const MODEL_COLOR: Record<string, string> = { linear_regression: '#a9bde8', random_forest: '#2f6bff', xgboost: '#0a2463' }

function Section({ title, desc, children, id }: { title: string; desc?: string; children: ReactNode; id: string }) {
  return (
    <section aria-labelledby={id} className="pt-10">
      <h2 id={id} className="text-2xl font-extrabold text-navy-deep">{title}</h2>
      {desc && <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-soft">{desc}</p>}
      <div className="mt-6">{children}</div>
    </section>
  )
}

function Panel({ title, children, className = '' }: { title: string; children: ReactNode; className?: string }) {
  return <div className={`card p-5 ${className}`}><h3 className="mb-4 border-b border-line/70 pb-3 text-sm font-bold text-navy-deep">{title}</h3>{children}</div>
}

function Fail({ err, retry, what }: { err: { message: string; status: number }; retry: () => void; what: string }) {
  return (
    <Alert kind={err.status === 0 ? 'error' : 'warn'} title={err.status === 0 ? 'Không kết nối được máy chủ' : `Chưa lấy được ${what}`} action={<Button variant="secondary" onClick={retry}>Thử lại</Button>}>
      {err.status === 404 ? `Máy chủ chưa cung cấp ${what} (chức năng chưa được tích hợp ở phiên bản backend này).` : err.message}
    </Alert>
  )
}

const tipStyle = { borderRadius: 10, border: '1px solid #e1e5eb', fontSize: 13 }

function Performance() {
  const m = useAsync(getMetrics, 'metrics')
  const ex = useAsync(getExperiments, 'experiments')
  if (m.error) return <Fail err={m.error} retry={m.reload} what="chỉ số mô hình" />
  if (m.loading || !m.data) return <Skeleton className="h-80" />
  const d = m.data
  const bars = d.models.map((x) => ({ name: x.label, MAE: toMillions(x.MAE ?? 0), RMSE: toMillions(x.RMSE ?? 0) }))
  const splits = [...new Set((ex.data ?? []).map((r) => r.split))].sort((a, b) => Number(b.split('/')[0]) - Number(a.split('/')[0]))
  const bySplit = splits.map((s) => ({ split: s, ...Object.fromEntries((ex.data ?? []).filter((r) => r.split === s).map((r) => [r.model, toMillions(r.RMSE)])) }))
  const labels = Object.fromEntries(d.models.map((x) => [x.model, x.label]))
  return (
    <div className="space-y-5">
      <div className="card overflow-x-auto">
        <table className="data w-full min-w-[640px] text-left text-sm">
          <caption className="sr-only">Chỉ số các mô hình trên tập kiểm tra giữ riêng</caption>
          <thead className="border-b border-line"><tr>
            <th scope="col" className="px-4 py-3 font-medium">Mô hình</th><th scope="col" className="px-4 py-3 font-medium">Cấu hình</th>
            <th scope="col" className="px-4 py-3 text-right font-medium">MAE (triệu)</th><th scope="col" className="px-4 py-3 text-right font-medium">RMSE (triệu)</th>
            <th scope="col" className="px-4 py-3 text-right font-medium">R²</th><th scope="col" className="px-4 py-3 text-right font-medium">MAPE</th><th scope="col" className="px-4 py-3 text-right font-medium">Sai số TB trung vị</th>
          </tr></thead>
          <tbody>
            {d.models.map((x) => (
              <tr key={x.model} className="border-b border-line last:border-0">
                <th scope="row" className="px-4 py-3 font-semibold text-ink">{x.label}{x.model === d.best_model && <span className="ml-2 rounded-full bg-ok-soft px-2 py-0.5 text-xs font-semibold text-ok">Tốt nhất theo kiểm định</span>}</th>
                <td className="px-4 py-3 text-ink-soft">{x.config_used === 'tuned' ? 'Đã tinh chỉnh' : x.config_used === 'baseline' ? 'Mặc định' : '—'}</td>
                <td className="px-4 py-3 text-right tabular-nums">{formatMillions(x.MAE)}</td><td className="px-4 py-3 text-right tabular-nums">{formatMillions(x.RMSE)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{x.R2 == null ? '—' : formatDecimal(x.R2, 2)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{formatPercent(x.MAPE)}</td><td className="px-4 py-3 text-right tabular-nums">{formatPercent(x.medAPE)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-relaxed text-mute">
        Đo trên {formatInt(d.n_test)} xe kiểm tra giữ riêng (chia {d.split}; {d.split_strategy ?? 'chia theo nhóm tin rao'}). Mô hình tốt nhất được chọn theo RMSE trên tập kiểm định, không chọn theo tập kiểm tra. Đơn vị tiền: triệu đồng. {d.unit_note}
      </p>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Sai số tuyệt đối trung bình (MAE) và căn sai số bình phương (RMSE), triệu đồng">
          <div className="h-64" role="img" aria-label={`So sánh MAE và RMSE: ${bars.map((b) => `${b.name} MAE ${formatDecimal(b.MAE)}, RMSE ${formatDecimal(b.RMSE)}`).join('; ')} triệu đồng`}>
            <ResponsiveContainer width="100%" height="100%"><BarChart data={bars} margin={{ left: 0, right: 8, top: 8 }}>
              <CartesianGrid stroke={C.grid} vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 12 }} tickLine={false} /><YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} width={44} />
              <Tooltip contentStyle={tipStyle} formatter={(v) => [`${formatDecimal(Number(v))} triệu`]} /><Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="MAE" fill={C.soft} radius={[4, 4, 0, 0]} isAnimationActive={false} /><Bar dataKey="RMSE" fill={C.navy} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart></ResponsiveContainer>
          </div>
        </Panel>
        <Panel title="RMSE theo tỷ lệ chia dữ liệu (huấn luyện / kiểm tra), triệu đồng">
          {ex.error ? <p className="text-sm text-mute">Chưa lấy được kết quả các tỷ lệ chia: {ex.error.message}</p> : ex.loading ? <Skeleton className="h-64" /> : (
            <div className="h-64" role="img" aria-label="So sánh RMSE của ba mô hình theo từng tỷ lệ chia dữ liệu">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={bySplit} margin={{ left: 0, right: 8, top: 8 }}>
                <CartesianGrid stroke={C.grid} vertical={false} /><XAxis dataKey="split" tick={{ fontSize: 12 }} tickLine={false} /><YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} width={44} />
                <Tooltip contentStyle={tipStyle} formatter={(v, n) => [`${formatDecimal(Number(v))} triệu`, labels[String(n)] ?? String(n)]} /><Legend formatter={(v) => labels[String(v)] ?? String(v)} wrapperStyle={{ fontSize: 12 }} />
                {d.models.map((x) => <Bar key={x.model} dataKey={x.model} fill={MODEL_COLOR[x.model] ?? C.azure} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
              </BarChart></ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}

function Operations() {
  const mon = useAsync(getMonitoring, 'monitoring')
  if (mon.error) return <Fail err={mon.error} retry={mon.reload} what="số liệu giám sát" />
  if (mon.loading || !mon.data) return <div className="grid gap-5 sm:grid-cols-4"><Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" /></div>
  const d = mon.data
  return (
    <div className="space-y-5">
      <dl className="card grid grid-cols-2 gap-x-6 gap-y-6 p-6 lg:grid-cols-4">
        <Stat label="Lượt dự đoán thành công" value={formatInt(d.n_predictions)} note="Đếm từ nhật ký dự đoán" />
        <Stat label="Yêu cầu bị lỗi" value={formatInt(d.n_errors)} note="Gồm dữ liệu đầu vào sai (mã 422)" />
        <Stat label="Độ trễ trung bình" value={d.latency_ms.mean == null ? '—' : `${formatDecimal(d.latency_ms.mean)} ms`} note="Thời gian máy chủ xử lý một dự đoán" />
        <Stat label="Độ trễ p95" value={d.latency_ms.p95 == null ? '—' : `${formatDecimal(d.latency_ms.p95)} ms`} note="95% yêu cầu nhanh hơn mức này" />
      </dl>
      <Panel title="Lỗi gần đây (tối đa 5)">
        {d.recent_errors.length === 0 ? <p className="text-sm text-mute">Chưa có yêu cầu lỗi nào được ghi nhận.</p> : (
          <div className="overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm"><thead className="text-xs text-mute"><tr><th scope="col" className="py-2 pr-4 font-medium">Thời điểm</th><th scope="col" className="py-2 pr-4 font-medium">Điểm cuối API</th><th scope="col" className="py-2 pr-4 font-medium">Mã</th><th scope="col" className="py-2 font-medium">Chi tiết</th></tr></thead>
            <tbody>{d.recent_errors.map((e, i) => <tr key={i} className="border-t border-line"><td className="py-2 pr-4 whitespace-nowrap">{formatDateTime(e.ts)}</td><td className="py-2 pr-4 font-mono text-xs">{e.endpoint}</td><td className="py-2 pr-4">{e.status}</td><td className="py-2 text-ink-soft">{e.detail}</td></tr>)}</tbody></table></div>
        )}
      </Panel>
    </div>
  )
}

const DRIFT_STATUS: Record<string, { text: string; cls: string }> = {
  no_signal: { text: 'Ổn định', cls: 'bg-ok-soft text-ok' },
  data_drift_signal: { text: 'Có dấu hiệu lệch dữ liệu', cls: 'bg-bad-soft text-bad' },
  insufficient_samples: { text: 'Chưa đủ dữ liệu để kết luận', cls: 'bg-warn-soft text-warn' },
  no_reference: { text: 'Chưa có dữ liệu tham chiếu', cls: 'bg-warn-soft text-warn' },
}

function DataMonitoring() {
  const mon = useAsync(getMonitoring, 'monitoring')
  const meta = useAsync(getMeta, 'meta')
  if (mon.error) return <Fail err={mon.error} retry={mon.reload} what="số liệu giám sát dữ liệu" />
  if (mon.loading || !mon.data) return <Skeleton className="h-72" />
  const { drift, feedback } = mon.data
  const st = DRIFT_STATUS[drift.status] ?? { text: drift.status, cls: 'bg-haze text-ink' }
  const psi = Object.entries(drift.features ?? {}).map(([k, v]) => ({ name: fieldLabelByName(k, meta.data?.features), psi: v.psi, flag: v.flag })).sort((a, b) => b.psi - a.psi)
  const thr = drift.threshold_psi ?? 0.2
  const daily = Object.entries(feedback.mae_by_day ?? {}).map(([day, v]) => ({ day, mae: toMillions(v) }))
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Panel title="Lệch phân phối dữ liệu đầu vào (PSI)">
        <div className="flex flex-wrap items-center gap-2"><span className={`rounded-full px-3 py-1 text-sm font-semibold ${st.cls}`}>{st.text}</span><span className="text-xs text-mute">{formatInt(drift.n_samples ?? 0)} yêu cầu gần nhất</span></div>
        {drift.message && <p className="mt-3 text-sm text-ink-soft">{drift.message}</p>}
        {psi.length > 0 ? (
          <>
            <div className="mt-3" style={{ height: Math.max(180, psi.length * 30 + 30) }} role="img" aria-label={`Chỉ số PSI theo đặc trưng; ngưỡng ${thr}. ${drift.drifted_features?.length ? `Vượt ngưỡng: ${drift.drifted_features.join(', ')}` : 'Không đặc trưng nào vượt ngưỡng'}`}>
              <ResponsiveContainer width="100%" height="100%"><BarChart data={psi} layout="vertical" margin={{ left: 0, right: 24 }}>
                <CartesianGrid stroke={C.grid} horizontal={false} /><XAxis type="number" tick={{ fontSize: 12 }} /><YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tipStyle} formatter={(v) => [formatDecimal(Number(v), 2), 'PSI']} />
                <ReferenceLine x={thr} stroke={C.bad} strokeDasharray="4 4" label={{ value: `Ngưỡng ${String(thr).replace('.', ',')}`, position: 'top', fontSize: 11, fill: C.bad }} />
                <Bar dataKey="psi" radius={[0, 4, 4, 0]} barSize={16} isAnimationActive={false}>{psi.map((p) => <Cell key={p.name} fill={p.flag ? C.bad : C.navy} />)}</Bar>
              </BarChart></ResponsiveContainer>
            </div>
            <p className="mt-2 text-sm text-ink-soft">{drift.drifted_features?.length ? <>Đặc trưng vượt ngưỡng: <strong>{drift.drifted_features.map((n) => fieldLabelByName(n, meta.data?.features)).join(', ')}</strong>.</> : 'Không có đặc trưng nào vượt ngưỡng.'}</p>
          </>
        ) : !drift.message && <p className="mt-3 text-sm text-mute">Chưa có PSI để hiển thị.</p>}
        {drift.note && <p className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-mute">{drift.note}</p>}
      </Panel>
      <Panel title="Phản hồi giá thực tế và sai số thực tế">
        {feedback.n === 0 ? <p className="text-sm leading-relaxed text-ink-soft">{feedback.message ?? 'Chưa có phản hồi giá thực tế.'}<br /><span className="text-mute">Gửi giá thực tế ở khung kết quả trang “Định giá ô tô” để bắt đầu theo dõi.</span></p> : (
          <>
            <dl className="grid grid-cols-3 gap-4"><Stat label="Số phản hồi" value={formatInt(feedback.n)} /><Stat label="MAE thực tế" value={`${formatMillions(feedback.MAE)} tr`} /><Stat label="RMSE thực tế" value={`${formatMillions(feedback.RMSE)} tr`} /></dl>
            {daily.length > 0 && (
              <div className="mt-4 h-52" role="img" aria-label="Sai số tuyệt đối trung bình theo ngày">
                <ResponsiveContainer width="100%" height="100%"><LineChart data={daily} margin={{ left: 0, right: 12, top: 8 }}><CartesianGrid stroke={C.grid} vertical={false} /><XAxis dataKey="day" tick={{ fontSize: 11 }} /><YAxis tick={{ fontSize: 12 }} width={44} axisLine={false} tickLine={false} /><Tooltip contentStyle={tipStyle} formatter={(v) => [`${formatDecimal(Number(v))} triệu`, 'MAE']} /><Line type="monotone" dataKey="mae" stroke={C.navy} strokeWidth={2} dot isAnimationActive={false} /></LineChart></ResponsiveContainer>
              </div>
            )}
          </>
        )}
      </Panel>
    </div>
  )
}

function Registry() {
  const info = useAsync(getModelInfo, 'model-info')
  const metrics = useAsync(getMetrics, 'metrics')
  if (info.error) return <Fail err={info.error} retry={info.reload} what="thông tin phiên bản mô hình" />
  if (info.loading || !info.data) return <Skeleton className="h-56" />
  const i = info.data
  return (
    <div className="space-y-5">
      <div className="card overflow-x-auto">
        <table className="data w-full min-w-[720px] text-left text-sm">
          <caption className="sr-only">Các mô hình đang được phục vụ</caption>
          <thead className="border-b border-line"><tr>
            <th scope="col" className="px-4 py-3 font-medium">Tên mô hình</th><th scope="col" className="px-4 py-3 font-medium">Phiên bản</th><th scope="col" className="px-4 py-3 font-medium">Huấn luyện lúc</th>
            <th scope="col" className="px-4 py-3 font-medium">Trạng thái</th><th scope="col" className="px-4 py-3 text-right font-medium">RMSE (triệu)</th><th scope="col" className="px-4 py-3 text-right font-medium">R²</th>
          </tr></thead>
          <tbody>
            {i.serving_models.map((k) => {
              const mm = metrics.data?.models.find((x) => x.model === k)
              return (
                <tr key={k} className="border-b border-line last:border-0">
                  <th scope="row" className="px-4 py-3 font-semibold">{i.models[k]?.label ?? k}</th>
                  <td className="px-4 py-3 font-mono text-xs">{i.model_version}</td>
                  <td className="px-4 py-3 whitespace-nowrap">{formatDateTime(i.trained_at)}</td>
                  <td className="px-4 py-3"><span className="rounded-full bg-ok-soft px-2.5 py-0.5 text-xs font-semibold text-ok">Đang phục vụ</span>{k === i.best_model && <span className="ml-1.5 rounded-full bg-azure-soft px-2.5 py-0.5 text-xs font-semibold text-navy">Mặc định</span>}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{formatMillions(mm?.RMSE)}</td><td className="px-4 py-3 text-right tabular-nums">{mm?.R2 == null ? '—' : formatDecimal(mm.R2, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Dữ liệu và cách huấn luyện">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-mute">Phiên bản dữ liệu</dt><dd className="font-mono text-xs">{i.data_version}</dd>
            <dt className="text-mute">Số bản ghi thô</dt><dd>{i.n_raw != null ? formatInt(i.n_raw) : '—'}</dd>
            <dt className="text-mute">Sau làm sạch</dt><dd>{i.n_rows_clean != null ? formatInt(i.n_rows_clean) : formatInt(i.n_train + i.n_test)}</dd>
            <dt className="text-mute">Huấn luyện / kiểm tra</dt><dd>{formatInt(i.n_train)} / {formatInt(i.n_test)} (chia {i.split})</dd>
            <dt className="text-mute">Cách chia</dt><dd>{i.split_strategy ?? '—'}</dd>
            <dt className="text-mute">Biến mục tiêu</dt><dd>{i.target} ({i.target_transform ?? 'không biến đổi'})</dd>
            <dt className="text-mute">Tinh chỉnh</dt><dd>{i.tuned ? `Có (${i.tuning_iterations ?? '?'} lần thử, ${i.cv ?? ''})` : 'Không'}</dd>
            <dt className="text-mute">Tiêu chí chọn</dt><dd>{i.selection_criterion}</dd>
          </dl>
        </Panel>
        <Panel title="Thư viện và quản lý phiên bản">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            {Object.entries(i.libraries).map(([k, v]) => <><dt key={`k${k}`} className="text-mute">{k}</dt><dd key={`v${k}`}>{v ?? 'không dùng'}</dd></>)}
          </dl>
          <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-mute">
            Mô hình được lưu theo thư mục phiên bản (<code>models/versions</code>) và chỉ được thay khi qua kiểm tra; bản cũ giữ ở <code>models/archive</code>. Các lần chạy được ghi bằng MLflow Tracking, và mô hình đang phục vụ được đăng ký vào MLflow Model Registry (bí danh “champion”); xem ở trang <a className="font-semibold text-navy underline" href="/mlflow">Theo dõi MLflow</a>. Cột “Trạng thái” ở đây phản ánh mô hình đang được API nạp.
          </p>
        </Panel>
      </div>
    </div>
  )
}

function Health({ tick }: { tick: number }) {
  const h = useAsync(getHealth, `health-${tick}`)
  if (h.loading) return <Skeleton className="h-8 w-48" />
  if (h.error) return <span className="inline-flex items-center gap-2 rounded-full bg-bad-soft px-3 py-1.5 text-sm font-semibold text-bad"><span className="size-2 rounded-full bg-bad" />{h.error.status === 503 ? 'Mô hình chưa sẵn sàng' : 'Không kết nối được máy chủ'}</span>
  return <span className="inline-flex items-center gap-2 rounded-full bg-ok-soft px-3 py-1.5 text-sm font-semibold text-ok"><span className="size-2 rounded-full bg-ok" />Máy chủ hoạt động · {h.data?.models?.length ?? 0} mô hình</span>
}

export default function AdminPage() {
  const toast = useToast()
  const [tick, setTick] = useState(0)
  const [busy, setBusy] = useState(false)
  async function refresh() {
    setBusy(true)
    try { await getHealth(); setTick((t) => t + 1); toast.notify('success', 'Đã làm mới số liệu.') } catch (e) { setTick((t) => t + 1); toast.notify('error', toApiError(e).message) } finally { setBusy(false) }
  }
  return (
    <>
      <PageHeader title="Quản trị hệ thống">Theo dõi chất lượng mô hình, tình trạng máy chủ và độ lệch dữ liệu. Trang này dành cho người vận hành đồ án; số liệu lấy trực tiếp từ API, không có dữ liệu minh họa.</PageHeader>
      <div className="container-page lift pb-12">
        <div className="card flex flex-wrap items-center justify-between gap-3 px-5 py-4">
          <Health tick={tick} />
          <Button variant="secondary" onClick={refresh} loading={busy}><RefreshCw className="size-4" aria-hidden />Làm mới số liệu</Button>
        </div>
        {/* key={tick}: làm mới = tải lại cả bốn khối */}
        <Section id="hieu-suat" title="Hiệu suất mô hình" desc="So sánh ba mô hình trên tập kiểm tra giữ riêng. MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt."><Performance key={tick} /></Section>
        <Section id="he-thong" title="Theo dõi hệ thống" desc="Số liệu từ nhật ký của API: lượt dự đoán, lỗi và độ trễ."><Operations key={tick} /></Section>
        <Section id="du-lieu" title="Giám sát dữ liệu" desc="PSI so sánh phân phối dữ liệu người dùng nhập với dữ liệu huấn luyện; vượt ngưỡng 0,2 là dấu hiệu lệch đáng kể. Chỉ có phản hồi giá thực tế mới cho biết mô hình có thực sự kém đi hay không."><DataMonitoring key={tick} /></Section>
        <Section id="kho-mo-hinh" title="Kho mô hình" desc="Phiên bản và kết quả đánh giá của các mô hình đang được phục vụ."><Registry key={tick} /></Section>
      </div>
    </>
  )
}
