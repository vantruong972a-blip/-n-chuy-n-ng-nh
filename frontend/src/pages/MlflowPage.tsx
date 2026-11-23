import { useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { Alert, Button, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { formatDateTime, formatDecimal, formatMillions, formatPercent } from '../lib/format'
import { getMlflow } from '../services/monitoringService'
import type { MlflowRun } from '../types/api'

const ALGO: Record<string, string> = { linear_regression: 'Linear Regression', random_forest: 'Random Forest', xgboost: 'XGBoost' }
const algoName = (a?: string | null) => (a ? ALGO[a] ?? a : '—')

function Badge({ children, tone }: { children: string; tone: 'ok' | 'info' }) {
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${tone === 'ok' ? 'bg-ok-soft text-ok ring-1 ring-ok/20' : 'bg-azure-soft text-navy ring-1 ring-azure/20'}`}>{children}</span>
}

function RunsTable({ runs }: { runs: MlflowRun[] }) {
  return (
    <div className="card overflow-x-auto" role="region" aria-label="Danh sách lần huấn luyện" tabIndex={0}>
      <table className="data w-full min-w-[860px] text-left text-sm">
        <caption className="sr-only">Các lần huấn luyện ghi trong MLflow. Sai số tính bằng triệu đồng trên tập kiểm tra.</caption>
        <thead>
          <tr>
            {['Mô hình', 'Tỷ lệ chia', 'Thời điểm', 'RMSE validation (triệu ₫)', 'RMSE test (triệu ₫)', 'MAE test (triệu ₫)', 'R² test', 'Sai số trung vị', 'Trạng thái'].map((h) => <th key={h} scope="col" className="px-4 py-3 font-semibold">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {runs.map((r) => (
            <tr key={r.run_id}>
              <th scope="row" className="px-4 py-3 font-semibold text-ink">{algoName(r.params.algorithm)}</th>
              <td className="px-4 py-3">{r.split ?? '—'}</td>
              <td className="px-4 py-3 text-mute">{formatDateTime(r.started_at)}</td>
              <td className="px-4 py-3">{formatMillions(r.metrics.val_RMSE)}</td>
              <td className="px-4 py-3">{formatMillions(r.metrics.test_RMSE)}</td>
              <td className="px-4 py-3">{formatMillions(r.metrics.test_MAE)}</td>
              <td className="px-4 py-3">{r.metrics.test_R2 == null ? '—' : formatDecimal(r.metrics.test_R2, 2)}</td>
              <td className="px-4 py-3">{formatPercent(r.metrics.test_medAPE)}</td>
              <td className="space-x-1 px-4 py-3">
                {r.selected_best && <Badge tone="ok">Tốt nhất (theo validation)</Badge>}
                {r.served && <Badge tone="info">Bản phục vụ</Badge>}
                {r.status !== 'FINISHED' && <Badge tone="info">{r.status}</Badge>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function MlflowPage() {
  const q = useAsync(getMlflow, 'mlflow')
  const [version, setVersion] = useState<string>('')
  const d = q.data
  const versions = useMemo(() => [...new Set((d?.runs ?? []).map((r) => r.model_version).filter((v): v is string => !!v))], [d])
  const shown = (d?.runs ?? []).filter((r) => !(version || versions[0]) || r.model_version === (version || versions[0]))
  const mlflowUrl = `${window.location.protocol}//${window.location.hostname}:5000`
  return (
    <>
      <PageHeader title="Theo dõi huấn luyện (MLflow)">
        Mỗi lần huấn luyện được ghi lại trong MLflow: tham số, chỉ số trên tập kiểm tra và mô hình. Mô hình đang phục vụ API được đăng ký vào Model Registry với bí danh “champion”.
      </PageHeader>
      <div className="container-page lift space-y-8 pb-12">
        {q.loading && <Skeleton className="h-72" />}
        {q.error && <Alert kind="error" title="Không tải được dữ liệu MLflow" action={<Button variant="secondary" onClick={q.reload}>Thử lại</Button>}>{q.error.message}</Alert>}
        {d && !d.available && <Alert kind="warn" title="Chưa có dữ liệu MLflow">{d.reason}</Alert>}
        {d?.available && (
          <>
            <section aria-labelledby="registry" className="space-y-3">
              <h2 id="registry" className="text-xl font-bold text-navy-deep">Model Registry</h2>
              {d.registry ? (
                <div className="card divide-y divide-line overflow-hidden">
                  {d.registry.versions.map((v) => (
                    <div key={v.version} className="flex flex-wrap items-center gap-x-6 gap-y-1 px-5 py-3 text-sm">
                      <span className="font-bold text-ink">{d.registry?.name} · phiên bản {v.version}</span>
                      {v.aliases.map((a) => <Badge key={a} tone="ok">{a === 'champion' ? 'champion (đang phục vụ)' : a}</Badge>)}
                      <span className="text-mute">{algoName(v.algorithm)}</span>
                      <span className="text-mute">bản huấn luyện {v.model_version ?? '—'}</span>
                      <span className="text-mute">{formatDateTime(v.created_at)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <Alert kind="info" title="Chưa có mô hình nào được đăng ký">Model Registry sẽ có dữ liệu sau lần huấn luyện kế tiếp được nâng cấp lên phục vụ (python src\train_supervised.py).</Alert>
              )}
            </section>
            <section aria-labelledby="runs" className="space-y-3">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 id="runs" className="text-xl font-bold text-navy-deep">Các lần huấn luyện</h2>
                  <p className="mt-1 text-sm text-mute">Thí nghiệm “{d.experiment?.name}”. Mô hình được chọn theo RMSE trên tập validation; tập test chỉ để báo cáo.</p>
                </div>
                {versions.length > 1 && (
                  <label className="text-sm font-semibold text-ink">Bản huấn luyện
                    <select className="ml-2 min-h-11 rounded-lg border border-line bg-white px-3 text-sm" value={version || versions[0]} onChange={(e) => setVersion(e.target.value)}>
                      {versions.map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </label>
                )}
              </div>
              <RunsTable runs={shown} />
            </section>
            <section aria-labelledby="goc" className="card price-hero flex flex-wrap items-center justify-between gap-4 p-6 text-white">
              <div className="max-w-2xl">
                <h2 id="goc" className="text-lg font-bold text-white">Giao diện MLflow gốc</h2>
                <p className="mt-1 text-sm leading-relaxed text-white/75">Có biểu đồ so sánh, tệp kết quả và mô hình đã lưu. Cần khởi động riêng bằng <code className="rounded bg-white/15 px-1.5 py-0.5 text-xs text-white">scripts\start_mlflow.ps1</code> (hoặc <code className="rounded bg-white/15 px-1.5 py-0.5 text-xs text-white">docker compose up</code>), cổng 5000.</p>
              </div>
              <a href={mlflowUrl} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-white px-4 text-sm font-semibold text-navy hover:bg-azure-soft">Mở MLflow gốc<ExternalLink className="size-4" aria-hidden /></a>
            </section>
          </>
        )}
      </div>
    </>
  )
}
