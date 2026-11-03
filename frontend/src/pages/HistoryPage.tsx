import { useState } from 'react'
import { Calculator, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { InputSummary } from '../components/InputSummary'
import { useToast } from '../components/toastContext'
import { Button, ButtonLink, PageHeader } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { clearHistory, removeHistory, usePredictionHistory } from '../hooks/usePredictionHistory'
import { formatDateTime, formatVnd } from '../lib/format'
import { getMeta } from '../services/predictionService'

export default function HistoryPage() {
  const list = usePredictionHistory()
  const meta = useAsync(getMeta, 'meta')
  const toast = useToast()
  const [confirm, setConfirm] = useState(false)
  return (
    <>
      <PageHeader title="Lịch sử dự đoán">Các lần định giá bạn đã thực hiện trên trình duyệt này, lưu cục bộ trên máy bạn (tối đa 50 lần gần nhất). Dữ liệu không được gửi đi đâu khác.</PageHeader>
      <div className="container-page py-10">
        {list.length === 0 ? (
          <div className="card mx-auto max-w-lg p-10 text-center">
            <Calculator className="mx-auto size-10 text-navy/40" aria-hidden />
            <h2 className="mt-4 text-xl font-bold text-navy-deep">Bạn chưa định giá xe nào</h2>
            <p className="mt-2 text-sm text-mute">Sau mỗi lần dự đoán thành công, kết quả sẽ xuất hiện ở đây để bạn so sánh.</p>
            <ButtonLink to="/dinh-gia" className="mt-6">Định giá ô tô ngay</ButtonLink>
          </div>
        ) : (
          <>
            <div className="mb-5 flex items-center justify-between gap-4">
              <p className="text-sm text-mute">{list.length} lần dự đoán</p>
              <Button variant="secondary" onClick={() => setConfirm(true)}><Trash2 className="size-4" aria-hidden />Xóa toàn bộ lịch sử</Button>
            </div>
            <ul className="grid gap-5 md:grid-cols-2">
              {list.map((e) => (
                <li key={e.id} className="card p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-3xl font-extrabold text-navy-deep">{formatVnd(e.result.price)}</p>
                      <p className="mt-0.5 text-xs text-mute">{e.result.model_label} · phiên bản {e.result.model_version} · {formatDateTime(e.createdAt)}</p>
                    </div>
                    <Button variant="ghost" className="-mr-2 px-2" aria-label="Xóa lần dự đoán này" onClick={() => { removeHistory(e.id); toast.notify('info', 'Đã xóa một mục khỏi lịch sử.') }}><Trash2 className="size-4" aria-hidden /></Button>
                  </div>
                  <div className="mt-4 border-t border-line pt-4"><InputSummary features={meta.data?.features} payload={e.payload} /></div>
                  {e.result.warnings.length > 0 && <p className="mt-3 text-xs text-warn">Có {e.result.warnings.length} cảnh báo khi dự đoán.</p>}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      <ConfirmDialog open={confirm} danger title="Xóa toàn bộ lịch sử?" confirmLabel="Xóa lịch sử" onCancel={() => setConfirm(false)}
        onConfirm={() => { clearHistory(); setConfirm(false); toast.notify('success', 'Đã xóa lịch sử dự đoán.') }}>
        {list.length} lần dự đoán sẽ bị xóa khỏi trình duyệt này. Không thể hoàn tác.
      </ConfirmDialog>
    </>
  )
}
