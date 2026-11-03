import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { CarPlaceholder } from '../components/CarBlueprint'
import { NotIntegrated } from '../components/NotIntegrated'
import { Alert, Button, ButtonLink, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { formatDecimal, formatInt, formatVnd, formatVndFull } from '../lib/format'
import { formatBrand, formatModel, valueLabel } from '../lib/labels'
import { getCarDetail, isCarDetailAvailable } from '../services/recommendationService'

export default function CarDetailPage() {
  const { id = '' } = useParams()
  const state = useAsync(async () => ((await isCarDetailAvailable()) ? { ready: true as const, car: await getCarDetail(id) } : { ready: false as const }), id)
  const car = state.data?.ready ? state.data.car : null
  const specs: [string, string | null][] = car ? [
    ['Năm sản xuất', String(car.year)],
    ['Số km đã đi', car.mileage_km != null ? `${formatInt(car.mileage_km)} km` : null],
    ['Dung tích động cơ', car.engine_capacity != null ? `${formatDecimal(car.engine_capacity, 2)} lít` : null],
    ['Loại nhiên liệu', car.fuel_type ? valueLabel('fuel_type', car.fuel_type) : null],
    ['Hộp số', car.transmission ? valueLabel('transmission', car.transmission) : null],
    ['Kiểu dáng', car.body_type ? valueLabel('body_type', car.body_type) : null],
    ['Số chỗ ngồi', car.seats != null ? `${car.seats} chỗ` : null],
    ['Xuất xứ', car.origin ? valueLabel('origin', car.origin) : null],
    ['Tỉnh/thành phố', car.province ? valueLabel('province', car.province) : null],
  ] : []
  return (
    <>
      <PageHeader title={car ? `${formatBrand(car.brand)} ${formatModel(car.model)} ${car.year}` : 'Chi tiết ô tô'} />
      <div className="container-page space-y-6 py-10">
        <Link to="/goi-y" className="inline-flex min-h-11 items-center gap-1.5 text-sm font-semibold text-navy hover:underline"><ArrowLeft className="size-4" aria-hidden />Quay lại gợi ý xe</Link>
        {state.loading && <div className="grid gap-6 lg:grid-cols-2"><Skeleton className="aspect-[2/1]" /><Skeleton className="h-64" /></div>}
        {state.error && <Alert kind="error" title={state.error.status === 404 ? 'Không tìm thấy xe này' : 'Không tải được thông tin xe'} action={<Button variant="secondary" onClick={state.reload}>Thử lại</Button>}>{state.error.message}</Alert>}
        {state.data && !state.data.ready && (
          <NotIntegrated title="Trang chi tiết xe chưa được tích hợp">
            <p>Backend chưa có API <code className="rounded bg-white px-1.5 py-0.5 text-xs">GET /api/cars/&#123;id&#125;</code> và chưa có kho ảnh xe, nên không thể hiển thị xe mã <strong>{id}</strong>. Trang sẽ tự hoạt động khi API này xuất hiện.</p>
            <ButtonLink to="/dinh-gia" variant="secondary">Định giá xe của bạn</ButtonLink>
          </NotIntegrated>
        )}
        {car && (
          <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
            <div className="card overflow-hidden">
              <div className="aspect-[2/1] bg-haze">{car.image_url ? <img src={car.image_url} alt={`${formatBrand(car.brand)} ${formatModel(car.model)}`} className="size-full object-cover" /> : <CarPlaceholder className="size-full" />}</div>
              {!car.image_url && <p className="px-5 py-3 text-xs text-mute">Chưa có ảnh thật của xe này.</p>}
            </div>
            <div className="card p-6">
              <p className="text-sm text-mute">Giá tham khảo</p>
              <p className="text-4xl font-extrabold text-navy-deep">{formatVnd(car.price)}</p>
              <p className="text-sm text-mute">{formatVndFull(car.price)}</p>
              <dl className="mt-6 divide-y divide-line text-sm">
                {specs.filter(([, v]) => v).map(([k, v]) => <div key={k} className="flex justify-between gap-4 py-2.5"><dt className="text-mute">{k}</dt><dd className="font-semibold text-ink">{v}</dd></div>)}
              </dl>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
