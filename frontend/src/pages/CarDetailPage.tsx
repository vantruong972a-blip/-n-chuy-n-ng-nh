import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { CarCard } from '../components/CarCard'
import { CarImage, ImageCreditLine } from '../components/CarImage'
import { NotIntegrated } from '../components/NotIntegrated'
import { Alert, Button, ButtonLink, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { formatDecimal, formatInt, formatVnd, formatVndFull } from '../lib/format'
import { formatBrand, formatModel, valueLabel } from '../lib/labels'
import { valuationLink } from '../lib/prefill'
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
    ['Nơi rao bán (tỉnh/thành)', car.province ? valueLabel('province', car.province) : null],
    ['Màu sắc', car.color ?? null],
    ['Nguồn tin rao', car.source ? `${car.source} (giá niêm yết, thu thập khoảng 2022)` : null],
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
              <div className="aspect-[2/1]"><CarImage car={car} variant="detail" /></div>
              <ImageCreditLine car={car} />
            </div>
            <div className="card p-6">
              <p className="text-sm text-mute">Giá niêm yết của tin rao</p>
              <p className="text-4xl font-extrabold text-navy-deep">{formatVnd(car.price)}</p>
              <p className="text-sm text-mute">{formatVndFull(car.price)}</p>
              <dl className="mt-6 divide-y divide-line text-sm">
                {specs.filter(([, v]) => v).map(([k, v]) => <div key={k} className="flex justify-between gap-4 py-2.5"><dt className="text-mute">{k}</dt><dd className="font-semibold text-ink">{v}</dd></div>)}
              </dl>
              <ButtonLink to={valuationLink(car)} className="mt-6 h-12 w-full text-base">Định giá xe này bằng mô hình</ButtonLink>
              <ButtonLink to={`/goi-y?like=${encodeURIComponent(car.id)}`} variant="secondary" className="mt-3 h-12 w-full text-base">Gợi ý xe giống xe này</ButtonLink>
              <p className="mt-2 text-xs leading-relaxed text-mute">Điền sẵn thông tin xe sang trang Định giá, rồi so giá niêm yết với giá do mô hình dự đoán. Giá niêm yết khoảng năm 2022.</p>
            </div>
          </div>
        )}
        {car && car.similar && car.similar.length > 0 && (
          <section aria-labelledby="xe-tuong-tu" className="pt-4">
            <h2 id="xe-tuong-tu" className="text-xl font-bold text-navy-deep">Xe tương tự</h2>
            <p className="mb-4 mt-1 text-sm text-mute">Các tin có giá và đặc điểm gần nhất (KNN), không tính chính xe này và các tin đăng lại của nó.</p>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">{car.similar.map((c) => <CarCard key={c.id} car={c} />)}</div>
          </section>
        )}
      </div>
    </>
  )
}
