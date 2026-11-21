import { Link } from 'react-router-dom'
import { Gauge, Fuel, Settings2, Globe2 } from 'lucide-react'
import { formatInt, formatVnd } from '../lib/format'
import { formatBrand, formatModel, valueLabel } from '../lib/labels'
import type { CarListing } from '../types/api'
import { CarImage } from './CarImage'

const Spec = ({ icon: Icon, children }: { icon: typeof Gauge; children: React.ReactNode }) => (
  <li className="inline-flex items-center gap-1.5 rounded-md bg-haze px-2.5 py-1.5 text-[13px] font-medium text-ink-soft">
    <Icon className="size-4 text-mute" aria-hidden />{children}
  </li>
)

/** Thẻ xe: ảnh minh họa chỉ hiển thị khi API trả `image_url` (kèm nguồn ở trang chi tiết); nếu không thì dùng hình giữ chỗ "chưa có ảnh". */
export function CarCard({ car }: { car: CarListing }) {
  const name = `${formatBrand(car.brand)} ${formatModel(car.model)}${car.variant ? ` ${car.variant.toUpperCase()}` : ''}`
  const score = car.match_score != null ? Math.round(car.match_score * 100) : null
  return (
    <article className="card group flex flex-col overflow-hidden transition duration-200 hover:-translate-y-0.5 hover:shadow-[var(--shadow-lift)]">
      <div className="relative aspect-[16/10]">
        <CarImage car={car} />
        {score != null && (
          <span className="absolute right-3 top-3 rounded-full bg-navy-deep/90 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur"
            title="Độ tương đồng giữa hồ sơ của bạn và xe này (0–100%), dùng để xếp hạng, không phải xác suất.">
            {score}% tương đồng
          </span>
        )}
      </div>
      <div className="flex flex-1 flex-col p-5">
        <div className="flex items-start justify-between gap-3">
          <h3 className="text-lg font-bold leading-snug text-navy-deep">{name}</h3>
          <span className="shrink-0 rounded-md border border-line px-2 py-0.5 text-sm font-semibold text-ink-soft" title="Năm sản xuất">{car.year}</span>
        </div>
        <p className="mt-2 text-[1.7rem] font-extrabold leading-none tracking-tight text-navy">{formatVnd(car.price)}</p>
        <ul className="mt-4 flex flex-wrap gap-2">
          {car.mileage_km != null && <Spec icon={Gauge}>{formatInt(car.mileage_km)} km</Spec>}
          {car.fuel_type && <Spec icon={Fuel}>{valueLabel('fuel_type', car.fuel_type)}</Spec>}
          {car.transmission && <Spec icon={Settings2}>{valueLabel('transmission', car.transmission)}</Spec>}
          {car.highlights?.includes('Xe nhập khẩu') && <Spec icon={Globe2}>Nhập khẩu</Spec>}
        </ul>
        {car.highlights && car.highlights.filter((h) => h !== 'Xe nhập khẩu').length > 0 && (
          <p className="mt-3 text-sm leading-relaxed text-ink-soft">{car.highlights.filter((h) => h !== 'Xe nhập khẩu').join(' · ')}</p>
        )}
        <div className="mt-auto pt-5">
          <Link to={`/xe/${encodeURIComponent(car.id)}`} className="inline-flex min-h-11 w-full items-center justify-center rounded-lg bg-azure-soft px-4 text-sm font-semibold text-navy transition-colors group-hover:bg-navy group-hover:text-white">
            Xem chi tiết
          </Link>
        </div>
      </div>
    </article>
  )
}
