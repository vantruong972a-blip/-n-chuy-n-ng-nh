import { Link } from 'react-router-dom'
import { Gauge, Fuel, Settings2 } from 'lucide-react'
import { formatInt, formatVnd } from '../lib/format'
import { formatBrand, formatModel, valueLabel } from '../lib/labels'
import type { CarListing } from '../types/api'
import { CarPlaceholder } from './CarBlueprint'

/** Thẻ xe: ảnh chỉ hiển thị khi API trả `image_url`; nếu không thì dùng hình giữ chỗ có ghi rõ "chưa có ảnh". */
export function CarCard({ car }: { car: CarListing }) {
  return (
    <article className="card flex flex-col overflow-hidden transition-shadow hover:shadow-[var(--shadow-lift)]">
      <div className="relative aspect-[2/1] bg-haze">
        {car.image_url ? <img src={car.image_url} alt={`${formatBrand(car.brand)} ${formatModel(car.model)} ${car.year}`} loading="lazy" className="size-full object-cover" /> : <CarPlaceholder className="size-full" />}
        {!car.image_url && <span className="absolute bottom-2 left-2 rounded bg-white/90 px-2 py-0.5 text-xs text-mute">Chưa có ảnh xe</span>}
      </div>
      <div className="flex flex-1 flex-col p-5">
        <h3 className="text-lg font-bold text-navy-deep">{formatBrand(car.brand)} {formatModel(car.model)}{car.variant ? ` ${car.variant.toUpperCase()}` : ''}</h3>
        <p className="text-sm text-mute">Đời {car.year}</p>
        <p className="mt-2 text-2xl font-extrabold text-navy-deep">{formatVnd(car.price)}</p>
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-ink-soft">
          {car.mileage_km != null && <li className="inline-flex items-center gap-1.5"><Gauge className="size-4 text-mute" aria-hidden />{formatInt(car.mileage_km)} km</li>}
          {car.fuel_type && <li className="inline-flex items-center gap-1.5"><Fuel className="size-4 text-mute" aria-hidden />{valueLabel('fuel_type', car.fuel_type)}</li>}
          {car.transmission && <li className="inline-flex items-center gap-1.5"><Settings2 className="size-4 text-mute" aria-hidden />{valueLabel('transmission', car.transmission)}</li>}
        </ul>
        {car.highlights && car.highlights.length > 0 && <p className="mt-3 text-sm text-ink-soft">{car.highlights.join(' · ')}</p>}
        <Link to={`/xe/${encodeURIComponent(car.id)}`} className="mt-5 inline-flex min-h-11 items-center justify-center rounded-lg border border-line px-4 text-sm font-semibold text-navy transition-colors hover:border-navy hover:bg-azure-soft">
          Xem chi tiết
        </Link>
      </div>
    </article>
  )
}
