import { formatBrand, formatModel } from '../lib/labels'
import type { CarListing } from '../types/api'
import { CarPlaceholder } from './CarBlueprint'

/** Chỉ cho phép liên kết https (nguồn ảnh đến từ file CREDITS.csv, không tin tưởng tuyệt đối). */
const safeUrl = (u?: string | null) => (u && /^https:\/\//i.test(u) ? u : undefined)

/** Ảnh xe. Ảnh (nếu có) là ảnh MINH HỌA dòng xe, không phải chiếc xe trong tin rao; luôn nói rõ điều đó. */
export function CarImage({ car, variant = 'card' }: { car: CarListing; variant?: 'card' | 'detail' }) {
  const alt = `${formatBrand(car.brand)} ${formatModel(car.model)}`
  return (
    <div className="relative size-full bg-haze">
      {car.image_url ? <img src={car.image_url} alt={`Ảnh minh họa ${alt}`} loading={variant === 'card' ? 'lazy' : 'eager'} className="size-full object-cover" /> : <CarPlaceholder className="size-full" />}
      <span className="absolute bottom-2 left-2 rounded bg-white/90 px-2 py-0.5 text-xs text-mute">
        {car.image_url ? 'Ảnh minh họa dòng xe' : 'Chưa có ảnh xe'}
      </span>
    </div>
  )
}

export function ImageCreditLine({ car }: { car: CarListing }) {
  if (!car.image_url) return <p className="px-5 py-3 text-xs text-mute">Chưa có ảnh của dòng xe này.</p>
  const c = car.image_credit
  const src = safeUrl(c?.source_url)
  const lic = safeUrl(c?.license_url)
  return (
    <p className="px-5 py-3 text-xs leading-relaxed text-mute">
      Ảnh minh họa dòng xe, <strong className="font-semibold text-ink-soft">không phải xe trong tin rao</strong> và có thể khác đời hoặc phiên bản.
      {c?.author && <> Tác giả: {c.author}.</>}
      {c?.license && <> Giấy phép: {lic ? <a className="underline" href={lic} target="_blank" rel="noreferrer noopener">{c.license}</a> : c.license}.</>}
      {src && <> <a className="underline" href={src} target="_blank" rel="noreferrer noopener">Nguồn: Wikimedia Commons</a>.</>}
    </p>
  )
}
