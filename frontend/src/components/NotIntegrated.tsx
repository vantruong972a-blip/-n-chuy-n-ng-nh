import { PlugZap } from 'lucide-react'
import type { ReactNode } from 'react'

/** Trạng thái "chưa tích hợp": nói rõ backend chưa có API, không hiển thị dữ liệu giả. */
export function NotIntegrated({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="card border-dashed border-navy/30 bg-haze p-6 sm:p-8" role="status">
      <PlugZap className="size-8 text-navy" aria-hidden />
      <h2 className="mt-3 text-xl font-bold text-navy-deep">{title}</h2>
      <div className="mt-2 space-y-3 text-sm leading-relaxed text-ink-soft">{children}</div>
    </div>
  )
}
