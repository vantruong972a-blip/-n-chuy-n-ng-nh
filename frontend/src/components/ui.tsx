import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { CheckCircle2, Info, Loader2, TriangleAlert, XCircle, type LucideIcon } from 'lucide-react'
import { Link, type LinkProps } from 'react-router-dom'
import { CarOutline } from './CarBlueprint'

type Variant = 'primary' | 'secondary' | 'ghost' | 'light' | 'outlineLight' | 'danger'
const VARIANT: Record<Variant, string> = {
  primary: 'bg-navy text-white hover:bg-navy-deep shadow-sm',
  secondary: 'border border-line bg-white text-ink hover:border-ink/40 hover:bg-haze',
  ghost: 'text-navy hover:bg-azure-soft',
  light: 'bg-white text-navy hover:bg-azure-soft',
  outlineLight: 'border border-white/40 bg-transparent text-white hover:border-white hover:bg-white/10',
  danger: 'bg-bad text-white hover:bg-[#8f1c12]',
}
const BASE =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-5 text-[15px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-55'

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }>(
  function Button({ variant = 'primary', loading, children, disabled, className = '', ...rest }, ref) {
    return (
      <button ref={ref} type="button" className={`${BASE} ${VARIANT[variant]} ${className}`} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
        {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
        {children}
      </button>
    )
  },
)

export function ButtonLink({ variant = 'primary', className = '', ...rest }: LinkProps & { variant?: Variant }) {
  return <Link className={`${BASE} ${VARIANT[variant]} ${className}`} {...rest} />
}

const ALERT = {
  error: { cls: 'border-bad/25 bg-bad-soft', icon: XCircle, color: 'text-bad' },
  warn: { cls: 'border-warn/25 bg-warn-soft', icon: TriangleAlert, color: 'text-warn' },
  success: { cls: 'border-ok/25 bg-ok-soft', icon: CheckCircle2, color: 'text-ok' },
  info: { cls: 'border-azure/25 bg-azure-soft', icon: Info, color: 'text-navy' },
}

export function Alert({ kind = 'info', title, children, action }: { kind?: keyof typeof ALERT; title?: string; children?: ReactNode; action?: ReactNode }) {
  const a = ALERT[kind]
  const Icon = a.icon
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={`flex gap-3 rounded-xl border p-4 text-sm ${a.cls}`}>
      <Icon className={`mt-0.5 size-5 shrink-0 ${a.color}`} aria-hidden />
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold text-ink">{title}</p>}
        {children && <div className={`${title ? 'mt-1 ' : ''}leading-relaxed text-ink-soft`}>{children}</div>}
        {action && <div className="mt-3">{action}</div>}
      </div>
    </div>
  )
}

export const Skeleton = ({ className = '' }: { className?: string }) => <div className={`skeleton ${className}`} aria-hidden />

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="grid-bg relative overflow-hidden bg-navy-deep text-white">
      <div className="glow pointer-events-none absolute inset-0" aria-hidden />
      <CarOutline className="pointer-events-none absolute -right-8 top-1/2 hidden w-[480px] -translate-y-1/2 opacity-[0.12] lg:block" />
      <div className="container-page relative py-10 sm:py-14">
        <h1 className="text-3xl font-extrabold sm:text-4xl">{title}</h1>
        {children && <p className="mt-3 max-w-2xl text-base leading-relaxed text-white/75">{children}</p>}
      </div>
    </header>
  )
}

/** Đầu thẻ có ô biểu tượng, dùng chung cho các khối nội dung chính. */
export function CardHead({ icon: Icon, title, children, aside }: { icon: LucideIcon; title: string; children?: ReactNode; aside?: ReactNode }) {
  return (
    <div className="card-head">
      <span className="icon-tile"><Icon className="size-5" aria-hidden /></span>
      <div className="min-w-0 flex-1">
        <h2 className="text-lg font-bold leading-tight text-navy-deep">{title}</h2>
        {children && <p className="mt-0.5 text-sm text-mute">{children}</p>}
      </div>
      {aside}
    </div>
  )
}

export function Stat({ label, value, note, loading }: { label: string; value?: ReactNode; note?: ReactNode; loading?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-sm text-mute">{label}</dt>
      <dd className="mt-1 text-3xl font-extrabold tracking-tight text-navy-deep sm:text-4xl">
        {loading ? <Skeleton className="h-9 w-28" /> : (value ?? '—')}
      </dd>
      {note && <p className="mt-1 text-xs leading-snug text-mute">{note}</p>}
    </div>
  )
}
