import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { CheckCircle2, Info, TriangleAlert, X } from 'lucide-react'
import { ToastContext, type ToastKind } from './toastContext'

interface Toast {
  id: number
  kind: ToastKind
  text: string
}

const STYLE: Record<ToastKind, string> = {
  success: 'border-ok/30 bg-ok-soft text-ok',
  error: 'border-bad/30 bg-bad-soft text-bad',
  info: 'border-azure/30 bg-azure-soft text-navy',
}
const ICON = { success: CheckCircle2, error: TriangleAlert, info: Info }

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const dismiss = useCallback((id: number) => setItems((l) => l.filter((t) => t.id !== id)), [])
  const notify = useCallback(
    (kind: ToastKind, text: string) => {
      const id = Date.now() + Math.random()
      setItems((l) => [...l.slice(-3), { id, kind, text }])
      setTimeout(() => dismiss(id), kind === 'error' ? 8000 : 4500)
    },
    [dismiss],
  )
  const value = useMemo(() => ({ notify }), [notify])
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex flex-col items-center gap-2 px-4" role="status" aria-live="polite">
        {items.map((t) => {
          const Icon = ICON[t.kind]
          return (
            <div key={t.id} className={`pop-in pointer-events-auto flex max-w-md items-start gap-2.5 rounded-xl border px-4 py-3 text-sm font-medium shadow-[var(--shadow-lift)] ${STYLE[t.kind]}`}>
              <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
              <span className="text-ink">{t.text}</span>
              <button type="button" onClick={() => dismiss(t.id)} aria-label="Đóng thông báo" className="-m-1 rounded p-1 text-mute hover:text-ink">
                <X className="size-4" />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
