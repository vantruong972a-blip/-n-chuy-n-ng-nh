import { useEffect, useRef, type ReactNode } from 'react'
import { Button } from './ui'

/** Hộp thoại xác nhận dùng thẻ <dialog> gốc (tự bẫy phím Tab và đóng bằng phím Esc). */
export function ConfirmDialog({
  open, title, children, confirmLabel, cancelLabel = 'Hủy', danger, loading, onConfirm, onCancel,
}: {
  open: boolean; title: string; children: ReactNode; confirmLabel: string; cancelLabel?: string
  danger?: boolean; loading?: boolean; onConfirm: () => void; onCancel: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])
  return (
    <dialog ref={ref} onClose={onCancel} onCancel={onCancel} className="m-auto w-[min(92vw,26rem)] rounded-2xl border border-line bg-white p-0 shadow-[var(--shadow-lift)]">
      <div className="p-6">
        <h2 className="text-lg font-bold text-navy-deep">{title}</h2>
        <div className="mt-2 text-sm leading-relaxed text-ink-soft">{children}</div>
        <div className="mt-6 flex justify-end gap-3">
          <Button variant="secondary" onClick={onCancel} disabled={loading}>{cancelLabel}</Button>
          <Button variant={danger ? 'danger' : 'primary'} onClick={onConfirm} loading={loading}>{confirmLabel}</Button>
        </div>
      </div>
    </dialog>
  )
}
