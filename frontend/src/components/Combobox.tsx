import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Check, ChevronDown, X } from 'lucide-react'
import { fold } from '../lib/text'

export interface ComboOption { value: string; label: string; hint?: string }

interface Props {
  id: string
  options: ComboOption[]
  /** Giá trị đang chọn ('' nghĩa là chưa chọn). */
  value: string
  onChange: (value: string) => void
  placeholder?: string
  /** Nhãn của lựa chọn "bỏ chọn" ở đầu danh sách, ví dụ "Tất cả hãng". Bỏ trống thì không có dòng này. */
  emptyLabel?: string
  disabled?: boolean
  invalid?: boolean
  describedBy?: string
  required?: boolean
  /** Chữ hiện khi không có kết quả khớp. */
  noResults?: string
}

/**
 * Ô chọn có gõ phím: bấm để xem toàn bộ danh sách, gõ để lọc (không phân biệt hoa thường và dấu tiếng Việt),
 * dùng phím ↑ ↓ Enter Esc. Chỉ nhận giá trị nằm trong danh sách; gõ dở rồi rời ô thì quay về lựa chọn trước đó.
 */
export function Combobox({ id, options, value, onChange, placeholder, emptyLabel, disabled, invalid, describedBy, required, noResults = 'Không có kết quả khớp' }: Props) {
  const uid = useId()
  const listId = `${uid}-list`
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState<string | null>(null)   // null: đang hiển thị lựa chọn đã chốt; chuỗi: người dùng đang gõ
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)

  const selected = options.find((o) => o.value === value)
  const text = query ?? selected?.label ?? ''

  const rows: ComboOption[] = useMemo(() => {
    const q = query ? fold(query) : ''
    let list = options
    if (q) {
      const starts: ComboOption[] = []
      const inside: ComboOption[] = []
      for (const o of options) {
        const f = fold(o.label)
        if (f.startsWith(q)) starts.push(o)
        else if (f.includes(q)) inside.push(o)
      }
      list = [...starts, ...inside]
    }
    return emptyLabel && !q ? [{ value: '', label: emptyLabel }, ...list] : list
  }, [options, query, emptyLabel])

  // Đóng khi bấm ra ngoài.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) { setOpen(false); setQuery(null) }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  // Giữ dòng đang chọn trong tầm nhìn khi duyệt bằng phím.
  useEffect(() => {
    if (open) document.getElementById(`${uid}-opt-${active}`)?.scrollIntoView({ block: 'nearest' })
  }, [active, open, uid])

  function openList() {
    if (disabled) return
    setOpen(true)
    const i = rows.findIndex((r) => r.value === value)
    setActive(i >= 0 ? i : 0)
  }
  function pick(o: ComboOption) {
    onChange(o.value)
    setQuery(null)
    setOpen(false)
  }
  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'ArrowDown') { e.preventDefault(); if (!open) openList(); else setActive((a) => Math.min(a + 1, rows.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); if (!open) openList(); else setActive((a) => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter') { if (open && rows[active]) { e.preventDefault(); pick(rows[active]) } }
    else if (e.key === 'Escape') { if (open) { e.preventDefault(); setOpen(false); setQuery(null) } }
    else if (e.key === 'Tab') { setOpen(false); setQuery(null) }
  }

  const showClear = Boolean(value) && !disabled
  return (
    <div ref={rootRef} className="relative">
      <input
        ref={inputRef} id={id} type="text" role="combobox" autoComplete="off" spellCheck={false} disabled={disabled}
        aria-expanded={open} aria-controls={listId} aria-autocomplete="list" aria-haspopup="listbox"
        aria-activedescendant={open && rows[active] ? `${uid}-opt-${active}` : undefined}
        aria-invalid={invalid ? true : undefined} aria-describedby={describedBy} aria-required={required || undefined}
        className="input pr-[4.5rem]" placeholder={placeholder} value={text}
        onFocus={(e) => { openList(); e.currentTarget.select() }}
        onClick={openList}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0) }}
        onKeyDown={onKey}
      />
      <div className="absolute inset-y-0 right-1.5 flex items-center gap-0.5">
        {showClear && (
          <button type="button" aria-label="Bỏ lựa chọn" className="grid size-8 place-items-center rounded-md text-mute hover:bg-haze hover:text-ink"
            onMouseDown={(e) => e.preventDefault()} onClick={() => { onChange(''); setQuery(null); inputRef.current?.focus() }}>
            <X className="size-4" aria-hidden />
          </button>
        )}
        <button type="button" tabIndex={-1} aria-hidden disabled={disabled} className="grid size-8 place-items-center rounded-md text-mute hover:text-ink"
          onMouseDown={(e) => e.preventDefault()} onClick={() => { if (open) { setOpen(false); setQuery(null) } else { inputRef.current?.focus(); openList() } }}>
          <ChevronDown className={`size-4 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
        </button>
      </div>
      {open && (
        <ul id={listId} role="listbox" className="absolute left-0 right-0 z-30 mt-1.5 max-h-72 overflow-auto rounded-xl border border-line bg-white p-1 shadow-[var(--shadow-lift)]">
          {rows.length === 0 && <li className="px-3 py-3 text-sm text-mute" role="presentation">{noResults}</li>}
          {rows.map((o, i) => (
            <li
              key={o.value || '__empty'} id={`${uid}-opt-${i}`} role="option" aria-selected={o.value === value}
              className={`flex min-h-10 cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2 text-[15px] ${i === active ? 'bg-azure-soft text-navy-deep' : 'text-ink'} ${o.value === '' ? 'text-mute' : ''}`}
              onMouseEnter={() => setActive(i)} onMouseDown={(e) => e.preventDefault()} onClick={() => pick(o)}
            >
              <span className={o.value === value && o.value !== '' ? 'font-semibold' : ''}>{o.label}</span>
              <span className="flex shrink-0 items-center gap-2 text-xs text-mute">
                {o.hint}
                {o.value === value && o.value !== '' && <Check className="size-4 text-navy" aria-hidden />}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
