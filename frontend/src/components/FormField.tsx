import { useId } from 'react'
import { FIELD_HELP, formatBrand, valueLabel } from '../lib/labels'
import { num, sanitizeNumberInput } from '../lib/validation'
import type { FeatureMeta, MetaResponse } from '../types/api'

interface Props {
  feature: FeatureMeta
  meta: MetaResponse
  value: string
  error?: string
  disabled?: boolean
  brand: string
  onChange: (name: string, value: string) => void
}

export function FormField({ feature: f, meta, value, error, disabled, brand, onChange }: Props) {
  const uid = useId()
  const id = `f-${f.name}`
  const help = FIELD_HELP[f.name]
  const describedBy = `${uid}-hint${error ? ` ${uid}-err` : ''}`
  const common = { id, name: f.name, disabled, 'aria-invalid': error ? (true as const) : undefined, 'aria-describedby': describedBy, 'aria-required': f.required || undefined }

  let control
  if (f.type === 'number') {
    control = (
      <div className="relative">
        <input
          {...common} type="text" inputMode={f.integer ? 'numeric' : 'decimal'} autoComplete="off" className={`input ${help?.unit ? 'pr-12' : ''}`}
          placeholder={help?.placeholder} value={value}
          onChange={(e) => { const next = sanitizeNumberInput(e.target.value, f.integer); if (next !== null) onChange(f.name, next) }}
        />
        {help?.unit && <span className="pointer-events-none absolute inset-y-0 right-3 grid place-items-center text-sm text-mute">{help.unit}</span>}
      </div>
    )
  } else if (f.name === 'model' || f.free_text) {
    const listId = `${id}-list`
    const options = f.name === 'model' ? (meta.brand_models[brand] ?? []) : f.options
    control = (
      <>
        <input {...common} type="text" autoComplete="off" maxLength={60} list={listId} className="input" placeholder={help?.placeholder} value={value}
          onChange={(e) => onChange(f.name, e.target.value)} />
        <datalist id={listId}>{options.slice(0, 200).map((o) => <option key={o} value={o} />)}</datalist>
      </>
    )
  } else {
    const opts = f.options.map((o) => ({ value: o, label: valueLabel(f.name, o) }))
    if (f.name === 'brand') opts.forEach((o) => (o.label = formatBrand(o.value)))
    opts.sort((a, b) => a.label.localeCompare(b.label, 'vi'))
    control = (
      <select {...common} className="input" value={value} onChange={(e) => onChange(f.name, e.target.value)}>
        <option value="">{f.required ? `Chọn ${f.label.toLowerCase()}` : 'Không chọn'}</option>
        {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    )
  }

  const range = f.type === 'number' && f.min != null && f.max != null ? ` (${num(f, f.min)} – ${num(f, f.max)})` : ''
  return (
    <div>
      <label htmlFor={id} className="label">
        {f.label}
        {f.required ? <span className="ml-0.5 text-bad" aria-hidden> *</span> : <span className="ml-1.5 text-xs font-normal text-mute">tùy chọn</span>}
      </label>
      {control}
      <p id={`${uid}-hint`} className="mt-1 text-xs leading-snug text-mute">{help?.hint}{range}</p>
      {error && <p id={`${uid}-err`} role="alert" className="mt-1 text-sm font-medium text-bad">{error}</p>}
    </div>
  )
}
