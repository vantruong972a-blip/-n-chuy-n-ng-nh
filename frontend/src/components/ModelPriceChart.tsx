import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatVnd } from '../lib/format'
import type { PredictionItem } from '../types/api'

/** So sánh giá do từng mô hình dự đoán cho CÙNG một xe (dữ liệu từ API). */
export function ModelPriceChart({ items, selected }: { items: PredictionItem[]; selected: string }) {
  const data = items.map((i) => ({ name: i.label, model: i.model, value: i.price }))
  return (
    <div role="img" aria-label={`Biểu đồ so sánh giá dự đoán: ${data.map((d) => `${d.name} ${formatVnd(d.value)}`).join('; ')}`} className="h-[170px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 70, top: 4, bottom: 4 }}>
          <XAxis type="number" hide domain={[0, 'dataMax']} />
          <YAxis type="category" dataKey="name" width={118} tickLine={false} axisLine={false} tick={{ fontSize: 13, fill: '#3b4250' }} />
          <Tooltip cursor={{ fill: '#f3f5f8' }} formatter={(v) => [formatVnd(Number(v)), 'Giá dự đoán']} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={22} isAnimationActive={false}>
            {data.map((d) => <Cell key={d.model} fill={d.model === selected ? '#0a2463' : '#a9bde8'} />)}
            <LabelList dataKey="value" position="right" formatter={(v) => formatVnd(Number(v))} style={{ fontSize: 13, fontWeight: 600, fill: '#0e1014' }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
