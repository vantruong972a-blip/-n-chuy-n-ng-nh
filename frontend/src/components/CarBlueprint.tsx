/** Bản vẽ kỹ thuật minh họa một chiếc sedan chung (tự vẽ, không phải một mẫu xe cụ thể). Các chú thích trùng với các trường nhập ở trang Định giá. */
const PARTS: { d: string; delay: number }[] = [
  { d: 'M90 252 L84 214 Q84 196 104 192 L232 180 L292 118 Q300 110 312 108 L440 106 Q470 106 492 128 L548 176 L640 190 Q676 196 682 222 L684 250 L604 254 A44 44 0 0 0 516 254 L234 254 A44 44 0 0 0 146 254 Z', delay: 0 },
  { d: 'M300 126 L258 177 L410 177 L410 122 Z M420 122 L420 177 L530 177 L478 128 Q470 122 450 122 Z', delay: 0.5 },
  { d: 'M414 122 L414 246 M650 200 L676 206 L674 214 L646 210 Z M90 204 L100 204 L100 218 L88 218 Z M440 196 L466 196', delay: 0.9 },
]

const CALLOUTS = [
  { dot: [330, 112], to: [330, 52], text: [330, 40], anchor: 'middle', label: 'Năm sản xuất', delay: 1.4 },
  { dot: [610, 192], to: [610, 100], text: [610, 88], anchor: 'middle', label: 'Dung tích động cơ', delay: 1.55 },
  { dot: [136, 198], to: [100, 128], text: [96, 116], anchor: 'middle', label: 'Nhiên liệu', delay: 1.7 },
  { dot: [190, 254], to: [190, 336], text: [190, 358], anchor: 'middle', label: 'Số km đã đi', delay: 1.85 },
  { dot: [560, 254], to: [560, 336], text: [560, 358], anchor: 'middle', label: 'Hộp số · Dẫn động', delay: 2.0 },
  { dot: [684, 240], to: [724, 300], text: [724, 322], anchor: 'middle', label: 'Hãng · Dòng xe', delay: 2.15 },
] as const

export function CarBlueprint({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 800 400" className={className} role="img" aria-label="Bản vẽ kỹ thuật một chiếc ô tô với các thông số dùng để định giá: năm sản xuất, dung tích động cơ, nhiên liệu, số km, hộp số và hãng xe">
      <g fill="none" stroke="#fff" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round">
        {PARTS.map((p, i) => (
          <path key={i} d={p.d} pathLength={1} className="draw" style={{ animationDelay: `${p.delay}s` }} strokeOpacity={i === 0 ? 1 : 0.75} />
        ))}
        {[190, 560].map((cx, i) => (
          <g key={cx}>
            <circle cx={cx} cy="254" r="42" pathLength={1} className="draw" style={{ animationDelay: `${0.6 + i * 0.15}s` }} />
            <circle cx={cx} cy="254" r="22" pathLength={1} className="draw" strokeOpacity={0.6} style={{ animationDelay: `${0.9 + i * 0.15}s` }} />
          </g>
        ))}
        <path d="M30 298 H770" stroke="#5f7bc0" strokeWidth="1" strokeDasharray="4 6" />
      </g>
      {CALLOUTS.map((c) => (
        <g key={c.label} className="callout" style={{ animationDelay: `${c.delay}s` }}>
          <path d={`M${c.dot[0]} ${c.dot[1]} L${c.to[0]} ${c.to[1]}`} stroke="#8fb0ff" strokeWidth="1.2" fill="none" />
          <circle cx={c.dot[0]} cy={c.dot[1]} r="4.5" fill="#2f6bff" stroke="#fff" strokeWidth="1.5" />
          <text x={c.text[0]} y={c.text[1]} textAnchor={c.anchor} fill="#dbe6ff" fontSize="15" fontWeight="500" fontFamily="inherit">
            {c.label}
          </text>
        </g>
      ))}
    </svg>
  )
}

/** Hình xe đơn giản dùng làm ảnh giữ chỗ khi chưa có ảnh thật (không gắn với một mẫu xe cụ thể). */
export function CarPlaceholder({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 800 400" className={className} role="img" aria-label="Chưa có ảnh xe">
      <rect width="800" height="400" fill="#f3f5f8" />
      <g fill="none" stroke="#9aa6bd" strokeWidth="3" strokeLinejoin="round" transform="translate(0 20)">
        <path d={PARTS[0].d} />
        <path d={PARTS[1].d} strokeOpacity={0.7} />
        <circle cx="190" cy="254" r="42" /><circle cx="560" cy="254" r="42" />
      </g>
    </svg>
  )
}
