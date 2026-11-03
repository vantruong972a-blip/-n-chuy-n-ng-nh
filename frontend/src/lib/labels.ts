import type { FeatureMeta } from '../types/api'

/** Nhãn tiếng Việt cho giá trị phân loại mà backend trả về dưới dạng mã. Không có trong bảng thì hiển thị nguyên giá trị. */
const VALUE_LABELS: Record<string, Record<string, string>> = {
  body_type: {
    sedan: 'Sedan', suv_crossover: 'SUV / Crossover', hatchback: 'Hatchback', pickup: 'Bán tải', van: 'Xe van',
    mpv: 'Đa dụng (MPV)', coupe: 'Coupe 2 cửa', convertible: 'Mui trần', other: 'Kiểu dáng khác',
  },
  transmission: { automatic: 'Số tự động', manual: 'Số sàn', semi_automatic: 'Số bán tự động' },
  fuel_type: { gasoline: 'Xăng', diesel: 'Dầu diesel', hybrid: 'Hybrid (xăng – điện)', electric: 'Điện' },
  drivetrain: { fwd: 'Cầu trước (FWD)', rwd: 'Cầu sau (RWD)', awd: 'Toàn thời gian (AWD)', '4wd': 'Hai cầu (4WD)' },
  origin: { domestic: 'Lắp ráp trong nước', imported: 'Nhập khẩu' },
}

const PROVINCES: Record<string, string> = {
  'ha-noi': 'Hà Nội', 'tp-ho-chi-minh': 'TP. Hồ Chí Minh', 'hai-phong': 'Hải Phòng', 'da-nang': 'Đà Nẵng', 'can-tho': 'Cần Thơ',
  'an-giang': 'An Giang', 'ba-ria-vung-tau': 'Bà Rịa – Vũng Tàu', 'bac-giang': 'Bắc Giang', 'bac-kan': 'Bắc Kạn',
  'bac-lieu': 'Bạc Liêu', 'bac-ninh': 'Bắc Ninh', 'ben-tre': 'Bến Tre', 'binh-dinh': 'Bình Định', 'binh-duong': 'Bình Dương',
  'binh-phuoc': 'Bình Phước', 'binh-thuan': 'Bình Thuận', 'ca-mau': 'Cà Mau', 'cao-bang': 'Cao Bằng', 'dak-lak': 'Đắk Lắk',
  'dak-nong': 'Đắk Nông', 'dien-bien': 'Điện Biên', 'dong-nai': 'Đồng Nai', 'dong-thap': 'Đồng Tháp', 'gia-lai': 'Gia Lai',
  'ha-giang': 'Hà Giang', 'ha-nam': 'Hà Nam', 'ha-tinh': 'Hà Tĩnh', 'hai-duong': 'Hải Dương', 'hau-giang': 'Hậu Giang',
  'hoa-binh': 'Hòa Bình', 'hung-yen': 'Hưng Yên', 'khanh-hoa': 'Khánh Hòa', 'kien-giang': 'Kiên Giang', 'kon-tum': 'Kon Tum',
  'lai-chau': 'Lai Châu', 'lam-dong': 'Lâm Đồng', 'lang-son': 'Lạng Sơn', 'lao-cai': 'Lào Cai', 'long-an': 'Long An',
  'nam-dinh': 'Nam Định', 'nghe-an': 'Nghệ An', 'ninh-binh': 'Ninh Bình', 'ninh-thuan': 'Ninh Thuận', 'phu-tho': 'Phú Thọ',
  'phu-yen': 'Phú Yên', 'quang-binh': 'Quảng Bình', 'quang-nam': 'Quảng Nam', 'quang-ngai': 'Quảng Ngãi', 'quang-ninh': 'Quảng Ninh',
  'quang-tri': 'Quảng Trị', 'soc-trang': 'Sóc Trăng', 'son-la': 'Sơn La', 'tay-ninh': 'Tây Ninh', 'thai-binh': 'Thái Bình',
  'thai-nguyen': 'Thái Nguyên', 'thanh-hoa': 'Thanh Hóa', 'thua-thien-hue': 'Thừa Thiên Huế', 'tien-giang': 'Tiền Giang',
  'tra-vinh': 'Trà Vinh', 'tuyen-quang': 'Tuyên Quang', 'vinh-long': 'Vĩnh Long', 'vinh-phuc': 'Vĩnh Phúc', 'yen-bai': 'Yên Bái',
}

const UPPER_BRANDS = new Set(['bmw', 'mg', 'byd', 'gmc', 'jrd', 'sym', 'ram'])

export function formatBrand(slug: string): string {
  if (UPPER_BRANDS.has(slug)) return slug.toUpperCase()
  return slug.replace(/(^|[\s-])([a-zà-ỹ])/g, (_m, sep: string, ch: string) => sep + ch.toUpperCase())
}

export function formatModel(slug: string): string {
  return slug.length <= 3 ? slug.toUpperCase() : slug.replace(/(^|\s)([a-zà-ỹ])/g, (_m, sep: string, ch: string) => sep + ch.toUpperCase())
}

/** Nhãn hiển thị cho một giá trị của trường `field`. */
export function valueLabel(field: string, value: string | number | null | undefined): string {
  if (value == null || value === '') return '—'
  const s = String(value)
  if (field === 'brand') return formatBrand(s)
  if (field === 'model') return formatModel(s)
  if (field === 'province') return PROVINCES[s] ?? s
  return VALUE_LABELS[field]?.[s] ?? s
}

/** Tên tiếng Việt của mô hình (backend trả về nhãn tiếng Anh như "Random Forest": đây là tên thuật toán nên giữ nguyên). */
export const MODEL_NOTE: Record<string, string> = {
  linear_regression: 'Đơn giản, dễ giải thích, thường kém ở xe hiếm',
  random_forest: 'Rừng ngẫu nhiên: nhiều cây quyết định bỏ phiếu',
  xgboost: 'Tăng cường gradient: các cây sửa sai lẫn nhau',
}

export interface FieldGroup {
  title: string
  description: string
  fields: string[]
}

export const FIELD_GROUPS: FieldGroup[] = [
  { title: 'Thông tin cơ bản', description: 'Xe của bạn là xe gì, đời nào?', fields: ['brand', 'model', 'variant', 'year', 'body_type'] },
  {
    title: 'Thông số kỹ thuật', description: 'Càng đầy đủ, dự đoán càng sát. Trường nào chưa rõ có thể bỏ trống.',
    fields: ['mileage_km', 'engine_capacity', 'fuel_type', 'transmission', 'drivetrain', 'seats'],
  },
  { title: 'Thông tin bổ sung', description: 'Nguồn gốc và nơi bán.', fields: ['origin', 'province'] },
]

export const FIELD_HELP: Record<string, { placeholder?: string; hint: string; unit?: string }> = {
  brand: { hint: 'Chọn hãng xe có trong dữ liệu huấn luyện.' },
  model: { placeholder: 'Ví dụ: vios, cx5, c (C-Class)', hint: 'Gõ tên dòng xe; danh sách gợi ý đổi theo hãng đã chọn.' },
  variant: { placeholder: 'Ví dụ: g, e, premium', hint: 'Không bắt buộc. Phiên bản chưa gặp khi huấn luyện chỉ bị cảnh báo.' },
  year: { placeholder: 'Ví dụ: 2018', hint: 'Năm sản xuất, không phải năm đăng ký.' },
  body_type: { hint: 'Không bắt buộc.' },
  mileage_km: { placeholder: 'Ví dụ: 60000', hint: 'Số km trên đồng hồ. Để trống nếu không rõ; nhập 0 được coi là "chưa khai báo".', unit: 'km' },
  engine_capacity: { placeholder: 'Ví dụ: 1,5', hint: 'Dung tích xi-lanh. Để trống, hệ thống tự điền theo dòng xe.', unit: 'lít' },
  fuel_type: { hint: 'Không bắt buộc.' },
  transmission: { hint: 'Không bắt buộc.' },
  drivetrain: { hint: 'Không bắt buộc. Dữ liệu dẫn động chủ yếu có ở nguồn Bonbanh.' },
  seats: { placeholder: 'Ví dụ: 5', hint: 'Từ 2 đến 16 chỗ. Để trống, hệ thống tự điền theo dòng xe.', unit: 'chỗ' },
  origin: { hint: 'Không bắt buộc.' },
  province: { hint: 'Không bắt buộc. Chỉ có ở dữ liệu Chợ Tốt.' },
}

export const fieldLabel = (f: FeatureMeta) => f.label || f.name
export const fieldLabelByName = (name: string, features: FeatureMeta[] | undefined) =>
  features?.find((f) => f.name === name)?.label ?? name

/** Lọc các cảnh báo kỹ thuật cũ của backend về dạng dễ đọc hơn (giữ nguyên nội dung, chỉ bỏ tên trường thô nếu có nhãn). */
export function humanizeMessage(msg: string, features: FeatureMeta[] | undefined): string {
  return msg.replace(/'([a-z_]+)'/g, (m, name: string) => {
    const l = features?.find((f) => f.name === name)?.label
    return l ? `“${l}”` : m
  })
}
