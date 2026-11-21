/** Chuẩn hóa để tìm kiếm: bỏ dấu tiếng Việt, về chữ thường, coi gạch nối và khoảng trắng như nhau.
 *  "Mercédès-Benz" → "mercedes benz"; "C-Class" và "C Class" cùng thành "c class". */
export function fold(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[đĐ]/g, 'd').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}
