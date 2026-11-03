/** Thông tin hiển thị chung. Liên hệ lấy từ biến môi trường (xem .env.example); để trống thì không hiển thị. */
export const SITE = {
  name: 'Giá Xe AI',
  email: (import.meta.env.VITE_CONTACT_EMAIL as string | undefined) ?? '',
  phone: (import.meta.env.VITE_CONTACT_PHONE as string | undefined) ?? '',
  repo: (import.meta.env.VITE_REPO_URL as string | undefined) ?? '',
}
