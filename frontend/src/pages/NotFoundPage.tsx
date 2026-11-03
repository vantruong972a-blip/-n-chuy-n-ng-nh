import { ButtonLink } from '../components/ui'

export default function NotFoundPage() {
  return (
    <div className="container-page py-24 text-center">
      <h1 className="text-4xl font-extrabold text-navy-deep">Không tìm thấy trang</h1>
      <p className="mx-auto mt-3 max-w-md text-ink-soft">Đường dẫn bạn mở không tồn tại hoặc đã được đổi.</p>
      <ButtonLink to="/" className="mt-8">Về trang chủ</ButtonLink>
    </div>
  )
}
