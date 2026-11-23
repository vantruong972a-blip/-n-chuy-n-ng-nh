import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { ToastProvider } from '../components/ToastProvider'
import { SITE } from '../site'

const NAV = [
  { to: '/', label: 'Trang chủ', end: true },
  { to: '/dinh-gia', label: 'Định giá ô tô' },
  { to: '/goi-y', label: 'Gợi ý xe phù hợp' },
  { to: '/lich-su', label: 'Lịch sử dự đoán' },
  { to: '/mlflow', label: 'Theo dõi MLflow' },
  { to: '/quan-tri', label: 'Quản trị hệ thống' },
]

export function Logo({ light = false }: { light?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
        <rect width="32" height="32" rx="7" fill={light ? '#fff' : '#0A2463'} />
        <path d="M5 19.5l2.2-5.2A3 3 0 0 1 10 12.5h8.4a3 3 0 0 1 2.2 1l2.6 2.9 2.5.6a1.8 1.8 0 0 1 1.3 1.7v1.3H5z" fill="none" stroke={light ? '#0A2463' : '#fff'} strokeWidth="1.6" strokeLinejoin="round" />
        <circle cx="11" cy="20.5" r="2.3" fill={light ? '#fff' : '#0A2463'} stroke={light ? '#0A2463' : '#fff'} strokeWidth="1.6" />
        <circle cx="23" cy="20.5" r="2.3" fill={light ? '#fff' : '#0A2463'} stroke={light ? '#0A2463' : '#fff'} strokeWidth="1.6" />
      </svg>
      <span className={`text-lg font-extrabold tracking-tight ${light ? 'text-white' : 'text-navy-deep'}`}>{SITE.name}</span>
    </span>
  )
}

function Header() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => {
    const t = setTimeout(() => setOpen(false), 0)
    return () => clearTimeout(t)
  }, [pathname])
  const link = ({ isActive }: { isActive: boolean }) =>
    `relative rounded-lg px-3 py-2 text-[15px] font-medium transition-colors ${isActive ? 'bg-azure-soft text-navy after:absolute after:inset-x-3 after:-bottom-[13px] after:h-[3px] after:rounded-full after:bg-azure' : 'text-ink-soft hover:bg-haze hover:text-ink'}`
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-white/90 shadow-[0_6px_18px_-14px_rgb(10_36_99/0.35)] backdrop-blur">
      <div className="container-page flex h-16 items-center justify-between gap-4">
        <Link to="/" aria-label={`${SITE.name} – về trang chủ`} className="rounded-lg"><Logo /></Link>
        <nav aria-label="Điều hướng chính" className="hidden items-center gap-1 lg:flex">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className={link}>{n.label}</NavLink>)}
        </nav>
        <button type="button" className="grid size-11 place-items-center rounded-lg border border-line lg:hidden" aria-expanded={open} aria-controls="menu-di-dong"
          aria-label={open ? 'Đóng menu' : 'Mở menu'} onClick={() => setOpen((o) => !o)}>
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </div>
      {open && (
        <nav id="menu-di-dong" aria-label="Điều hướng chính" className="border-t border-line bg-white lg:hidden">
          <div className="container-page flex flex-col gap-1 py-3">
            {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className={(s) => `${link(s)} min-h-11 content-center`}>{n.label}</NavLink>)}
          </div>
        </nav>
      )}
    </header>
  )
}

function Footer() {
  return (
    <footer className="bg-navy-deep text-white/80">
      <div className="container-page grid gap-10 py-12 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <Logo light />
          <p className="mt-4 max-w-sm text-sm leading-relaxed">
            Hệ thống định giá ô tô cũ tại Việt Nam bằng học máy, xây dựng như một dự án MLOps hoàn chỉnh: dữ liệu, huấn luyện, theo dõi thí nghiệm, triển khai và giám sát.
          </p>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-white">Khám phá</h2>
          <ul className="mt-3 space-y-2 text-sm">
            {NAV.map((n) => <li key={n.to}><Link className="hover:text-white hover:underline" to={n.to}>{n.label}</Link></li>)}
          </ul>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-white">Liên hệ</h2>
          <ul className="mt-3 space-y-2 text-sm">
            {SITE.email && <li><a className="hover:text-white hover:underline" href={`mailto:${SITE.email}`}>{SITE.email}</a></li>}
            {SITE.phone && <li><a className="hover:text-white hover:underline" href={`tel:${SITE.phone.replace(/\s/g, '')}`}>{SITE.phone}</a></li>}
            {SITE.repo && <li><a className="hover:text-white hover:underline" href={SITE.repo} target="_blank" rel="noreferrer">Mã nguồn dự án</a></li>}
            {!SITE.email && !SITE.phone && !SITE.repo && <li>Dự án học tập, chưa có kênh liên hệ công khai.</li>}
          </ul>
        </div>
      </div>
      <div className="border-t border-white/10">
        <p className="container-page py-5 text-xs leading-relaxed text-white/60">
          © {new Date().getFullYear()} {SITE.name}. Giá hiển thị là giá tham khảo suy ra từ tin rao khoảng năm 2022, không phải báo giá hay giá giao dịch.
        </p>
      </div>
    </footer>
  )
}

export function MainLayout() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo({ top: 0 }) }, [pathname])
  return (
    <ToastProvider>
      <a href="#noi-dung" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:shadow-lg">
        Bỏ qua điều hướng
      </a>
      <Header />
      <main id="noi-dung" tabIndex={-1} className={`min-h-[60vh] outline-none ${pathname === '/' ? 'bg-white' : 'page-bg'}`}>
        <Outlet />
      </main>
      <Footer />
    </ToastProvider>
  )
}
