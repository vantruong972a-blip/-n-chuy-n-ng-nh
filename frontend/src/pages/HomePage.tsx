import { Link } from 'react-router-dom'
import { Activity, ArrowRight, Calculator, Search } from 'lucide-react'
import { CarBlueprint } from '../components/CarBlueprint'
import { Alert, Button, ButtonLink, Stat } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { formatInt, formatPercent } from '../lib/format'
import { getMeta } from '../services/predictionService'
import { getMetrics } from '../services/monitoringService'

const STEPS = [
  { title: 'Nhập thông tin xe', text: 'Hãng, dòng xe và năm sản xuất là đủ để bắt đầu; thêm số km, hộp số… để dự đoán sát hơn.' },
  { title: 'Kiểm tra và chuẩn hóa', text: 'Máy chủ từ chối giá trị vô lý (năm 1850, âm km…) và cảnh báo khi xe nằm ngoài phạm vi dữ liệu huấn luyện.' },
  { title: 'Ba mô hình cùng dự đoán', text: 'Linear Regression, Random Forest và XGBoost nhận cùng một đầu vào; mô hình tốt nhất được chọn theo tập kiểm định.' },
  { title: 'Nhận giá tham khảo', text: 'Kết quả kèm tên mô hình, phiên bản và cảnh báo. Bạn có thể gửi lại giá thực tế để hệ thống theo dõi sai số.' },
]

function HeroStats() {
  const metrics = useAsync(getMetrics, 'metrics')
  const meta = useAsync(getMeta, 'meta')
  const loading = metrics.loading || meta.loading
  const failed = metrics.error ?? meta.error
  const best = metrics.data?.models.find((m) => m.model === metrics.data?.best_model)
  if (failed && !loading) {
    return (
      <Alert kind="warn" title="Chưa lấy được số liệu từ máy chủ" action={<Button variant="secondary" onClick={() => { metrics.reload(); meta.reload() }}>Thử lại</Button>}>
        {failed.message}
      </Alert>
    )
  }
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-8 lg:grid-cols-4">
      <Stat loading={loading} label="Tin rao đã dùng để học và kiểm tra" value={metrics.data && formatInt(metrics.data.n_train + metrics.data.n_test)} note="Từ Bonbanh và Chợ Tốt, sau khi làm sạch" />
      <Stat loading={loading} label="Mô hình cùng dự đoán" value={meta.data && Object.keys(meta.data.models).length} note="Linear Regression, Random Forest, XGBoost" />
      <Stat loading={loading} label="Sai số tương đối trung vị" value={best && formatPercent(best.medAPE)} note={metrics.data && `Mô hình tốt nhất, trên ${formatInt(metrics.data.n_test)} xe kiểm tra chưa từng thấy`} />
      <Stat loading={loading} label="Phiên bản mô hình" value={meta.data && <span className="text-xl sm:text-2xl">{meta.data.model_version.split('-').slice(0, 2).join('-')}</span>} note="Đang phục vụ" />
    </dl>
  )
}

export default function HomePage() {
  return (
    <>
      <section className="relative overflow-hidden bg-navy-deep text-white" style={{ backgroundImage: 'linear-gradient(rgb(255 255 255 / 0.045) 1px, transparent 1px), linear-gradient(90deg, rgb(255 255 255 / 0.045) 1px, transparent 1px)', backgroundSize: '44px 44px' }}>
        <div className="container-page grid items-center gap-8 py-14 sm:py-20 lg:grid-cols-[1fr_1.1fr] lg:py-24">
          <div>
            <h1 className="text-4xl font-extrabold leading-[1.1] sm:text-5xl lg:text-[3.4rem]">
              Biết xe của bạn đáng giá bao nhiêu trước khi rao bán hoặc mua.
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-white/80">
              Hệ thống định giá ô tô cũ bằng ba mô hình học máy, huấn luyện trên hàng chục nghìn tin rao thật tại Việt Nam. Nhập vài thông tin, nhận giá tham khảo trong vài giây.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <ButtonLink to="/dinh-gia" variant="light" className="h-12 px-6 text-base"><Calculator className="size-5" aria-hidden />Định giá ô tô ngay</ButtonLink>
              <ButtonLink to="/goi-y" variant="outlineLight" className="h-12 px-6 text-base">
                <Search className="size-5" aria-hidden />Khám phá xe phù hợp
              </ButtonLink>
            </div>
          </div>
          <CarBlueprint className="mx-auto w-full max-w-[640px]" />
        </div>
      </section>

      <section aria-label="Số liệu hệ thống" className="border-b border-line">
        <div className="container-page py-10"><HeroStats /></div>
      </section>

      <section className="container-page py-16 sm:py-20">
        <h2 className="max-w-2xl text-3xl font-extrabold text-navy-deep sm:text-4xl">Ba việc bạn có thể làm ở đây</h2>
        <div className="mt-10 grid gap-5 lg:grid-cols-[1.5fr_1fr]">
          <article className="card flex flex-col justify-between border-navy/20 p-7 sm:p-9">
            <div>
              <Calculator className="size-9 text-navy" aria-hidden />
              <h3 className="mt-5 text-2xl font-bold text-navy-deep">Định giá ô tô</h3>
              <p className="mt-3 max-w-xl leading-relaxed text-ink-soft">
                Nhập hãng, dòng xe, năm sản xuất và các thông số bạn biết. Mỗi trường nhập được kiểm tra ngay; trường nào chưa rõ có thể để trống. Kết quả cho thấy giá của cả ba mô hình để bạn thấy độ phân tán của dự đoán.
              </p>
            </div>
            <div className="mt-7"><ButtonLink to="/dinh-gia">Bắt đầu định giá<ArrowRight className="size-4" aria-hidden /></ButtonLink></div>
          </article>
          <div className="grid gap-5">
            <article className="card p-6">
              <Search className="size-7 text-navy" aria-hidden />
              <h3 className="mt-4 text-lg font-bold text-navy-deep">Gợi ý xe phù hợp</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">Lọc theo ngân sách và nhu cầu. Máy chủ chưa có API gợi ý nên chức năng này chưa hoạt động; giao diện đã sẵn sàng để nối KNN và lọc theo nội dung.</p>
              <Link to="/goi-y" className="mt-3 inline-flex min-h-11 items-center gap-1 text-sm font-semibold text-navy hover:underline">Xem trạng thái<ArrowRight className="size-4" aria-hidden /></Link>
            </article>
            <article className="card p-6">
              <Activity className="size-7 text-navy" aria-hidden />
              <h3 className="mt-4 text-lg font-bold text-navy-deep">Quản trị và giám sát</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">Độ chính xác từng mô hình, độ trễ API, lệch phân phối dữ liệu (PSI) và sai số thực tế khi có phản hồi.</p>
              <Link to="/quan-tri" className="mt-3 inline-flex min-h-11 items-center gap-1 text-sm font-semibold text-navy hover:underline">Mở trang quản trị<ArrowRight className="size-4" aria-hidden /></Link>
            </article>
          </div>
        </div>
      </section>

      <section className="bg-haze py-16 sm:py-20">
        <div className="container-page">
          <h2 className="max-w-2xl text-3xl font-extrabold text-navy-deep sm:text-4xl">Quy trình định giá bằng AI</h2>
          <ol className="mt-12 grid gap-10 sm:grid-cols-2 lg:grid-cols-4 lg:gap-6">
            {STEPS.map((s, i) => (
              <li key={s.title} className="relative lg:pt-7">
                <span aria-hidden className="absolute left-0 top-0 hidden h-px w-full bg-navy/25 lg:block" />
                <span aria-hidden className="absolute -top-[5px] left-0 hidden size-[11px] rounded-full bg-navy lg:block" />
                <span className="grid size-9 place-items-center rounded-full bg-navy text-sm font-bold text-white lg:hidden">{i + 1}</span>
                <p className="mt-3 text-sm font-bold text-azure lg:mt-0">Bước {i + 1}</p>
                <h3 className="mt-1 text-lg font-bold text-navy-deep">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-soft">{s.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="container-page grid gap-10 py-16 sm:py-20 lg:grid-cols-2">
        <div>
          <h2 className="text-3xl font-extrabold text-navy-deep sm:text-4xl">Về hệ thống</h2>
          <p className="mt-5 leading-relaxed text-ink-soft">
            Dữ liệu gồm các tin rao xe cũ trên Bonbanh và Chợ Tốt, được làm sạch: loại tin trùng, loại giá nhập sai (như 5 đồng hay 420 tỷ cho một chiếc Kia Soluto) và giữ lại xe sang có giá cao nhưng hợp lệ. Mô hình được đánh giá trên các xe hoàn toàn chưa thấy khi huấn luyện, và các tin đăng lại cùng một xe luôn nằm cùng một phía để kết quả không đẹp giả.
          </p>
          <Link to="/quan-tri" className="mt-5 inline-flex min-h-11 items-center gap-1 font-semibold text-navy hover:underline">Xem số liệu đánh giá chi tiết<ArrowRight className="size-4" aria-hidden /></Link>
        </div>
        <div className="card border-warn/30 bg-warn-soft p-6 sm:p-8">
          <h3 className="text-lg font-bold text-ink">Cần biết trước khi dùng kết quả</h3>
          <ul className="mt-4 space-y-3 text-sm leading-relaxed text-ink-soft">
            <li>Đây là <strong className="text-ink">giá rao</strong> của người bán, thường cao hơn giá chốt, không phải giá giao dịch.</li>
            <li>Dữ liệu thu thập khoảng <strong className="text-ink">năm 2022</strong>, nên không phản ánh thị trường hiện nay.</li>
            <li>Xe sang, xe hiếm có rất ít mẫu nên sai số có thể lớn.</li>
            <li>Giá chỉ để tham khảo, không dùng làm căn cứ định giá pháp lý hay báo giá.</li>
          </ul>
        </div>
      </section>
    </>
  )
}
