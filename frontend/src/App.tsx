import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { MainLayout } from './layouts/MainLayout'
import HomePage from './pages/HomePage'
import NotFoundPage from './pages/NotFoundPage'
import { Skeleton } from './components/ui'

const ValuationPage = lazy(() => import('./pages/ValuationPage'))
const RecommendPage = lazy(() => import('./pages/RecommendPage'))
const CarDetailPage = lazy(() => import('./pages/CarDetailPage'))
const HistoryPage = lazy(() => import('./pages/HistoryPage'))
const MlflowPage = lazy(() => import('./pages/MlflowPage'))
const AdminPage = lazy(() => import('./pages/AdminPage'))

const Loading = () => <div className="container-page py-16"><Skeleton className="h-10 w-1/3" /><Skeleton className="mt-6 h-64" /></div>

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<MainLayout />}>
          <Route index element={<HomePage />} />
          <Route path="dinh-gia" element={<ValuationPage />} />
          <Route path="goi-y" element={<RecommendPage />} />
          <Route path="xe/:id" element={<CarDetailPage />} />
          <Route path="lich-su" element={<HistoryPage />} />
          <Route path="mlflow" element={<MlflowPage />} />
          <Route path="quan-tri" element={<AdminPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  )
}
