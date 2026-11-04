/** Kiểu dữ liệu khớp với phản hồi thật của FastAPI (app/main.py). Không đổi tên trường. */

export interface NumberFeature {
  name: string
  label: string
  required: boolean
  type: 'number'
  integer: boolean
  min: number | null // giới hạn hợp lệ tuyệt đối mà backend chấp nhận
  max: number | null
  train_min: number
  train_max: number
  median: number
}

export interface CategoryFeature {
  name: string
  label: string
  required: boolean
  type: 'category'
  free_text: boolean
  options: string[]
  default: string
}

export type FeatureMeta = NumberFeature | CategoryFeature

export interface MetaResponse {
  features: FeatureMeta[]
  models: Record<string, string>
  default_model: string
  unit: string
  unit_note: string
  required: string[]
  model_version: string
  data_version: string
  brand_models: Record<string, string[]>
}

export interface PredictionItem {
  model: string
  label: string
  price: number
}

export interface PredictResponse {
  prediction_id: string
  model: string
  model_label: string
  price: number
  unit: string
  unit_note: string
  model_version: string
  data_version: string
  all: PredictionItem[]
  warnings: string[]
  defaults_used: string[]
}

export type PredictValues = Record<string, string | number>

export interface ModelMetric {
  model: string
  label: string
  config_used: string | null
  cv_rmse: number | null
  val_RMSE: number | null
  MAE: number | null
  RMSE: number | null
  R2: number | null
  MAPE: number | null
  medAPE: number | null
}

export interface MetricsResponse {
  split: string
  n_train: number
  n_val?: number | null
  n_test: number
  selection_criterion: string
  best_model: string
  unit: string
  unit_note: string
  split_strategy?: string | null
  models: ModelMetric[]
}

export interface ExperimentRow {
  split: string
  model: string
  config_used: string
  MAE: number
  RMSE: number
  R2: number
  medAPE: number
  val_RMSE: number
  selected_best: boolean
}

export interface ModelInfo {
  model_version: string
  trained_at: string
  data_version: string
  n_rows_clean: number | null
  n_raw: number | null
  split: string
  split_strategy: string | null
  n_train: number
  n_val: number | null
  n_test: number
  target: string
  target_transform: string | null
  price_unit: string
  price_unit_note: string
  reference_year: number | null
  cv: string | null
  tuned: boolean
  tuning_iterations: number | null
  selection_criterion: string
  best_model: string
  libraries: Record<string, string | null>
  serving_models: string[]
  models: Record<string, { label: string; config_used: string | null; best_params: Record<string, unknown> }>
}

export interface DriftFeature {
  psi: number
  flag: boolean
  type: 'numeric' | 'categorical'
}

export interface DriftReport {
  status: 'insufficient_samples' | 'no_signal' | 'data_drift_signal' | 'no_reference'
  n_samples?: number
  threshold_psi?: number
  features?: Record<string, DriftFeature>
  drifted_features?: string[]
  message?: string
  note?: string
}

export interface FeedbackSummary {
  n: number
  message?: string
  MAE?: number
  RMSE?: number
  mae_by_day?: Record<string, number>
}

export interface RecentError {
  ts: string
  endpoint: string
  status: number
  detail: string
  latency_ms: number | null
}

export interface MonitoringResponse {
  model_version: string | null
  n_predictions: number
  n_errors: number
  latency_ms: { mean: number | null; p95: number | null }
  recent_errors: RecentError[]
  drift: DriftReport
  feedback: FeedbackSummary
}

export interface HealthResponse {
  status: string
  models?: string[]
  model_version?: string
  detail?: string
}

/* ---- Gợi ý xe: CHƯA có endpoint trong backend. Đây là hợp đồng đề xuất để tích hợp sau (xem README). ---- */
export interface RecommendFilters {
  budget_min?: number
  budget_max?: number
  brand?: string
  model?: string
  year_from?: number
  fuel_type?: string
  transmission?: string
  body_type?: string
  method: 'knn' | 'content_based'
  limit: number
}

export interface CarListing {
  id: string
  brand: string
  model: string
  variant?: string | null
  year: number
  mileage_km?: number | null
  engine_capacity?: number | null
  fuel_type?: string | null
  transmission?: string | null
  body_type?: string | null
  origin?: string | null
  province?: string | null
  seats?: number | null
  price: number
  image_url?: string | null
  highlights?: string[]
  similar_ids?: string[]
  source?: string | null
  color?: string | null
}

export interface RecommendResponse {
  items: CarListing[]
  method: 'knn' | 'content_based'
  total?: number
}
