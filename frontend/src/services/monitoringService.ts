import type { ExperimentRow, MetricsResponse, ModelInfo, MonitoringResponse } from '../types/api'
import { request } from './http'

export const getMetrics = () => request<MetricsResponse>('/api/metrics')
export const getExperiments = () => request<ExperimentRow[]>('/api/metrics/all')
export const getMonitoring = () => request<MonitoringResponse>('/api/monitoring')
export const getModelInfo = () => request<ModelInfo>('/api/model-info')
export const getMlflow = () => request<import('../types/api').MlflowSummary>('/api/mlflow')
export const getModelStrengths = () => request<import('../types/api').StrengthsResponse>('/api/model-strengths')
