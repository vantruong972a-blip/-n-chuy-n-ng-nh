import { useSyncExternalStore } from 'react'
import type { PredictResponse, PredictValues } from '../types/api'

/** Lịch sử dự đoán của riêng người dùng, lưu trong trình duyệt (localStorage) từ kết quả thật của API. */
export interface HistoryEntry {
  id: string
  createdAt: string
  payload: PredictValues
  result: PredictResponse
}

const KEY = 'giaxe.lichsu.v1'
const MAX = 50
const listeners = new Set<() => void>()
let snapshot: HistoryEntry[] | null = null

function read(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY)
    const arr = raw ? (JSON.parse(raw) as HistoryEntry[]) : []
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

function write(list: HistoryEntry[]) {
  snapshot = list
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    /* trình duyệt chặn lưu trữ: vẫn giữ trong bộ nhớ của phiên */
  }
  listeners.forEach((l) => l())
}

export function addHistory(payload: PredictValues, result: PredictResponse) {
  const entry: HistoryEntry = { id: result.prediction_id, createdAt: new Date().toISOString(), payload, result }
  write([entry, ...getSnapshot().filter((e) => e.id !== entry.id)].slice(0, MAX))
}
export const removeHistory = (id: string) => write(getSnapshot().filter((e) => e.id !== id))
export const clearHistory = () => write([])

function getSnapshot(): HistoryEntry[] {
  snapshot ??= read()
  return snapshot
}
function subscribe(cb: () => void) {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

export const usePredictionHistory = () => useSyncExternalStore(subscribe, getSnapshot, () => [])
