import { createContext, useContext } from 'react'

export type ToastKind = 'success' | 'error' | 'info'
export const ToastContext = createContext<{ notify: (kind: ToastKind, text: string) => void }>({ notify: () => {} })
export const useToast = () => useContext(ToastContext)
