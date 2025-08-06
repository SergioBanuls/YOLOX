import { useRef, useCallback } from 'react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const useThrottledCallback = <T extends (...args: any[]) => any>(
    callback: T,
    delay: number
): T => {
    const lastCallRef = useRef<number>(0)
    const timeoutRef = useRef<number | undefined>(undefined)

    return useCallback(
        (...args: Parameters<T>) => {
            const now = Date.now()

            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current)
            }

            if (now - lastCallRef.current >= delay) {
                lastCallRef.current = now
                callback(...args)
            } else {
                timeoutRef.current = setTimeout(() => {
                    lastCallRef.current = Date.now()
                    callback(...args)
                }, delay - (now - lastCallRef.current))
            }
        },
        [callback, delay]
    ) as T
}
