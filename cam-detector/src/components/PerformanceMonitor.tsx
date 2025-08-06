import React, { useState, useEffect } from 'react'
import type { ModelStats } from '../types/detection'

interface PerformanceMonitorProps {
    stats: ModelStats | null
    isModelLoaded: boolean
    provider: string
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
    stats,
    isModelLoaded,
    provider,
}) => {
    const [fps, setFps] = useState(0)
    const [frameCount, setFrameCount] = useState(0)
    const [lastTime, setLastTime] = useState(Date.now())

    useEffect(() => {
        const interval = setInterval(() => {
            const now = Date.now()
            const deltaTime = now - lastTime

            if (deltaTime >= 1000) {
                const currentFps = (frameCount * 1000) / deltaTime
                setFps(Math.round(currentFps * 10) / 10)
                setFrameCount(0)
                setLastTime(now)
            }
        }, 100)

        return () => clearInterval(interval)
    }, [frameCount, lastTime])

    // Incrementar contador cuando hay nuevas stats
    useEffect(() => {
        if (stats) {
            setFrameCount((prev) => prev + 1)
        }
    }, [stats])

    const getStatusColor = () => {
        if (!isModelLoaded) return '#ff6b6b'
        if (fps >= 0.4) return '#51cf66'
        if (fps >= 0.2) return '#ffd43b'
        return '#ff8787'
    }

    return (
        <div className='performance-monitor'>
            <div className='performance-header'>
                <h4>📊 Monitor de Rendimiento</h4>
            </div>

            <div className='performance-grid'>
                <div className='perf-item'>
                    <span className='perf-label'>Estado:</span>
                    <span
                        className='perf-value'
                        style={{ color: getStatusColor() }}
                    >
                        {isModelLoaded ? '✅ Activo' : '⏳ Cargando'}
                    </span>
                </div>

                <div className='perf-item'>
                    <span className='perf-label'>Backend:</span>
                    <span className='perf-value'>{provider.toUpperCase()}</span>
                </div>

                <div className='perf-item'>
                    <span className='perf-label'>FPS Modelo:</span>
                    <span
                        className='perf-value'
                        style={{ color: getStatusColor() }}
                    >
                        {fps.toFixed(1)}
                    </span>
                </div>

                {stats && (
                    <>
                        <div className='perf-item'>
                            <span className='perf-label'>Tiempo Proc.:</span>
                            <span className='perf-value'>
                                {stats.processingTimeMs}ms
                            </span>
                        </div>

                        <div className='perf-item'>
                            <span className='perf-label'>FPS Estimado:</span>
                            <span className='perf-value'>
                                {stats.fps.toFixed(1)}
                            </span>
                        </div>

                        <div className='perf-item'>
                            <span className='perf-label'>Caras:</span>
                            <span className='perf-value'>
                                {stats.faceDetections}
                            </span>
                        </div>

                        <div className='perf-item'>
                            <span className='perf-label'>Documentos:</span>
                            <span className='perf-value'>
                                {stats.docDetections}
                            </span>
                        </div>

                        <div className='perf-item'>
                            <span className='perf-label'>Total:</span>
                            <span className='perf-value'>
                                {stats.validDetections}/{stats.totalDetections}
                            </span>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}
