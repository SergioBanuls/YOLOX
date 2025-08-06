import React from 'react'
import type { Detection, ModelStats } from '../types/detection'

interface StatsDisplayProps {
    detections: Detection[]
    stats: ModelStats
    isModelLoaded: boolean
}

export const StatsDisplay: React.FC<StatsDisplayProps> = ({
    detections,
    stats,
    isModelLoaded,
}) => {
    return (
        <div className='stats-container'>
            <h3>📊 Estadísticas del Modelo</h3>

            <div className='status-section'>
                <h4>🔧 Estado</h4>
                <div className='stat-item'>
                    <span className='label'>Modelo:</span>
                    <span
                        className={`value ${
                            isModelLoaded ? 'success' : 'warning'
                        }`}
                    >
                        {isModelLoaded ? '✅ Cargado' : '⏳ Cargando...'}
                    </span>
                </div>
                <div className='stat-item'>
                    <span className='label'>FPS:</span>
                    <span className='value'>{stats.fps.toFixed(1)}</span>
                </div>
                <div className='stat-item'>
                    <span className='label'>Tiempo proc:</span>
                    <span className='value'>
                        {stats.processingTimeMs.toFixed(1)}ms
                    </span>
                </div>
            </div>

            <div className='detection-section'>
                <h4>🎯 Detecciones</h4>
                <div className='stat-item'>
                    <span className='label'>Total puntos:</span>
                    <span className='value'>
                        {stats.totalDetections.toLocaleString()}
                    </span>
                </div>
                <div className='stat-item'>
                    <span className='label'>Detecciones válidas:</span>
                    <span className='value'>{stats.validDetections}</span>
                </div>
                <div className='stat-item'>
                    <span className='label'>👤 Caras:</span>
                    <span className='value face-count'>
                        {stats.faceDetections}
                    </span>
                </div>
                <div className='stat-item'>
                    <span className='label'>📄 Documentos:</span>
                    <span className='value doc-count'>
                        {stats.docDetections}
                    </span>
                </div>
            </div>

            <div className='objectness-section'>
                <h4>📈 Objectness</h4>
                <div className='stat-item'>
                    <span className='label'>Mínimo:</span>
                    <span className='value'>
                        {stats.objectnessMin.toFixed(4)}
                    </span>
                </div>
                <div className='stat-item'>
                    <span className='label'>Máximo:</span>
                    <span className='value'>
                        {stats.objectnessMax.toFixed(4)}
                    </span>
                </div>
                <div className='stat-item'>
                    <span className='label'>Promedio:</span>
                    <span className='value'>
                        {stats.objectnessMean.toFixed(4)}
                    </span>
                </div>
            </div>

            {detections.length > 0 && (
                <div className='detections-section'>
                    <h4>🔍 Detecciones Actuales</h4>
                    {detections.map((detection, index) => (
                        <div
                            key={index}
                            className={`detection-item ${detection.class_name}`}
                        >
                            <div className='detection-header'>
                                <span className='detection-class'>
                                    {detection.class_name === 'face'
                                        ? '👤'
                                        : '📄'}{' '}
                                    {detection.class_name}
                                </span>
                                <span className='detection-score'>
                                    {(detection.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className='detection-coords'>
                                <small>
                                    ({detection.x1.toFixed(0)},{' '}
                                    {detection.y1.toFixed(0)}) - (
                                    {detection.x2.toFixed(0)},{' '}
                                    {detection.y2.toFixed(0)})
                                </small>
                            </div>
                            <div className='detection-size'>
                                <small>
                                    {(detection.x2 - detection.x1).toFixed(0)} ×{' '}
                                    {(detection.y2 - detection.y1).toFixed(0)}px
                                </small>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <div className='legend-section'>
                <h4>🎨 Leyenda</h4>
                <div className='legend-item'>
                    <div className='legend-color face-color'></div>
                    <span>Cara (Verde)</span>
                </div>
                <div className='legend-item'>
                    <div className='legend-color doc-color'></div>
                    <span>Documento (Rojo)</span>
                </div>
            </div>
        </div>
    )
}
