import { useState, useEffect, useCallback } from 'react'
import { WebcamCanvas } from './components/WebcamCanvasOptimized'
import { StatsDisplay } from './components/StatsDisplay'
import { ModelConfig } from './components/ModelConfig'
import { PerformanceMonitor } from './components/PerformanceMonitor'
import { PhotoCapture } from './components/PhotoCapture'
import { YOLOXDetector, type ExecutionProvider } from './utils/detector'
import type { Detection, ModelStats } from './types/detection'
import './App.css'

function App() {
    const [detector] = useState(() => new YOLOXDetector())
    const [detections, setDetections] = useState<Detection[]>([])
    const [stats, setStats] = useState<ModelStats>({
        objectnessMin: 0,
        objectnessMax: 0,
        objectnessMean: 0,
        totalDetections: 0,
        validDetections: 0,
        faceDetections: 0,
        docDetections: 0,
        processingTimeMs: 0,
        fps: 0,
    })
    const [isModelLoaded, setIsModelLoaded] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [selectedProvider, setSelectedProvider] =
        useState<ExecutionProvider>('webgpu') // Cambiar a WebGPU por defecto
    const [capturedImage, setCapturedImage] = useState<string | null>(null)
    const [showCapture, setShowCapture] = useState(false)
    const [captureConfidence, setCaptureConfidence] = useState<number>(0)
    const [autoCapture, setAutoCapture] = useState<boolean>(true)
    const [showGpuSuggestion, setShowGpuSuggestion] = useState<boolean>(false)

    useEffect(() => {
        const loadModel = async () => {
            try {
                setIsModelLoaded(false)
                setError('Cargando modelo...')
                console.log(
                    'Cargando modelo YOLOX con backend:',
                    selectedProvider
                )

                await detector.loadModel(
                    '/yolox_doc_face_decoded.onnx',
                    selectedProvider
                )
                setIsModelLoaded(true)
                setError(null)
                console.log('Modelo cargado exitosamente con', selectedProvider)

                // Mostrar sugerencia de GPU si está usando CPU
                if (selectedProvider === 'cpu') {
                    setShowGpuSuggestion(true)
                    // Auto-ocultar la sugerencia después de 10 segundos
                    setTimeout(() => setShowGpuSuggestion(false), 10000)
                } else {
                    setShowGpuSuggestion(false)
                }
            } catch (err) {
                console.error('Error cargando el modelo:', err)
                setError('Error cargando el modelo: ' + (err as Error).message)
                setIsModelLoaded(false)
            }
        }

        // Solo cargar si hay un cambio real en el provider
        if (
            !isModelLoaded ||
            detector.getCurrentProvider() !== selectedProvider
        ) {
            loadModel()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [detector, selectedProvider])

    const handleProviderChange = useCallback(
        async (newProvider: ExecutionProvider) => {
            if (newProvider === selectedProvider) return

            console.log(
                'Cambiando provider de',
                selectedProvider,
                'a',
                newProvider
            )
            setIsModelLoaded(false)
            setError('Cambiando backend...')

            try {
                await detector.switchExecutionProvider(newProvider)
                setSelectedProvider(newProvider)
                setIsModelLoaded(true)
                setError(null)
                console.log('Provider cambiado exitosamente a:', newProvider)

                // Mostrar sugerencia de GPU si cambió a CPU
                if (newProvider === 'cpu') {
                    setShowGpuSuggestion(true)
                    setTimeout(() => setShowGpuSuggestion(false), 10000)
                } else {
                    setShowGpuSuggestion(false)
                }
            } catch (error) {
                console.error('Error cambiando provider:', error)
                setError(
                    `Error cambiando a ${newProvider}: ${
                        (error as Error).message
                    }`
                )
                setIsModelLoaded(false)

                // Si falla, intentar volver a CPU como fallback
                if (newProvider !== 'cpu') {
                    console.log('Fallback a CPU...')
                    try {
                        await detector.switchExecutionProvider('cpu')
                        setSelectedProvider('cpu')
                        setIsModelLoaded(true)
                        setError(null)
                        console.log('Fallback a CPU exitoso')

                        // Mostrar sugerencia de GPU cuando hace fallback a CPU
                        setShowGpuSuggestion(true)
                        setTimeout(() => setShowGpuSuggestion(false), 15000) // Más tiempo ya que fue fallback
                    } catch (fallbackError) {
                        console.error('Error en fallback:', fallbackError)
                        setError(
                            `Error crítico: ${(fallbackError as Error).message}`
                        )
                    }
                }
            }
        },
        [detector, selectedProvider]
    )

    const handleReloadModel = useCallback(async () => {
        if (!isModelLoaded) return

        console.log(
            'Recargando modelo manualmente con provider:',
            selectedProvider
        )
        setIsModelLoaded(false)
        setError('Recargando modelo...')

        try {
            await detector.switchExecutionProvider(selectedProvider)
            setIsModelLoaded(true)
            setError(null)
            console.log('Modelo recargado exitosamente')

            // Mostrar sugerencia de GPU si está recargando con CPU
            if (selectedProvider === 'cpu') {
                setShowGpuSuggestion(true)
                setTimeout(() => setShowGpuSuggestion(false), 10000)
            } else {
                setShowGpuSuggestion(false)
            }
        } catch (error) {
            console.error('Error recargando modelo:', error)
            setError(`Error recargando modelo: ${(error as Error).message}`)
            setIsModelLoaded(false)
        }
    }, [detector, selectedProvider, isModelLoaded])

    const handleImageData = useCallback(
        async (imageData: ImageData, videoElement?: HTMLVideoElement) => {
            if (!isModelLoaded) return

            try {
                // Procesar en el siguiente tick para no bloquear la UI
                setTimeout(async () => {
                    try {
                        const result = await detector.detect(imageData)
                        setDetections(result.detections)
                        setStats(result.stats)

                        // Mostrar sugerencia de GPU si el rendimiento es muy bajo y está usando CPU
                        if (
                            selectedProvider === 'cpu' &&
                            result.stats.fps < 5 &&
                            result.stats.fps > 0
                        ) {
                            setShowGpuSuggestion(true)
                            setTimeout(() => setShowGpuSuggestion(false), 12000)
                        }

                        // Verificar si hay una cara Y un documento con >80% de confianza para capturar foto
                        const highConfidenceFace = result.detections.find(
                            (detection) =>
                                detection.class_name === 'face' &&
                                detection.score >= 0.8
                        )
                        const highConfidenceDoc = result.detections.find(
                            (detection) =>
                                detection.class_name === 'doc_quad' &&
                                detection.score >= 0.8
                        )

                        if (
                            highConfidenceFace &&
                            highConfidenceDoc &&
                            videoElement &&
                            !showCapture &&
                            autoCapture
                        ) {
                            const faceConfidence = (
                                highConfidenceFace.score * 100
                            ).toFixed(1)
                            const docConfidence = (
                                highConfidenceDoc.score * 100
                            ).toFixed(1)
                            console.log(
                                `¡Cara detectada con ${faceConfidence}% y documento con ${docConfidence}% de confianza! Capturando foto de alta calidad...`
                            )

                            // Usar la nueva función de captura de alta calidad si está disponible
                            const triggerCapture = (
                                window as unknown as Record<string, unknown>
                            ).triggerHighQualityCapture as
                                | (() => Promise<void>)
                                | undefined

                            if (triggerCapture) {
                                // Configurar el callback para cuando la captura esté lista
                                ;(
                                    window as unknown as Record<string, unknown>
                                ).onHighQualityCaptureReady = (
                                    canvas: HTMLCanvasElement
                                ) => {
                                    // Obtener la imagen limpia en formato JPG con alta calidad
                                    const capturedDataUrl = canvas.toDataURL(
                                        'image/jpeg',
                                        0.95
                                    )
                                    setCapturedImage(capturedDataUrl)
                                    // Usar la confianza más alta entre face y documento
                                    setCaptureConfidence(
                                        Math.max(
                                            highConfidenceFace.score,
                                            highConfidenceDoc.score
                                        )
                                    )
                                    setShowCapture(true)
                                }

                                // Activar la captura de alta calidad
                                triggerCapture().catch(console.error)
                            } else {
                                // Fallback al método anterior si la nueva función no está disponible
                                const tempCanvas =
                                    document.createElement('canvas')
                                tempCanvas.width = videoElement.videoWidth
                                tempCanvas.height = videoElement.videoHeight
                                const tempCtx = tempCanvas.getContext('2d')!

                                // Dibujar solo el video, sin las detecciones
                                tempCtx.drawImage(videoElement, 0, 0)

                                // Obtener la imagen limpia en formato JPG
                                const capturedDataUrl = tempCanvas.toDataURL(
                                    'image/jpeg',
                                    0.9
                                )
                                setCapturedImage(capturedDataUrl)
                                // Usar la confianza más alta entre face y documento
                                setCaptureConfidence(
                                    Math.max(
                                        highConfidenceFace.score,
                                        highConfidenceDoc.score
                                    )
                                )
                                setShowCapture(true)
                            }
                        }
                    } catch (err) {
                        console.error('Error en detección:', err)
                    }
                }, 0)
            } catch (err) {
                console.error('Error configurando detección:', err)
            }
        },
        [detector, isModelLoaded, showCapture, autoCapture, selectedProvider]
    )

    const handleRetakePhoto = useCallback(() => {
        setCapturedImage(null)
        setShowCapture(false)
        setCaptureConfidence(0)
        console.log('Preparado para capturar nueva foto...')
    }, [])

    const handleHighQualityCapture = useCallback(
        (canvas: HTMLCanvasElement) => {
            // Esta función se llamará cuando la captura de alta calidad esté lista
            const callback = (window as unknown as Record<string, unknown>)
                .onHighQualityCaptureReady as
                | ((canvas: HTMLCanvasElement) => void)
                | undefined
            if (callback) {
                callback(canvas)
            }
        },
        []
    )

    const handleSavePhoto = useCallback(() => {
        console.log('Foto guardada exitosamente')
        // La foto ya se descarga en el componente PhotoCapture
        // Aquí podrías agregar lógica adicional si es necesario
    }, [])

    const handleCloseGpuSuggestion = useCallback(() => {
        setShowGpuSuggestion(false)
    }, [])

    const handleSwitchToGpu = useCallback(async () => {
        setShowGpuSuggestion(false)
        await handleProviderChange('webgpu')
    }, [handleProviderChange])

    const handleCloseCapture = useCallback(() => {
        setCapturedImage(null)
        setShowCapture(false)
        setCaptureConfidence(0)
    }, [])

    return (
        <div className='app'>
            <header className='app-header'>
                <h1>🔍 Detector de Documentos y Caras</h1>
                <p>Detección en tiempo real usando YOLOX + ONNX.js</p>
                {error && <div className='error-banner'>⚠️ {error}</div>}
                {showGpuSuggestion && (
                    <div className='gpu-suggestion-banner'>
                        <div className='gpu-suggestion-content'>
                            <div className='gpu-suggestion-icon'>⚡</div>
                            <div className='gpu-suggestion-text'>
                                <strong>¡Mejora el rendimiento!</strong>
                                <p>
                                    Estás usando CPU. Para mejor rendimiento,
                                    activa la aceleración por GPU cambiando a{' '}
                                    <strong>WebGPU</strong> o{' '}
                                    <strong>WebGL</strong> en la configuración
                                    del modelo.
                                </p>
                                <div className='gpu-suggestion-benefits'>
                                    <span>✅ Procesamiento más rápido</span>
                                    <span>✅ Mayor FPS</span>
                                    <span>✅ Mejor experiencia</span>
                                </div>
                                <div className='gpu-suggestion-actions'>
                                    <button
                                        className='gpu-suggestion-action-btn'
                                        onClick={handleSwitchToGpu}
                                    >
                                        🚀 Cambiar a WebGPU ahora
                                    </button>
                                </div>
                            </div>
                            <button
                                className='gpu-suggestion-close'
                                onClick={handleCloseGpuSuggestion}
                                title='Cerrar sugerencia'
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                )}
                {showCapture && (
                    <div className='capture-notification'>
                        📸 ¡Foto capturada! Cara y documento detectados con alta
                        confianza ({(captureConfidence * 100).toFixed(1)}% máx.)
                    </div>
                )}
            </header>

            <main className='app-main'>
                <div className='video-section'>
                    <WebcamCanvas
                        detections={detections}
                        onImageData={handleImageData}
                        isModelLoaded={isModelLoaded}
                        onHighQualityCapture={handleHighQualityCapture}
                    />

                    {/* Switch para captura automática */}
                    <div className='auto-capture-control'>
                        <label className='switch-container'>
                            <span className='switch-label'>
                                📸 Captura Automática
                            </span>
                            <label className='switch'>
                                <input
                                    type='checkbox'
                                    checked={autoCapture}
                                    onChange={(e) =>
                                        setAutoCapture(e.target.checked)
                                    }
                                />
                                <span className='slider'></span>
                            </label>
                        </label>
                        {autoCapture && (
                            <p className='capture-info'>
                                Se capturará automáticamente cuando detecte
                                TANTO una cara como un documento con ≥80% de
                                confianza
                            </p>
                        )}
                    </div>
                </div>

                <div className='stats-section'>
                    <ModelConfig
                        selectedProvider={selectedProvider}
                        onProviderChange={handleProviderChange}
                        isModelLoaded={isModelLoaded}
                        onReloadModel={handleReloadModel}
                    />

                    <PerformanceMonitor
                        stats={stats}
                        isModelLoaded={isModelLoaded}
                        provider={selectedProvider}
                    />

                    <StatsDisplay
                        detections={detections}
                        stats={stats}
                        isModelLoaded={isModelLoaded}
                    />
                </div>
            </main>

            <footer className='app-footer'>
                <p>
                    💡 Asegúrate de colocar el archivo{' '}
                    <code>yolox_doc_face_decoded.onnx</code> en la carpeta{' '}
                    <code>public/</code>
                </p>
            </footer>

            {/* Componente de captura de fotos */}
            <PhotoCapture
                capturedImage={capturedImage}
                confidence={captureConfidence}
                onRetake={handleRetakePhoto}
                onSave={handleSavePhoto}
                onClose={handleCloseCapture}
            />
        </div>
    )
}

export default App
