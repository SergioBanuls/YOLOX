import React, { useRef, useEffect, useState, useCallback } from 'react'
import type { Detection, WebcamDimensions } from '../types/detection'

interface WebcamCanvasProps {
    detections: Detection[]
    onImageData: (imageData: ImageData, videoElement?: HTMLVideoElement) => void
    isModelLoaded: boolean
    onHighQualityCapture?: (canvas: HTMLCanvasElement) => void
}

export const WebcamCanvas: React.FC<WebcamCanvasProps> = ({
    detections,
    onImageData,
    isModelLoaded,
    onHighQualityCapture,
}) => {
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [isStreaming, setIsStreaming] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [dimensions, setDimensions] = useState<WebcamDimensions>({
        width: 640,
        height: 480,
    })
    const intervalRef = useRef<number | undefined>(undefined)
    const previousFrameRef = useRef<ImageData | null>(null)
    const motionThreshold = 15 // Threshold para detectar movimiento

    // Función para calcular la nitidez de una imagen usando el operador Sobel
    const calculateImageSharpness = useCallback(
        (imageData: ImageData): number => {
            const data = imageData.data
            const width = imageData.width
            const height = imageData.height
            let sharpness = 0

            // Sobel operator kernels
            const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
            const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

            // Muestrear solo una porción del centro de la imagen para eficiencia
            const startX = Math.floor(width * 0.25)
            const endX = Math.floor(width * 0.75)
            const startY = Math.floor(height * 0.25)
            const endY = Math.floor(height * 0.75)

            for (let y = startY + 1; y < endY - 1; y++) {
                for (let x = startX + 1; x < endX - 1; x++) {
                    let gx = 0,
                        gy = 0

                    // Aplicar kernels Sobel
                    for (let i = 0; i < 3; i++) {
                        for (let j = 0; j < 3; j++) {
                            const pixelIndex =
                                ((y + i - 1) * width + (x + j - 1)) * 4
                            const gray =
                                (data[pixelIndex] +
                                    data[pixelIndex + 1] +
                                    data[pixelIndex + 2]) /
                                3

                            gx += gray * sobelX[i * 3 + j]
                            gy += gray * sobelY[i * 3 + j]
                        }
                    }

                    // Magnitud del gradiente
                    sharpness += Math.sqrt(gx * gx + gy * gy)
                }
            }

            return sharpness / ((endX - startX) * (endY - startY))
        },
        []
    )

    // Función para detectar movimiento entre frames
    const detectMotion = useCallback((currentFrame: ImageData): number => {
        if (!previousFrameRef.current) {
            previousFrameRef.current = currentFrame
            return 0
        }

        const prevData = previousFrameRef.current.data
        const currData = currentFrame.data
        const width = currentFrame.width
        const height = currentFrame.height

        let totalDiff = 0
        let pixelCount = 0

        // Muestrear solo el centro de la imagen para eficiencia
        const startX = Math.floor(width * 0.3)
        const endX = Math.floor(width * 0.7)
        const startY = Math.floor(height * 0.3)
        const endY = Math.floor(height * 0.7)

        for (let y = startY; y < endY; y += 4) {
            // Saltar píxeles para mejor performance
            for (let x = startX; x < endX; x += 4) {
                const pixelIndex = (y * width + x) * 4

                const prevGray =
                    (prevData[pixelIndex] +
                        prevData[pixelIndex + 1] +
                        prevData[pixelIndex + 2]) /
                    3
                const currGray =
                    (currData[pixelIndex] +
                        currData[pixelIndex + 1] +
                        currData[pixelIndex + 2]) /
                    3

                totalDiff += Math.abs(currGray - prevGray)
                pixelCount++
            }
        }

        previousFrameRef.current = currentFrame
        return totalDiff / pixelCount
    }, [])

    // Función para capturar foto de alta calidad
    const captureHighQualityPhoto =
        useCallback(async (): Promise<HTMLCanvasElement | null> => {
            const video = videoRef.current
            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
                return null
            }

            console.log(
                'Iniciando captura de alta calidad con detección de movimiento...'
            )

            // Primer paso: esperar a que no haya movimiento
            let isStable = false
            let attempts = 0
            const maxWaitAttempts = 20 // Máximo 2 segundos de espera

            while (!isStable && attempts < maxWaitAttempts) {
                // Crear canvas temporal para verificar movimiento
                const tempCanvas = document.createElement('canvas')
                tempCanvas.width = video.videoWidth
                tempCanvas.height = video.videoHeight
                const tempCtx = tempCanvas.getContext('2d')!
                tempCtx.drawImage(video, 0, 0)

                const currentFrame = tempCtx.getImageData(
                    0,
                    0,
                    tempCanvas.width,
                    tempCanvas.height
                )
                const motionLevel = detectMotion(currentFrame)

                console.log(
                    `Intento ${
                        attempts + 1
                    }: nivel de movimiento = ${motionLevel.toFixed(2)}`
                )

                if (motionLevel < motionThreshold) {
                    isStable = true
                    console.log(
                        'Imagen estable detectada, procediendo con captura...'
                    )
                } else {
                    attempts++
                    await new Promise((resolve) => setTimeout(resolve, 100)) // Esperar 100ms
                }
            }

            if (!isStable) {
                console.log(
                    'No se pudo obtener imagen estable, procediendo de todos modos...'
                )
            }

            // Segundo paso: tomar múltiples frames en rápida sucesión para seleccionar el mejor
            const frames: { canvas: HTMLCanvasElement; sharpness: number }[] =
                []
            const frameCount = isStable ? 3 : 7 // Más frames si hay movimiento
            const frameDelay = 20 // Reducido para capturar más rápido

            for (let i = 0; i < frameCount; i++) {
                if (i > 0) {
                    await new Promise((resolve) =>
                        setTimeout(resolve, frameDelay)
                    )
                }

                // Crear canvas para este frame
                const canvas = document.createElement('canvas')
                canvas.width = video.videoWidth
                canvas.height = video.videoHeight
                const ctx = canvas.getContext('2d')!

                // Capturar frame actual
                ctx.drawImage(video, 0, 0)

                // Calcular "sharpness" del frame (basado en gradientes)
                const imageData = ctx.getImageData(
                    0,
                    0,
                    canvas.width,
                    canvas.height
                )
                const sharpness = calculateImageSharpness(imageData)

                frames.push({ canvas, sharpness })
            }

            // Seleccionar el frame con mayor nitidez
            const bestFrame = frames.reduce((best, current) =>
                current.sharpness > best.sharpness ? current : best
            )

            console.log(
                'Frames capturados:',
                frames.map((f) => f.sharpness.toFixed(2))
            )
            console.log(
                'Mejor frame seleccionado con sharpness:',
                bestFrame.sharpness.toFixed(2)
            )
            console.log('Imagen estable:', isStable ? 'Sí' : 'No')

            // Limpiar los canvas no utilizados
            frames.forEach((frame) => {
                if (frame.canvas !== bestFrame.canvas) {
                    // No necesitamos limpiar explícitamente, el GC se encargará
                }
            })

            return bestFrame.canvas
        }, [calculateImageSharpness, detectMotion, motionThreshold])

    // Exponer la función de captura de alta calidad
    useEffect(() => {
        if (onHighQualityCapture) {
            // Crear una función wrapper que llame a nuestra función de captura
            const triggerCapture = async () => {
                const canvas = await captureHighQualityPhoto()
                if (canvas) {
                    onHighQualityCapture(canvas)
                }
            }

            // Exponer la función al componente padre
            ;(
                window as unknown as Record<string, unknown>
            ).triggerHighQualityCapture = triggerCapture
        }
    }, [onHighQualityCapture, captureHighQualityPhoto])

    // Función simple para dibujar el video y las detecciones
    const draw = useCallback(() => {
        const video = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA)
            return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Limpiar y dibujar video
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Dibujar detecciones
        detections.forEach((detection) => {
            const { x1, y1, x2, y2, score, class_name } = detection

            // Escalar al tamaño del canvas
            const scaleX = canvas.width / video.videoWidth
            const scaleY = canvas.height / video.videoHeight

            const scaledX1 = x1 * scaleX
            const scaledY1 = y1 * scaleY
            const scaledX2 = x2 * scaleX
            const scaledY2 = y2 * scaleY

            // Color según la clase
            const color = class_name === 'face' ? '#00ff00' : '#ff0000'

            // Grosor de línea basado en confianza (para mostrar detecciones débiles)
            const lineWidth = Math.max(1, Math.min(3, score * 2000))

            // Dibujar bounding box
            ctx.strokeStyle = color
            ctx.lineWidth = lineWidth
            ctx.strokeRect(
                scaledX1,
                scaledY1,
                scaledX2 - scaledX1,
                scaledY2 - scaledY1
            )

            // Etiqueta con más precisión para valores bajos
            const percentage =
                score < 0.01
                    ? (score * 100).toFixed(3)
                    : (score * 100).toFixed(1)
            const label = `${class_name}: ${percentage}%`

            // Fondo semitransparente para mejor legibilidad
            ctx.fillStyle = color + '80'
            const textWidth = ctx.measureText(label).width
            ctx.fillRect(scaledX1, scaledY1 - 20, textWidth + 10, 20)

            // Texto
            ctx.fillStyle = 'white'
            ctx.font = '12px Arial'
            ctx.fillText(label, scaledX1 + 5, scaledY1 - 5)
        })
    }, [detections])

    // Configurar webcam
    useEffect(() => {
        let isMounted = true
        let stream: MediaStream | null = null
        const video = videoRef.current // Guardar referencia

        const startWebcam = async () => {
            try {
                console.log('Iniciando webcam...')
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280, min: 640 },
                        height: { ideal: 720, min: 480 },
                        facingMode: 'user',
                        frameRate: { ideal: 30, min: 15 },
                    },
                })

                console.log('Stream obtenido:', stream)

                // Verificar que el componente sigue montado
                if (!isMounted || !video) {
                    stream.getTracks().forEach((track) => track.stop())
                    return
                }

                video.srcObject = stream
                console.log('Stream asignado al video')

                video.onloadedmetadata = () => {
                    if (!isMounted || !video) return

                    const { videoWidth, videoHeight } = video
                    console.log('Video cargado:', videoWidth, 'x', videoHeight)
                    setDimensions({ width: videoWidth, height: videoHeight })

                    if (canvasRef.current) {
                        canvasRef.current.width = videoWidth
                        canvasRef.current.height = videoHeight
                        console.log(
                            'Canvas configurado:',
                            videoWidth,
                            'x',
                            videoHeight
                        )
                    }
                }

                // Esperar un poco antes de iniciar reproducción
                await new Promise((resolve) => setTimeout(resolve, 100))

                if (!isMounted || !video) {
                    stream.getTracks().forEach((track) => track.stop())
                    return
                }

                // Forzar reproducción del video de forma más segura
                try {
                    console.log('Intentando reproducir video...')
                    await video.play()
                    if (isMounted) {
                        console.log('Video iniciado correctamente')
                        setIsStreaming(true)
                        setError(null)
                    }
                } catch (playError) {
                    console.error('Error iniciando video:', playError)
                    if (isMounted) {
                        // Intentar de nuevo después de un breve delay
                        setTimeout(async () => {
                            if (isMounted && video) {
                                try {
                                    console.log(
                                        'Reintentando reproducir video...'
                                    )
                                    await video.play()
                                    setIsStreaming(true)
                                    setError(null)
                                } catch (retryError) {
                                    console.error(
                                        'Error en segundo intento:',
                                        retryError
                                    )
                                    setError(
                                        'Error iniciando video: ' +
                                            (retryError as Error).message
                                    )
                                }
                            }
                        }, 500)
                    }
                }
            } catch (err) {
                console.error('Error accediendo a la webcam:', err)
                if (isMounted) {
                    setError(
                        'Error accediendo a la webcam: ' +
                            (err as Error).message
                    )
                    setIsStreaming(false)
                }
            }
        }

        startWebcam()

        return () => {
            isMounted = false

            // Limpiar stream
            if (stream) {
                stream.getTracks().forEach((track) => track.stop())
            } else if (video?.srcObject) {
                const tracks = (video.srcObject as MediaStream).getTracks()
                tracks.forEach((track) => track.stop())
            }

            // Limpiar video
            if (video) {
                video.srcObject = null
            }
        }
    }, [])

    // Loop de dibujo - solo dibujar, muy simple
    useEffect(() => {
        if (!isStreaming) return

        const animate = () => {
            draw()
            requestAnimationFrame(animate)
        }

        const animationId = requestAnimationFrame(animate)

        return () => cancelAnimationFrame(animationId)
    }, [isStreaming, detections, draw])

    // Enviar datos al modelo - MUY poco frecuente
    useEffect(() => {
        if (!isStreaming || !isModelLoaded) return

        const processForModel = () => {
            const video = videoRef.current
            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) return

            const canvas = document.createElement('canvas')
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            const ctx = canvas.getContext('2d')!
            ctx.drawImage(video, 0, 0)

            const imageData = ctx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
            )
            onImageData(imageData, videoRef.current || undefined)
        }

        // Solo procesar cada 2 segundos para no saturar
        intervalRef.current = setInterval(processForModel, 2000)

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current)
            }
        }
    }, [isStreaming, isModelLoaded, onImageData])

    if (error) {
        return (
            <div className='error-container'>
                <p className='error-text'>{error}</p>
                <div className='error-help'>
                    <h4>💡 Posibles soluciones:</h4>
                    <ul>
                        <li>
                            Verifica que tu navegador tenga permisos para
                            acceder a la cámara
                        </li>
                        <li>
                            Asegúrate de que no hay otra aplicación usando la
                            cámara
                        </li>
                        <li>Intenta refrescar la página (F5)</li>
                        <li>
                            En Chrome/Edge: Ve a Configuración → Privacidad →
                            Configuración del sitio → Cámara
                        </li>
                        <li>
                            Verifica que la cámara esté conectada y funcionando
                        </li>
                    </ul>
                    <button
                        onClick={() => window.location.reload()}
                        className='retry-button'
                    >
                        🔄 Reintentar
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className='webcam-container'>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ display: 'none' }}
            />
            <canvas
                ref={canvasRef}
                className='webcam-canvas'
                width={dimensions.width}
                height={dimensions.height}
            />
            {!isStreaming && (
                <div className='loading-overlay'>
                    <p>Iniciando webcam...</p>
                </div>
            )}
            {!isModelLoaded && isStreaming && (
                <div className='model-loading-overlay'>
                    <p>Cargando modelo...</p>
                </div>
            )}
        </div>
    )
}
