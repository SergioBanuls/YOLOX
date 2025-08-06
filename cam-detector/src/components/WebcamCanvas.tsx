import React, { useRef, useEffect, useState, useCallback } from 'react'
import type { Detection, WebcamDimensions } from '../types/detection'

interface WebcamCanvasProps {
    detections: Detection[]
    onImageData: (imageData: ImageData) => void
    isModelLoaded: boolean
}

export const WebcamCanvas: React.FC<WebcamCanvasProps> = ({
    detections,
    onImageData,
    isModelLoaded,
}) => {
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [isStreaming, setIsStreaming] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [dimensions, setDimensions] = useState<WebcamDimensions>({
        width: 640,
        height: 480,
    })
    const animationFrameRef = useRef<number | undefined>(undefined)
    const lastProcessTimeRef = useRef<number>(0)
    const processingRef = useRef<boolean>(false)

    const drawDetections = useCallback(
        (ctx: CanvasRenderingContext2D) => {
            // Limpiar canvas
            ctx.clearRect(0, 0, dimensions.width, dimensions.height)

            // Dibujar video frame SIEMPRE
            if (videoRef.current && videoRef.current.readyState >= 2) {
                ctx.drawImage(
                    videoRef.current,
                    0,
                    0,
                    dimensions.width,
                    dimensions.height
                )
                // console.log('Video dibujado'); // Descomentar para debug
            } else {
                // Fondo gris si no hay video
                ctx.fillStyle = '#333'
                ctx.fillRect(0, 0, dimensions.width, dimensions.height)
                ctx.fillStyle = 'white'
                ctx.font = '16px Arial'
                ctx.textAlign = 'center'
                ctx.fillText(
                    'Esperando webcam...',
                    dimensions.width / 2,
                    dimensions.height / 2
                )
            }

            // Dibujar detecciones solo si las hay
            detections.forEach((detection) => {
                const { x1, y1, x2, y2, score, class_name } = detection

                // Escalar coordenadas al tamaño del canvas
                const scaleX =
                    dimensions.width /
                    (videoRef.current?.videoWidth || dimensions.width)
                const scaleY =
                    dimensions.height /
                    (videoRef.current?.videoHeight || dimensions.height)

                const scaledX1 = x1 * scaleX
                const scaledY1 = y1 * scaleY
                const scaledX2 = x2 * scaleX
                const scaledY2 = y2 * scaleY

                // Color según la clase
                const color = class_name === 'face' ? '#00ff00' : '#ff0000' // Verde para cara, rojo para documento

                // Dibujar bounding box
                ctx.strokeStyle = color
                ctx.lineWidth = 3
                ctx.strokeRect(
                    scaledX1,
                    scaledY1,
                    scaledX2 - scaledX1,
                    scaledY2 - scaledY1
                )

                // Dibujar etiqueta
                const label = `${class_name}: ${(score * 100).toFixed(1)}%`
                ctx.fillStyle = color
                ctx.font = '16px Arial'

                // Fondo para el texto
                const textMetrics = ctx.measureText(label)
                ctx.fillRect(
                    scaledX1,
                    scaledY1 - 25,
                    textMetrics.width + 10,
                    25
                )

                // Texto
                ctx.fillStyle = 'white'
                ctx.fillText(label, scaledX1 + 5, scaledY1 - 8)
            })
        },
        [detections, dimensions]
    )

    const processFrame = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) {
            animationFrameRef.current = requestAnimationFrame(processFrame)
            return
        }

        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        if (!ctx || video.readyState !== video.HAVE_ENOUGH_DATA) {
            animationFrameRef.current = requestAnimationFrame(processFrame)
            return
        }

        // Dibujar frame actual SIEMPRE para mantener fluidez
        drawDetections(ctx)

        // Solo procesar para el modelo ocasionalmente y de forma asíncrona
        const now = performance.now()
        if (
            isModelLoaded &&
            !processingRef.current &&
            now - lastProcessTimeRef.current > 1000
        ) {
            // Solo cada segundo

            processingRef.current = true
            lastProcessTimeRef.current = now

            // Procesar en el siguiente tick para no bloquear
            setTimeout(() => {
                if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
                    const tempCanvas = document.createElement('canvas')
                    tempCanvas.width = video.videoWidth
                    tempCanvas.height = video.videoHeight
                    const tempCtx = tempCanvas.getContext('2d')!
                    tempCtx.drawImage(video, 0, 0)

                    const imageData = tempCtx.getImageData(
                        0,
                        0,
                        video.videoWidth,
                        video.videoHeight
                    )
                    onImageData(imageData)
                }
                processingRef.current = false
            }, 0)
        }

        animationFrameRef.current = requestAnimationFrame(processFrame)
    }, [drawDetections, onImageData, isModelLoaded])

    useEffect(() => {
        let isMounted = true
        let stream: MediaStream | null = null
        const video = videoRef.current // Guardar referencia

        const startWebcam = async () => {
            try {
                console.log('Iniciando webcam...')
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user',
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

                    const width = video.videoWidth
                    const height = video.videoHeight
                    console.log('Video cargado:', width, 'x', height)
                    setDimensions({ width, height })

                    // Ajustar canvas al tamaño del video
                    if (canvasRef.current) {
                        canvasRef.current.width = width
                        canvasRef.current.height = height
                        console.log('Canvas configurado:', width, 'x', height)
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

            // Parar animación
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current)
            }

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

    useEffect(() => {
        if (isStreaming) {
            // Remover la dependencia de isModelLoaded
            processFrame()
        }

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current)
            }
        }
    }, [isStreaming, processFrame])

    if (error) {
        return (
            <div className='error-container'>
                <p className='error-text'>{error}</p>
                <p>
                    Asegúrate de que tu navegador tenga permisos para acceder a
                    la cámara.
                </p>
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
            {!isModelLoaded && (
                <div className='model-loading-overlay'>
                    <p>Cargando modelo...</p>
                </div>
            )}
        </div>
    )
}
