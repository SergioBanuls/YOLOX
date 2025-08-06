import type { Detection, ModelStats } from '../types/detection'
import { ort } from './onnx-config'

type InferenceSession = ort.InferenceSession
type Tensor = ort.Tensor
export type ExecutionProvider = 'cpu' | 'webgl' | 'wasm' | 'webgpu'

export class YOLOXDetector {
    private session: InferenceSession | null = null
    private inputShape = [640, 640]
    private classes = ['face', 'doc_quad']
    private isLoaded = false
    private currentProvider: ExecutionProvider = 'cpu'

    async loadModel(
        modelPath: string,
        provider: ExecutionProvider = 'cpu'
    ): Promise<void> {
        try {
            console.log('Cargando modelo ONNX con backend:', provider)
            this.currentProvider = provider

            // Configurar opciones de sesión según el provider
            const executionProviders = this.getExecutionProviders(provider)
            const sessionOptions = {
                executionProviders,
                graphOptimizationLevel:
                    provider === 'cpu'
                        ? ('disabled' as const)
                        : ('basic' as const),
                enableCpuMemArena: provider === 'cpu',
                enableMemPattern: provider !== 'cpu',
            }

            console.log('Usando providers:', executionProviders)
            this.session = await ort.InferenceSession.create(
                modelPath,
                sessionOptions
            )
            this.isLoaded = true
            console.log('Modelo cargado exitosamente con', provider)
            if (this.session) {
                console.log('Inputs:', this.session.inputNames)
                console.log('Outputs:', this.session.outputNames)
            }
        } catch (error) {
            console.error('Error cargando el modelo:', error)
            this.isLoaded = false
            throw error
        }
    }

    private getExecutionProviders(provider: ExecutionProvider): string[] {
        switch (provider) {
            case 'cpu':
                return ['cpu']
            case 'webgl':
                return ['webgl', 'cpu'] // Fallback a CPU si WebGL falla
            case 'wasm':
                return ['wasm', 'cpu']
            case 'webgpu':
                return ['webgpu', 'webgl', 'cpu'] // Múltiples fallbacks
            default:
                return ['cpu']
        }
    }

    getCurrentProvider(): ExecutionProvider {
        return this.currentProvider
    }

    async switchExecutionProvider(provider: ExecutionProvider): Promise<void> {
        if (this.currentProvider === provider) {
            console.log(`Ya usando execution provider: ${provider}`)
            return
        }

        console.log(
            `Cambiando execution provider de ${this.currentProvider} a ${provider}`
        )

        try {
            // Liberar sesión actual si existe
            if (this.session) {
                console.log('Liberando sesión anterior...')
                await this.session.release()
                this.session = null
                this.isLoaded = false
            }

            this.currentProvider = provider

            // Crear nueva sesión con el provider seleccionado
            console.log(`Cargando modelo con ${provider}...`)
            await this.loadModel('/yolox_doc_face_decoded.onnx', provider)

            console.log(`✅ Execution provider cambiado a: ${provider}`)
        } catch (error) {
            console.error(
                `❌ Error cambiando execution provider a ${provider}:`,
                error
            )

            // Fallback a CPU si falla
            if (provider !== 'cpu') {
                console.log('🔄 Fallback a CPU...')
                this.currentProvider = 'cpu'
                try {
                    await this.loadModel('/yolox_doc_face_decoded.onnx', 'cpu')
                    console.log('✅ Fallback a CPU exitoso')
                } catch (fallbackError) {
                    console.error('❌ Error en fallback a CPU:', fallbackError)
                    throw fallbackError
                }
            } else {
                throw error
            }
        }
    }

    async reloadModel(
        modelPath: string,
        newProvider: ExecutionProvider
    ): Promise<void> {
        if (this.session) {
            this.session.release()
            this.session = null
            this.isLoaded = false
        }
        await this.loadModel(modelPath, newProvider)
    }

    async detect(
        imageData: ImageData
    ): Promise<{ detections: Detection[]; stats: ModelStats }> {
        if (!this.session || !this.isLoaded) {
            throw new Error('Modelo no cargado')
        }

        const startTime = performance.now()

        // Preprocesar imagen
        const preprocessed = this.preprocessImage(imageData)

        console.log('Ejecutando inferencia...')

        // Ejecutar inferencia
        const output = await this.session.run({ images: preprocessed })

        // Obtener el primer (y único) output
        const outputKeys = Object.keys(output)
        console.log('Output map keys:', outputKeys)

        const firstKey = outputKeys[0]
        console.log('First output key:', firstKey)

        const outputTensor = output[firstKey]
        const outputData = outputTensor.data as Float32Array

        console.log('Inferencia completada')
        console.log('Forma de salida:', outputTensor.dims)
        console.log('Tamaño de datos:', outputData.length)
        console.log('Primeros 10 valores:', Array.from(outputData.slice(0, 10)))

        // La forma debe ser [1, 8400, 7] = [batch, num_anchors, data_per_anchor]
        // Donde data_per_anchor = [cx, cy, w, h, objectness, class1_prob, class2_prob]

        if (
            outputTensor.dims.length !== 3 ||
            outputTensor.dims[0] !== 1 ||
            outputTensor.dims[2] !== 7
        ) {
            console.error('Formato de salida inesperado:', outputTensor.dims)
            throw new Error(
                `Formato de salida inesperado: ${outputTensor.dims}`
            )
        }

        // Verificar rango de valores
        const maxVal = Math.max(...Array.from(outputData.slice(0, 1000)))
        const minVal = Math.min(...Array.from(outputData.slice(0, 1000)))
        console.log(
            `Rango de valores: ${minVal.toFixed(6)} - ${maxVal.toFixed(6)}`
        )

        // Post-procesar resultados
        const { detections, stats } = this.postprocess(
            outputData,
            imageData.width,
            imageData.height
        )

        const processingTime = performance.now() - startTime
        stats.processingTimeMs = processingTime
        stats.fps = 1000 / processingTime

        return { detections, stats }
    }

    private preprocessImage(imageData: ImageData): Tensor {
        const { width, height } = imageData

        // Calcular ratio para mantener aspecto
        const scale = Math.min(
            this.inputShape[0] / width,
            this.inputShape[1] / height
        )
        const newWidth = Math.round(width * scale)
        const newHeight = Math.round(height * scale)

        // Crear canvas para redimensionar
        const canvas = document.createElement('canvas')
        canvas.width = this.inputShape[0]
        canvas.height = this.inputShape[1]
        const ctx = canvas.getContext('2d')!

        // CAMBIO CRÍTICO: Usar gris 114 como en YOLOX
        ctx.fillStyle = 'rgb(114, 114, 114)'
        ctx.fillRect(0, 0, this.inputShape[0], this.inputShape[1])

        // Crear ImageData temporal para la imagen original
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = width
        tempCanvas.height = height
        const tempCtx = tempCanvas.getContext('2d')!
        tempCtx.putImageData(imageData, 0, 0)

        // Dibujar imagen redimensionada centrada
        const offsetX = (this.inputShape[0] - newWidth) / 2
        const offsetY = (this.inputShape[1] - newHeight) / 2
        ctx.drawImage(tempCanvas, offsetX, offsetY, newWidth, newHeight)

        // Obtener datos redimensionados
        const resizedImageData = ctx.getImageData(
            0,
            0,
            this.inputShape[0],
            this.inputShape[1]
        )

        // CAMBIO CRÍTICO: Organización correcta del tensor en formato CHW
        const channelSize = this.inputShape[0] * this.inputShape[1]
        const tensor = new Float32Array(3 * channelSize)

        // YOLOX espera BGR y valores [0-255] sin normalizar
        // Organizar en formato CHW (Channel, Height, Width)
        for (let y = 0; y < this.inputShape[0]; y++) {
            for (let x = 0; x < this.inputShape[1]; x++) {
                const pixelIndex = (y * this.inputShape[1] + x) * 4

                const r = resizedImageData.data[pixelIndex]
                const g = resizedImageData.data[pixelIndex + 1]
                const b = resizedImageData.data[pixelIndex + 2]

                const tensorIndex = y * this.inputShape[1] + x

                // Canal B (índice 0)
                tensor[0 * channelSize + tensorIndex] = b
                // Canal G (índice 1)
                tensor[1 * channelSize + tensorIndex] = g
                // Canal R (índice 2)
                tensor[2 * channelSize + tensorIndex] = r
            }
        }

        console.log('Preprocesamiento YOLOX:')
        console.log('- Padding color: 114')
        console.log('- Orden de canales: BGR')
        console.log('- Formato: CHW')
        console.log('- Valores: [0-255] sin normalizar')
        console.log('- Scale:', scale)
        console.log('- Offset X:', offsetX, 'Offset Y:', offsetY)

        // Guardar valores para postprocesamiento
        this.lastScale = scale
        this.lastOffsetX = offsetX
        this.lastOffsetY = offsetY

        return new ort.Tensor('float32', tensor, [
            1,
            3,
            this.inputShape[0],
            this.inputShape[1],
        ])
    }

    // Agregar propiedades para guardar offsets
    private lastScale = 1.0
    private lastOffsetX = 0
    private lastOffsetY = 0

    private postprocess(
        outputData: Float32Array,
        originalWidth: number,
        originalHeight: number
    ): { detections: Detection[]; stats: ModelStats } {
        const startTime = performance.now()
        const numDetections = 8400
        const numClasses = 2
        const dataPerAnchor = 7

        console.log('Post-procesando con YOLOX decode_in_inference=True')

        const candidates: Detection[] = []
        const objectnessValues: number[] = []

        // Usar el scale guardado del preprocesamiento
        const scale = this.lastScale
        const offsetX = this.lastOffsetX
        const offsetY = this.lastOffsetY

        let highConfidenceCount = 0

        for (let i = 0; i < numDetections; i++) {
            const baseIndex = i * dataPerAnchor

            const cx = outputData[baseIndex]
            const cy = outputData[baseIndex + 1]
            const w = outputData[baseIndex + 2]
            const h = outputData[baseIndex + 3]
            const objectness = outputData[baseIndex + 4]
            const faceProb = outputData[baseIndex + 5]
            const docProb = outputData[baseIndex + 6]

            objectnessValues.push(objectness)

            // CAMBIO CRÍTICO: Usar mismo threshold de objectness que Python
            if (objectness > 0.5) {
                highConfidenceCount++

                // Calcular scores finales como en Python
                const faceScore = objectness * faceProb
                const docScore = objectness * docProb

                // Encontrar la clase con mayor score
                const scores = [faceScore, docScore]
                const maxScore = Math.max(...scores)
                const classId = scores.indexOf(maxScore)

                // CAMBIO IMPORTANTE: Usar threshold uniforme de 0.6 como en Python
                // En lugar de thresholds diferentes por clase
                const threshold = 0.6

                // CAMBIO: Aplicar el mismo filtro dual que Python
                if (maxScore > threshold && objectness > 0.5) {
                    // Log de las primeras detecciones
                    if (candidates.length < 5) {
                        console.log(`Candidato ${candidates.length + 1}:`, {
                            cx,
                            cy,
                            w,
                            h,
                            objectness: objectness.toFixed(4),
                            faceProb: faceProb.toFixed(4),
                            docProb: docProb.toFixed(4),
                            maxScore: maxScore.toFixed(4),
                            class: this.classes[classId],
                            threshold: threshold,
                        })
                    }

                    // Verificar coordenadas razonables como en Python
                    if (
                        w > 5 &&
                        h > 5 &&
                        w < 1000 &&
                        h < 1000 &&
                        cx > 0 &&
                        cy > 0 &&
                        cx < 640 &&
                        cy < 640
                    ) {
                        // Convertir de cxcywh a xyxy
                        const x1 = cx - w / 2
                        const y1 = cy - h / 2
                        const x2 = cx + w / 2
                        const y2 = cy + h / 2

                        // Convertir coordenadas al espacio original
                        const x1_orig = (x1 - offsetX) / scale
                        const y1_orig = (y1 - offsetY) / scale
                        const x2_orig = (x2 - offsetX) / scale
                        const y2_orig = (y2 - offsetY) / scale

                        // Clip a los límites de la imagen
                        const detection: Detection = {
                            x1: Math.max(0, Math.min(originalWidth, x1_orig)),
                            y1: Math.max(0, Math.min(originalHeight, y1_orig)),
                            x2: Math.max(0, Math.min(originalWidth, x2_orig)),
                            y2: Math.max(0, Math.min(originalHeight, y2_orig)),
                            score: maxScore,
                            class_id: classId,
                            class_name: this.classes[classId],
                        }

                        // Verificar que el box es válido
                        if (
                            detection.x2 > detection.x1 &&
                            detection.y2 > detection.y1
                        ) {
                            candidates.push(detection)
                        }
                    }
                }
            }
        }

        console.log(`Análisis de detecciones:`)
        console.log(`- Objectness > 0.5: ${highConfidenceCount}`)
        console.log(`- Candidatos válidos: ${candidates.length}`)
        console.log(`- Threshold uniforme: 0.6 para todas las clases`)
        console.log(
            `- Top scores:`,
            candidates
                .slice(0, 5)
                .map((d) => `${d.class_name}:${d.score.toFixed(3)}`)
                .join(', ')
        )

        // Aplicar NMS con threshold 0.3 como en Python
        const finalDetections: Detection[] = []

        for (let classId = 0; classId < numClasses; classId++) {
            const classDetections = candidates.filter(
                (det) => det.class_id === classId
            )
            const nmsDetections = this.applyNMS(classDetections, 0.3)
            finalDetections.push(...nmsDetections)
        }

        console.log(
            `Detecciones finales después de NMS: ${finalDetections.length}`
        )
        finalDetections.forEach((det, i) => {
            console.log(
                `${i + 1}. ${det.class_name}: score=${det.score.toFixed(3)}, ` +
                    `box=(${det.x1.toFixed(0)},${det.y1.toFixed(
                        0
                    )},${det.x2.toFixed(0)},${det.y2.toFixed(0)})`
            )
        })

        const endTime = performance.now()
        const processingTime = endTime - startTime

        // Calcular estadísticas
        const stats: ModelStats = {
            objectnessMin: Math.min(...objectnessValues),
            objectnessMax: Math.max(...objectnessValues),
            objectnessMean:
                objectnessValues.reduce((a, b) => a + b, 0) /
                objectnessValues.length,
            totalDetections: numDetections,
            validDetections: finalDetections.length,
            faceDetections: finalDetections.filter(
                (d) => d.class_name === 'face'
            ).length,
            docDetections: finalDetections.filter(
                (d) => d.class_name === 'doc_quad'
            ).length,
            processingTimeMs: Math.round(processingTime),
            fps: processingTime > 0 ? Math.round(1000 / processingTime) : 0,
        }

        return { detections: finalDetections, stats }
    }

    private applyNMS(
        detections: Detection[],
        iouThreshold: number
    ): Detection[] {
        if (detections.length === 0) return []

        // Crear una copia y ordenar por score descendente
        const sorted = [...detections].sort((a, b) => b.score - a.score)
        const keep: Detection[] = []
        const suppressed = new Set<number>()

        for (let i = 0; i < sorted.length; i++) {
            if (suppressed.has(i)) continue

            keep.push(sorted[i])

            // Comparar con detecciones posteriores
            for (let j = i + 1; j < sorted.length; j++) {
                if (suppressed.has(j)) continue

                const iou = this.calculateIOU(sorted[i], sorted[j])
                if (iou > iouThreshold) {
                    suppressed.add(j)
                }
            }
        }

        console.log(`NMS: ${detections.length} → ${keep.length} detecciones`)
        return keep
    }

    private calculateIOU(det1: Detection, det2: Detection): number {
        const x1 = Math.max(det1.x1, det2.x1)
        const y1 = Math.max(det1.y1, det2.y1)
        const x2 = Math.min(det1.x2, det2.x2)
        const y2 = Math.min(det1.y2, det2.y2)

        if (x2 <= x1 || y2 <= y1) return 0

        const intersection = (x2 - x1) * (y2 - y1)
        const area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        const area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        const union = area1 + area2 - intersection

        return intersection / union
    }

    isModelLoaded(): boolean {
        return this.isLoaded
    }
}
