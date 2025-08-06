// Configuración global para ONNX.js
declare global {
    interface Window {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ort: any
    }
}

export const initONNX = async () => {
    if (typeof window !== 'undefined' && !window.ort) {
        try {
            // Importar ONNX de forma dinámica
            const ort = await import('onnxruntime-web')

            // Configurar para uso en navegador
            ort.env.wasm.wasmPaths = '/wasm/'
            ort.env.wasm.numThreads = 1
            ort.env.wasm.simd = true // Permitir SIMD ya que es lo que tenemos
            ort.env.wasm.proxy = false
            ort.env.logLevel = 'info'

            window.ort = ort
            console.log('ONNX inicializado correctamente')
            return ort
        } catch (error) {
            console.error('Error inicializando ONNX:', error)
            throw error
        }
    }
    return window.ort
}
