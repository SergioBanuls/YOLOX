// Configuración simple de ONNX.js usando solo CPU/JS
import * as ort from 'onnxruntime-web'

// Configurar ONNX.js para usar solo el backend de CPU (JavaScript puro)
ort.env.logLevel = 'info'

console.log('ONNX configurado para usar CPU/JS')

export { ort }
