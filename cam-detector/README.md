# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

-   [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
-   [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

# 🔍 Webcam Document Detector

Detector de documentos y caras en tiempo real usando YOLOX + ONNX.js con React y TypeScript.

## 🚀 Características

-   **Detección en tiempo real**: Procesa feed de webcam usando YOLOX
-   **Modelo ONNX**: Inferencia en el navegador sin servidor
-   **Detección dual**: Reconoce caras y documentos
-   **Estadísticas en vivo**: Muestra métricas detalladas del modelo
-   **Interfaz moderna**: UI responsiva con React + TypeScript

## 🛠️ Tecnologías

-   **React 19** con TypeScript
-   **Vite** para desarrollo rápido
-   **ONNX.js** para inferencia en navegador
-   **PNPM** para gestión de paquetes
-   **Canvas API** para dibujo de bounding boxes

## 📋 Requisitos

-   Node.js 18+
-   PNPM
-   Navegador con soporte para WebRTC
-   Cámara web

## 🚀 Instalación

```bash
# Instalar dependencias
pnpm install

# Asegurarse de que el modelo ONNX esté en public/
# El archivo yolox_doc_face_decoded.onnx debe estar en public/

# Iniciar desarrollo
pnpm dev
```

## 📁 Estructura del Proyecto

```
src/
├── components/
│   ├── WebcamCanvas.tsx    # Componente de cámara y detección
│   └── StatsDisplay.tsx    # Panel de estadísticas
├── types/
│   └── detection.ts       # Tipos TypeScript
├── utils/
│   └── detector.ts        # Lógica del detector YOLOX
├── App.tsx                # Componente principal
└── App.css               # Estilos
```

## 🎯 Funcionalidades

### Detección

-   **Caras**: Marcadas en verde con confianza >60%
-   **Documentos**: Marcados en rojo con confianza >60%
-   **NMS**: Supresión de no-máximos para evitar duplicados

### Estadísticas

-   FPS en tiempo real
-   Tiempo de procesamiento
-   Contadores de detecciones por clase
-   Métricas de objectness del modelo
-   Coordenadas y tamaños de detecciones

### Interfaz

-   Vista de cámara con bounding boxes
-   Panel lateral con estadísticas detalladas
-   Leyenda de colores
-   Diseño responsivo

## 🔧 Configuración del Modelo

El modelo YOLOX debe:

-   Estar exportado con `decode_in_inference=True`
-   Formato ONNX con opset 14+
-   Entrada: (1, 3, 640, 640)
-   Salida: (1, 8400, 7) - [cx, cy, w, h, objectness, face_prob, doc_prob]

## 📱 Uso

1. Abrir la aplicación en el navegador
2. Permitir acceso a la cámara
3. Esperar a que cargue el modelo ONNX
4. ¡Mostrar documentos o caras para ver las detecciones!

## 🎨 Personalización

### Cambiar clases

Modificar `classes` en `src/utils/detector.ts`:

```typescript
private classes = ['face', 'doc_quad']; // Tus clases aquí
```

### Ajustar thresholds

En `src/utils/detector.ts`:

```typescript
if (objectness > 0.5) { // Threshold objectness
if (finalScore > 0.6) { // Threshold final
```

### Colores de bounding boxes

En `src/components/WebcamCanvas.tsx`:

```typescript
const color = class_name === 'face' ? '#00ff00' : '#ff0000'
```

## 🐛 Troubleshooting

### Modelo no carga

-   Verificar que `yolox_doc_face_decoded.onnx` esté en `public/`
-   Comprobar consola del navegador para errores
-   Verificar formato del modelo (debe ser ONNX con decode_in_inference=True)

### Webcam no funciona

-   Verificar permisos de cámara en navegador
-   Probar en HTTPS (requerido para webcam en producción)
-   Comprobar que no hay otras apps usando la cámara

### Performance lenta

-   Reducir resolución de webcam
-   Ajustar frecuencia de procesamiento en `processFrame`
-   Verificar hardware (GPU/CPU)

## 📄 Licencia

MIT

## 🤝 Contribuir

¡Pull requests bienvenidos! Para cambios importantes, abrir issue primero.

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
    globalIgnores(['dist']),
    {
        files: ['**/*.{ts,tsx}'],
        extends: [
            // Other configs...
            // Enable lint rules for React
            reactX.configs['recommended-typescript'],
            // Enable lint rules for React DOM
            reactDom.configs.recommended,
        ],
        languageOptions: {
            parserOptions: {
                project: ['./tsconfig.node.json', './tsconfig.app.json'],
                tsconfigRootDir: import.meta.dirname,
            },
            // other options...
        },
    },
])
```
