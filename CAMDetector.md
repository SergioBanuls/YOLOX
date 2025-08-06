# CAMDetector - Detección en Tiempo Real

CAMDetector es una **aplicación web** desarrollada en **React + TypeScript** que utiliza el modelo ONNX exportado desde YOLOX para realizar **detección en tiempo real** a través de la webcam.

## 🎯 Funcionalidades principales

La aplicación web:

-   **Abre la webcam** del usuario
-   **Carga el modelo ONNX** exportado desde el proyecto YOLOX
-   **Detecta en tiempo real** dos tipos de objetos:
    -   `face` - Caras humanas
    -   `doc_quad` - Documentos (mostrados como "doc4" en la interfaz)
-   **Captura automática** cuando ambas detecciones superan el umbral de confianza

## 📁 Ubicación del modelo

**IMPORTANTE:** El modelo ONNX debe colocarse en la siguiente ubicación para que funcione correctamente:

```
cam-detector/
└── public/
    └── yolox_doc_face_decoded.onnx  # ← Modelo exportado desde YOLOX
```

## 🔧 Tecnologías utilizadas

-   **React 19** - Framework de interfaz de usuario
-   **TypeScript** - Tipado estático
-   **Vite** - Herramienta de desarrollo y build
-   **ONNX Runtime Web** - Ejecución de modelos ONNX en el navegador
-   **Canvas API** - Renderizado de detecciones en tiempo real

## ⚡ Captura automática

La aplicación incluye un **switch activable** en la interfaz que habilita la captura automática:

### 📷 Condiciones para captura automática:

-   **AMBAS detecciones** (cara Y documento) deben estar presentes
-   **Confianza ≥ 80-85%** para cada detección
-   El switch de "📸 Captura Automática" debe estar **activado**

### 🎛️ Interfaz de captura:

-   Switch claramente marcado en la interfaz
-   Mensaje informativo: _"Se capturará automáticamente cuando detecte TANTO una cara como un documento con ≥80% de confianza"_
-   Feedback visual del nivel de confianza

## 🚀 Instalación y ejecución

### 1. Instalar dependencias

```bash
cd cam-detector
npm install
```

### 2. Colocar el modelo ONNX

Copia el modelo exportado desde YOLOX:

```bash
# Desde la raíz del proyecto YOLOX
cp yolox_doc_face_fixed.onnx ../cam-detector/public/yolox_doc_face_decoded.onnx
```

### 3. Ejecutar en modo desarrollo

```bash
npm run dev
```

### 4. Construir para producción

```bash
npm run build
npm run preview
```

## 🖥️ Componentes principales

### `WebcamCanvas.tsx`

-   Maneja el stream de video de la webcam
-   Renderiza las detecciones en tiempo real
-   Gestiona la captura de frames para procesamiento

### `PhotoCapture.tsx`

-   Interfaz de captura automática
-   Visualización de imágenes capturadas
-   Funcionalidad de descarga

### `ModelConfig.tsx`

-   Configuración del modelo ONNX
-   Selección de proveedor de ejecución (CPU/GPU)
-   Monitoreo del estado del modelo

### `PerformanceMonitor.tsx` & `StatsDisplay.tsx`

-   Métricas de rendimiento en tiempo real
-   Estadísticas de detección
-   FPS y tiempos de procesamiento

## 🎮 Uso de la aplicación

1. **Abrir la aplicación** en el navegador
2. **Permitir acceso** a la webcam cuando se solicite
3. **Esperar** a que el modelo ONNX se cargue completamente
4. **Activar** el switch de "📸 Captura Automática" si se desea
5. **Posicionar** cara y documento frente a la cámara
6. La aplicación **capturará automáticamente** cuando detecte ambos objetos con alta confianza

## 🔍 Proveedores de ejecución

La aplicación soporta diferentes proveedores para ejecutar el modelo ONNX:

-   **WebGL** - Aceleración por GPU (recomendado)
-   **CPU** - Procesamiento por CPU (fallback)

## 📊 Métricas mostradas

-   **FPS** - Frames por segundo de procesamiento
-   **Tiempo de inferencia** - Latencia del modelo
-   **Número de detecciones** - Caras y documentos detectados
-   **Nivel de confianza** - Score de cada detección

## 🔗 Integración con YOLOX

Para actualizar el modelo:

1. Entrena un nuevo modelo en YOLOX
2. Exporta usando `export_correct_ONNX.py`
3. Copia el nuevo `.onnx` a `cam-detector/public/`
4. Reinicia la aplicación web

---

**Paso anterior:** Consulta [YOLOX.md](./YOLOX.md) para entrenar y exportar el modelo.
