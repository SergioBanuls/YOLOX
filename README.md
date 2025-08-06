# YOLOX + CAMDetector

Este repositorio contiene dos proyectos complementarios para detección de objetos usando YOLOX:

## 📦 Proyectos incluidos

### 1. YOLOX

Proyecto de entrenamiento y exportación de modelos de detección basado en el framework YOLOX. Se encarga de entrenar un modelo personalizado para detectar **caras** y **documentos**, y exportarlo al formato ONNX para su uso en aplicaciones web.

**📁 Ubicación:** `./YOLOX/`  
**📄 Documentación detallada:** [YOLOX.md](./YOLOX.md)

### 2. CAMDetector

Aplicación web desarrollada en React + TypeScript que utiliza el modelo ONNX exportado desde YOLOX para realizar detección en tiempo real a través de la webcam. Incluye funcionalidades de captura automática cuando se detectan ambos objetos con alta confianza.

**📁 Ubicación:** `./cam-detector/`  
**📄 Documentación detallada:** [CAMDetector.md](./CAMDetector.md)

## 🔄 Flujo de trabajo

1. **Entrenar modelo** → Usar YOLOX para entrenar y exportar modelo ONNX
2. **Transferir modelo** → Copiar el archivo `.onnx` generado al proyecto CAMDetector
3. **Ejecutar detección** → Usar CAMDetector para detección en tiempo real

## 🚀 Inicio rápido

Para obtener instrucciones detalladas de configuración y uso de cada proyecto, consulta la documentación específica:

-   **[YOLOX.md](./YOLOX.md)** - Entrenamiento y exportación de modelos
-   **[CAMDetector.md](./CAMDetector.md)** - Aplicación web de detección

---

**Nota:** Este es un proyecto de detección especializado para casos de uso que requieren identificar simultáneamente caras y documentos en tiempo real.
