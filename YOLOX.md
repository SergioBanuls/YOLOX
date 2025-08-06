# YOLOX - Entrenamiento y Exportación de Modelos

Este proyecto se encarga de **entrenar un modelo YOLOX personalizado** para detectar **caras** (`face`) y **documentos** (`doc_quad`) y **exportarlo al formato ONNX** para su uso en aplicaciones web.

## 📋 Descripción general

El proyecto YOLOX permite:

-   Entrenar un modelo de detección personalizado usando el framework YOLOX
-   Exportar el modelo entrenado al formato ONNX
-   Validar y comparar la precisión entre el modelo PyTorch original y la versión ONNX exportada
-   Generar modelos optimizados para detección en tiempo real

## �️ Instalación y Preparación del Entorno

### Prerequisitos

-   **Anaconda** instalado en tu sistema
-   **GPU compatible con CUDA** (recomendado para entrenamiento)
-   **Windows 10/11** con soporte para GPU

### 1. Crear entorno conda

Abre **Anaconda Prompt** y ejecuta los siguientes comandos:

```bash
# Crear un nuevo entorno con Python 3.8
conda create -n yolox python=3.8 -y

# Activar el entorno
conda activate yolox
```

### 2. Instalar PyTorch con soporte CUDA

```bash
# Para GPU con CUDA 11.8 (verificar tu versión de CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Si no tienes GPU, usa la versión CPU:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 3. Instalar dependencias del proyecto

Navega al directorio del proyecto YOLOX y ejecuta:

```bash
# Navegar al directorio del proyecto
cd ruta\a\tu\proyecto\YOLOX

# Instalar YOLOX en modo desarrollo
pip install -v -e .

# Instalar dependencias adicionales
pip install cython
pip install pycocotools
pip install tensorboard
pip install onnx
pip install onnxruntime
```

### 4. Verificar instalación

Verifica que todo esté instalado correctamente:

```bash
# Verificar PyTorch y CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only')"

# Verificar YOLOX
python -c "import yolox; print('YOLOX instalado correctamente')"
```

### 5. Descargar modelo preentrenado

Descarga el modelo base YOLOX-S:

```bash
# Crear directorio para modelos si no existe
mkdir -p models

# Descargar modelo YOLOX-S preentrenado
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

**Nota:** Si `wget` no está disponible en Windows, puedes descargar manualmente desde [GitHub Releases](https://github.com/Megvii-BaseDetection/YOLOX/releases) y colocar `yolox_s.pth` en la raíz del proyecto.

### 6. Configuración del archivo de experimento

Asegúrate de que tienes el archivo de configuración personalizado:

```bash
# Verificar que existe el archivo de configuración
dir exps\example\custom\yolox_doc_face.py
```

Si no existe, créalo basándote en los archivos de ejemplo en `exps/example/yolox_voc/`.

### ⚠️ Notas importantes

-   **Siempre activa el entorno conda** antes de trabajar: `conda activate yolox`
-   **Verifica tu versión de CUDA** con `nvidia-smi` antes de instalar PyTorch
-   **Para entrenamientos largos**, considera usar `screen` o `tmux` para mantener la sesión activa
-   **El entrenamiento con CPU es extremadamente lento**, se recomienda encarecidamente usar GPU

## �🔧 Scripts principales

### `compare_models.py`

Script para **validar y comparar** el rendimiento entre:

-   Modelo PyTorch original (`.pth`)
-   Modelo ONNX exportado (`.onnx`)

Útil para verificar que la exportación se realizó correctamente y que ambos modelos producen resultados similares.

### `export_correct_ONNX.py`

Script principal para **exportar el modelo entrenado al formato ONNX**. Este script:

-   Carga el checkpoint del modelo entrenado
-   Configura correctamente las clases y parámetros
-   Exporta el modelo a formato ONNX optimizado para inferencia web
-   Genera el archivo `yolox_doc_face_fixed.onnx`

## 📁 Preparación del Dataset

**IMPORTANTE:** Antes de ejecutar cualquier entrenamiento, debes preparar correctamente tu dataset siguiendo estos pasos:

### 1. Estructura de directorios requerida

```
datasets/
└── VOCdevkit/
    ├── annotations_cache/  # Caché generado automáticamente
    └── VOC2020/
        ├── Annotations/    # Archivos .xml con anotaciones
        ├── JPEGImages/     # Imágenes del dataset
        └── ImageSets/
            └── Main/
                ├── train.txt  # Lista de archivos de entrenamiento
                └── val.txt    # Lista de archivos de validación
```

### 2. 📸 Preparación de las imágenes

-   **Coloca todas las imágenes en la carpeta `JPEGImages/`**
-   Las imágenes deben estar en formato JPEG (extensión `.jpg`)
-   **Cada imagen debe tener un archivo XML de anotación correspondiente con el mismo nombre**
    -   Ejemplo: `imagen001.jpg` → `imagen001.xml`

### 3. 📝 Formato de anotaciones XML

Cada archivo XML en la carpeta `Annotations/` debe seguir el formato VOC:

```xml
<annotation>
  <filename>imagen001.jpg</filename>
  <size>
    <width>2268</width>
    <height>4032</height>
    <depth>3</depth>
  </size>
  <object>
    <name>face</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
  <object>
    <name>doc_quad</name>
    <bndbox>
      <xmin>500</xmin>
      <ymin>600</ymin>
      <xmax>800</xmax>
      <ymax>900</ymax>
    </bndbox>
  </object>
  <!-- Agregar más objetos según sea necesario -->
</annotation>
```

### 4. Archivos de división del dataset

Genera dos archivos de texto en `ImageSets/Main/`:

#### `train.txt`

Contiene la lista de nombres de imágenes (sin extensión .xml) para entrenamiento:

```
imagen001
imagen002
imagen003
...
```

#### `val.txt`

Contiene la lista de nombres de imágenes (sin extensión .xml) para validación:

```
imagen004
imagen005
imagen006
...
```

**Notas importantes:**

-   **Cada línea debe contener solo el nombre del archivo sin extensión** (sin .jpg ni .xml)
-   Típicamente se usa una división 80/20 (80% entrenamiento, 20% validación)
-   Los nombres en estos archivos deben corresponder a imágenes y archivos XML existentes
-   Cada imagen listada debe tener tanto un archivo `.jpg` en `JPEGImages/` como un archivo `.xml` en `Annotations/`

### 5. ✅ Lista de verificación

Antes del entrenamiento, asegúrate de que:

-   [ ] Todas las imágenes están en la carpeta `JPEGImages/`
-   [ ] Todas las anotaciones XML están en la carpeta `Annotations/`
-   [ ] Cada imagen tiene un archivo XML correspondiente con el mismo nombre
-   [ ] Los archivos `train.txt` y `val.txt` están correctamente generados
-   [ ] Los archivos XML siguen el formato VOC correcto
-   [ ] No faltan archivos referenciados en train.txt o val.txt

### ⚠️ Importante sobre el caché

-   Si agregas **nuevas imágenes** al dataset, **DEBES BORRAR** el directorio `annotations_cache/` que se encuentra dentro de `VOCdevkit/`
-   Este caché puede causar problemas si no se actualiza después de modificar el dataset

## 🚀 Cómo usar

### 0. Activar entorno conda

**IMPORTANTE:** Siempre activa el entorno antes de ejecutar cualquier comando:

```bash
conda activate yolox
```

### 1. Preparar el dataset

Asegúrate de tener la estructura de directorios correcta con:

-   Imágenes en `VOCdevkit/VOC2020/JPEGImages/`
-   Anotaciones XML en `VOCdevkit/VOC2020/Annotations/`
-   Archivos de división en `VOCdevkit/VOC2020/ImageSets/Main/`

### 2. Ejecutar el entrenamiento

```bash
python tools/train.py -f exps/example/custom/yolox_doc_face.py -d 1 -b 8 --fp16 -o -c yolox_s.pth
```

### 3. Exportar a ONNX

```bash
python export_correct_ONNX.py
```

### 4. Validar el modelo exportado

```bash
python compare_models.py
```

## 🔧 Solución de Problemas Comunes

### Error de CUDA out of memory

```bash
# Reducir batch size en el comando de entrenamiento
python tools/train.py -f exps/example/custom/yolox_doc_face.py -d 1 -b 4 --fp16 -o -c yolox_s.pth
```

### Error de dependencias faltantes

```bash
# Reinstalar dependencias principales
pip install --upgrade setuptools wheel
pip install cython numpy
pip install -v -e .
```

### Error "module 'yolox' not found"

```bash
# Verificar que estás en el entorno correcto
conda activate yolox

# Reinstalar YOLOX
pip uninstall yolox -y
pip install -v -e .
```

## 📤 Salida del modelo

El modelo exportado se guarda como:

-   **Archivo:** `yolox_doc_face_fixed.onnx`
-   **Ubicación:** Raíz del proyecto YOLOX
-   **Uso:** Debe copiarse a `cam-detector/public/` para usar en la aplicación web

## 🔍 Verificación de resultados

-   **Carpeta de salida:** `demo_output/` - Contiene imágenes de prueba procesadas
-   **Logs de entrenamiento:** `YOLOX_outputs/` - Contiene checkpoints y métricas del entrenamiento

## 🎯 Clases detectadas

El modelo está entrenado para detectar:

1. **`face`** - Caras humanas
2. **`doc_quad`** - Documentos (cuadriláteros)

## 🔗 Integración con CAMDetector

Una vez exportado el modelo ONNX:

1. Copia `yolox_doc_face_fixed.onnx` a `cam-detector/public/`
2. Renómbralo a `yolox_doc_face_decoded.onnx` (según configuración de CAMDetector)
3. Ejecuta la aplicación web CAMDetector para pruebas en tiempo real

---

**Siguiente paso:** Consulta [CAMDetector.md](./CAMDetector.md) para usar el modelo exportado en la aplicación web.
