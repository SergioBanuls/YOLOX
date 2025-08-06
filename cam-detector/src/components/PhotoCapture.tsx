import React from 'react'

interface PhotoCaptureProps {
    capturedImage: string | null
    confidence: number
    onRetake: () => void
    onSave: () => void
    onClose: () => void
}

const getConfidenceLevel = (confidence: number): string => {
    if (confidence >= 0.9) return 'excellent'
    if (confidence >= 0.8) return 'high'
    if (confidence >= 0.7) return 'good'
    return 'medium'
}

export const PhotoCapture: React.FC<PhotoCaptureProps> = ({
    capturedImage,
    confidence,
    onRetake,
    onSave,
    onClose,
}) => {
    if (!capturedImage) return null

    const handleSave = () => {
        // Crear un enlace temporal para descargar la imagen
        const link = document.createElement('a')
        link.href = capturedImage
        link.download = `face-capture-${new Date()
            .toISOString()
            .slice(0, 19)
            .replace(/:/g, '-')}.jpg`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        onSave()
    }

    return (
        <div className='photo-capture-overlay'>
            <div className='photo-capture-modal'>
                <div className='photo-capture-header'>
                    <div className='capture-title'>
                        <h3>📸 Foto Capturada</h3>
                        <div
                            className={`confidence-badge ${getConfidenceLevel(
                                confidence
                            )}`}
                        >
                            {(confidence * 100).toFixed(1)}% confianza máx.
                        </div>
                    </div>
                    <button
                        className='close-button'
                        onClick={onClose}
                        aria-label='Cerrar'
                    >
                        ✕
                    </button>
                </div>

                <div className='photo-capture-content'>
                    <img
                        src={capturedImage}
                        alt='Captura de rostro detectado'
                        className='captured-photo'
                    />
                </div>

                <div className='photo-capture-actions'>
                    <button className='retake-button' onClick={onRetake}>
                        🔄 Repetir Foto
                    </button>
                    <button className='save-button' onClick={handleSave}>
                        💾 Guardar Imagen
                    </button>
                </div>
            </div>
        </div>
    )
}
