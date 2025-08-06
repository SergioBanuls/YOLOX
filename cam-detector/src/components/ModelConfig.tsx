import React from 'react'
import type { ExecutionProvider } from '../utils/detector'

interface ModelConfigProps {
    selectedProvider: ExecutionProvider
    onProviderChange: (provider: ExecutionProvider) => void
    isModelLoaded: boolean
    onReloadModel: () => void
}

const providerInfo = {
    cpu: {
        name: '🖥️ CPU',
        description: 'Procesamiento en CPU - Más lento pero más compatible',
    },
    webgl: {
        name: '🎮 WebGL',
        description:
            'Aceleración GPU usando WebGL - Balance entre velocidad y compatibilidad',
    },
    wasm: {
        name: '⚡ WebAssembly',
        description: 'WebAssembly optimizado - Rápido en CPU',
    },
    webgpu: {
        name: '🚀 WebGPU',
        description: 'Máximo rendimiento GPU - Requiere navegador moderno',
    },
}

export const ModelConfig: React.FC<ModelConfigProps> = ({
    selectedProvider,
    onProviderChange,
    isModelLoaded,
    onReloadModel,
}) => {
    const [isChanging, setIsChanging] = React.useState(false)

    const handleProviderChange = async (provider: ExecutionProvider) => {
        if (provider === selectedProvider || isChanging) return

        setIsChanging(true)
        try {
            await onProviderChange(provider)
        } finally {
            setIsChanging(false)
        }
    }

    return (
        <div className='model-config'>
            <h4>⚙️ Configuración del Modelo</h4>

            <div className='provider-selection'>
                <label className='config-label'>Backend de ejecución:</label>

                <div className='provider-options'>
                    {(Object.keys(providerInfo) as ExecutionProvider[]).map(
                        (provider) => (
                            <div key={provider} className='provider-option'>
                                <label
                                    className={`provider-label ${
                                        isChanging ? 'disabled' : ''
                                    }`}
                                >
                                    <input
                                        type='radio'
                                        name='executionProvider'
                                        value={provider}
                                        checked={selectedProvider === provider}
                                        onChange={() =>
                                            handleProviderChange(provider)
                                        }
                                        disabled={isChanging}
                                    />
                                    <div className='provider-info'>
                                        <div className='provider-name'>
                                            {providerInfo[provider].name}
                                            {selectedProvider === provider &&
                                                isModelLoaded &&
                                                ' ✅'}
                                            {selectedProvider === provider &&
                                                !isModelLoaded &&
                                                ' ⏳'}
                                        </div>
                                        <div className='provider-desc'>
                                            {providerInfo[provider].description}
                                        </div>
                                    </div>
                                </label>
                            </div>
                        )
                    )}
                </div>
            </div>

            <div className='model-actions'>
                <button
                    onClick={onReloadModel}
                    disabled={!isModelLoaded || isChanging}
                    className='reload-button'
                >
                    🔄 Recargar Modelo
                </button>

                <div className='config-note'>
                    {isChanging
                        ? '⏳ Cambiando backend...'
                        : '💡 Puedes cambiar el backend en cualquier momento'}
                </div>
            </div>
        </div>
    )
}
