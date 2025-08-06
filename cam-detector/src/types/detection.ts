export interface Detection {
    x1: number
    y1: number
    x2: number
    y2: number
    score: number
    class_id: number
    class_name: string
}

export interface ModelStats {
    objectnessMin: number
    objectnessMax: number
    objectnessMean: number
    totalDetections: number
    validDetections: number
    faceDetections: number
    docDetections: number
    processingTimeMs: number
    fps: number
}

export interface WebcamDimensions {
    width: number
    height: number
}
