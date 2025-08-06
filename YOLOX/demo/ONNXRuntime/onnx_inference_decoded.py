#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, vis

CLASSES_PERSONALIZADAS = ['face', 'doc_quad']


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--decoded",
        action="store_true",
        help="Whether the model includes decode_in_inference (post-processing)"
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    
    if origin_img is None:
        print(f"Error: No se pudo cargar la imagen en {args.image_path}")
        exit(1)
    
    h, w = origin_img.shape[:2]
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)
    print(f"Modelo cargado: {args.model}")
    print(f"Inputs del modelo: {[input.name for input in session.get_inputs()]}")
    print(f"Outputs del modelo: {[output.name for output in session.get_outputs()]}")
    
    # Verificar dimensiones esperadas
    for output in session.get_outputs():
        print(f"Output '{output.name}': shape={output.shape}, type={output.type}")
    
    # Verificar que el modelo fue entrenado con 2 clases
    print(f"\n=== VERIFICACIÓN DE CONFIGURACIÓN DEL MODELO ===")
    expected_output_dim = 4 + 1 + 2  # bbox(4) + objectness(1) + num_classes(2)
    print(f"Dimensión esperada de salida por detección: {expected_output_dim}")
    
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    
    print(f"Shape de output: {output[0].shape}")
    print(f"Rango de valores en output: min={np.min(output[0])}, max={np.max(output[0])}")
    
    # Verificar si hay valores infinitos o NaN
    if np.any(np.isinf(output[0])) or np.any(np.isnan(output[0])):
        print("Warning: Se detectaron valores infinitos o NaN en la salida del modelo")
        output[0] = np.nan_to_num(output[0], nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Usar post-procesamiento manual sin demo_postprocess para evitar overflow
    print("Usando post-procesamiento manual directo")
    from yolox.utils import multiclass_nms
    
    # Post-procesamiento manual básico
    output_raw = output[0][0]  # Shape: (8400, 7)
    
    print(f"Output raw shape: {output_raw.shape}")
    print(f"Primeras 3 salidas raw del modelo:")
    for i in range(3):
        print(f"  Raw {i}: {output_raw[i]}")
    
    # Con decode_in_inference=True, las salidas ya están procesadas (sigmoid aplicado)
    # Las columnas son: [cx, cy, w, h, objectness, class_0_prob, class_1_prob, ...]
    print(f"Analizando estructura de salidas (decode_in_inference=True):")
    print(f"  Columnas 0-3: coordenadas (cx, cy, w, h)")
    print(f"  Columna 4: objectness (ya con sigmoid)")
    print(f"  Columnas 5+: probabilidades de clase (ya con sigmoid)")
    
    # NO aplicar sigmoid porque ya está aplicado por el modelo
    objectness = output_raw[:, 4]  # Ya procesado
    class_probs = output_raw[:, 5:]  # Ya procesado
    
    print(f"Estadísticas de objectness: min={np.min(objectness):.4f}, max={np.max(objectness):.4f}, mean={np.mean(objectness):.4f}")
    print(f"Estadísticas de class_probs shape: {class_probs.shape}")
    print(f"Class_probs primeras 3 filas:")
    for i in range(3):
        print(f"  Fila {i}: face={class_probs[i,0]:.4f}, doc_quad={class_probs[i,1]:.4f}")
    
    # Agregar análisis más detallado de las probabilidades
    print(f"\n=== ANÁLISIS DETALLADO DE PROBABILIDADES ===")
    print(f"Estadísticas de face_prob: min={np.min(class_probs[:,0]):.4f}, max={np.max(class_probs[:,0]):.4f}, mean={np.mean(class_probs[:,0]):.4f}")
    print(f"Estadísticas de doc_quad_prob: min={np.min(class_probs[:,1]):.4f}, max={np.max(class_probs[:,1]):.4f}, mean={np.mean(class_probs[:,1]):.4f}")
    
    # Encontrar las mejores detecciones específicamente para doc_quad
    doc_quad_scores = objectness * class_probs[:, 1]
    top_doc_indices = np.argsort(doc_quad_scores)[-5:][::-1]
    print(f"\nTop 5 detecciones específicas de doc_quad:")
    for i, idx in enumerate(top_doc_indices):
        print(f"  {i+1}. idx={idx}: doc_score={doc_quad_scores[idx]:.4f}, obj={objectness[idx]:.4f}, doc_prob={class_probs[idx,1]:.4f}, coords=(cx={output_raw[idx,0]:.1f}, cy={output_raw[idx,1]:.1f}, w={output_raw[idx,2]:.1f}, h={output_raw[idx,3]:.1f})")
    
    # Calcular scores finales de forma más explícita
    # Para cada clase, multiplicar objectness por probabilidad de clase
    face_scores = objectness * class_probs[:, 0]  # face es clase 0
    doc_scores = objectness * class_probs[:, 1]   # doc_quad es clase 1
    
    # Usar thresholds diferentes por clase debido al desbalance
    face_threshold = args.score_thr
    doc_threshold = min(args.score_thr, 0.35)  # Threshold más bajo para doc_quad
    
    print(f"\nUsando thresholds: face={face_threshold:.2f}, doc_quad={doc_threshold:.2f}")
    
    # Filtrar detecciones por clase con diferentes thresholds
    valid_face = face_scores > face_threshold
    valid_doc = doc_scores > doc_threshold
    
    # Combinar las detecciones válidas
    face_indices = np.where(valid_face)[0]
    doc_indices = np.where(valid_doc)[0]
    
    print(f"Detecciones válidas: {len(face_indices)} faces, {len(doc_indices)} doc_quads")
    
    if len(face_indices) == 0 and len(doc_indices) == 0:
        print("No se encontraron detecciones válidas con los thresholds dados")
        predictions = np.array([])
    else:
        # Combinar todas las detecciones válidas
        all_valid_indices = np.concatenate([face_indices, doc_indices])
        all_valid_boxes = output_raw[all_valid_indices, :4]
        
        # Crear arrays de scores y clases
        all_scores = np.concatenate([
            face_scores[face_indices],
            doc_scores[doc_indices]
        ])
        all_classes = np.concatenate([
            np.zeros(len(face_indices)),  # face = 0
            np.ones(len(doc_indices))     # doc_quad = 1
        ])
        
        print(f"Total detecciones antes de filtros de tamaño: {len(all_valid_boxes)}")
        
        # Filtrar detecciones con tamaños poco razonables
        widths = all_valid_boxes[:, 2]
        heights = all_valid_boxes[:, 3]
        reasonable_mask = (widths > 1) & (heights > 1) & (widths < input_shape[0]) & (heights < input_shape[1])
        
        valid_boxes = all_valid_boxes[reasonable_mask]
        valid_scores = all_scores[reasonable_mask]
        valid_classes = all_classes[reasonable_mask]
        
        print(f"Detecciones con tamaños razonables: {len(valid_boxes)}")
        
        if len(valid_boxes) == 0:
            print("No hay detecciones con tamaños razonables")
            predictions = np.array([])
        else:
            # Desescalar las coordenadas
            print(f"Ratio usado: {ratio}")
            valid_boxes[:, [0, 2]] /= ratio  # cx, w
            valid_boxes[:, [1, 3]] /= ratio  # cy, h
            
            # Convertir de cxcywh a xyxy
            boxes_xyxy = np.zeros_like(valid_boxes)
            boxes_xyxy[:, 0] = valid_boxes[:, 0] - valid_boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = valid_boxes[:, 1] - valid_boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = valid_boxes[:, 0] + valid_boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = valid_boxes[:, 1] + valid_boxes[:, 3] / 2  # y2
            
            print(f"Boxes después de escalado:")
            for i in range(min(3, len(boxes_xyxy))):
                box = boxes_xyxy[i]
                print(f"  Box {i}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f} (w={box[2]-box[0]:.1f}, h={box[3]-box[1]:.1f})")
            
            # Clip a los límites de la imagen
            h, w = origin_img.shape[:2]
            print(f"Imagen original: w={w}, h={h}")
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)
            
            # Preparar scores para NMS multiclase
            # Necesitamos los scores por clase (objectness * class_prob)
            scores_per_class = np.zeros((len(valid_boxes), len(CLASSES_PERSONALIZADAS)))
            
            # Obtener los índices originales de las detecciones válidas
            face_valid_idx = np.arange(len(face_indices))
            doc_valid_idx = np.arange(len(face_indices), len(face_indices) + len(doc_indices))
            
            # Asignar scores por clase
            face_mask = valid_classes == 0
            doc_mask = valid_classes == 1
            
            scores_per_class[face_mask, 0] = valid_scores[face_mask]  # face scores
            scores_per_class[doc_mask, 1] = valid_scores[doc_valid_idx]  # doc_quad scores
            
            # Usar NMS class-aware para manejar múltiples clases correctamente
            dets = multiclass_nms(boxes_xyxy, scores_per_class, nms_thr=0.45, score_thr=args.score_thr, class_agnostic=False)
            if dets is not None and len(dets) > 0:
                print(f"Detecciones después de NMS: {len(dets)}")
                
                final_boxes = dets[:, :4]
                final_scores = dets[:, 4] 
                final_cls_inds = dets[:, 5].astype(int)  # Las clases vienen correctamente del NMS
                
                print(f"Detecciones finales:")
                for i in range(len(final_boxes)):
                    x1, y1, x2, y2 = final_boxes[i]
                    score = final_scores[i]
                    cls = int(final_cls_inds[i]) if i < len(final_cls_inds) else 0
                    cls_name = CLASSES_PERSONALIZADAS[cls] if cls < len(CLASSES_PERSONALIZADAS) else f"class_{cls}"
                    print(f"  {cls_name}: score={score:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) size=({x2-x1:.1f}x{y2-y1:.1f})")
                
                print(f"Se encontraron {len(final_boxes)} detecciones válidas")
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                               conf=args.score_thr, class_names=CLASSES_PERSONALIZADAS)
                predictions = dets  # Marcar que tenemos predicciones válidas
            else:
                print("No se encontraron detecciones después del NMS")
                predictions = np.array([])
    
    if 'predictions' not in locals() or (hasattr(predictions, 'shape') and predictions.shape[0] == 0):
        print("No se encontraron detecciones finales")

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
    print(f"Resultado guardado en: {output_path}")
