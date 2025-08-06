#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
#from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
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
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    
    if origin_img is None:
        print(f"Error: No se pudo cargar la imagen en {args.image_path}")
        exit(1)
    
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)
    print(f"Modelo cargado: {args.model}")
    print(f"Inputs del modelo: {[input.name for input in session.get_inputs()]}")
    print(f"Outputs del modelo: {[output.name for output in session.get_outputs()]}")

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    
    print(f"Shape de output: {output[0].shape}")
    print(f"Rango de valores en output: min={np.min(output[0])}, max={np.max(output[0])}")
    
    # Verificar si hay valores infinitos o NaN
    if np.any(np.isinf(output[0])) or np.any(np.isnan(output[0])):
        print("Warning: Se detectaron valores infinitos o NaN en la salida del modelo")
        output[0] = np.nan_to_num(output[0], nan=0.0, posinf=1e6, neginf=-1e6)
    
    predictions = demo_postprocess(output[0], input_shape)[0]
    
    # Verificar si hay predicciones válidas
    if predictions.shape[0] == 0:
        print("No se encontraron detecciones")
        mkdir(args.output_dir)
        output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
        cv2.imwrite(output_path, origin_img)
        exit(0)

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    # Filtrar predicciones con valores válidos
    valid_mask = np.all(np.isfinite(boxes), axis=1) & np.all(np.isfinite(scores), axis=1)
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    
    if len(boxes) == 0:
        print("No se encontraron detecciones válidas después del filtrado")
        mkdir(args.output_dir)
        output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
        cv2.imwrite(output_path, origin_img)
        exit(0)

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    
    # Verificar que las coordenadas estén dentro de rangos razonables
    h, w = origin_img.shape[:2]
    boxes_xyxy = np.clip(boxes_xyxy, 0, max(h, w))
    
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        
        # Verificar que las coordenadas finales sean válidas
        valid_final_mask = np.all(np.isfinite(final_boxes), axis=1) & np.isfinite(final_scores) & np.isfinite(final_cls_inds)
        final_boxes = final_boxes[valid_final_mask]
        final_scores = final_scores[valid_final_mask]
        final_cls_inds = final_cls_inds[valid_final_mask]
        
        if len(final_boxes) > 0:
            print(f"Se encontraron {len(final_boxes)} detecciones válidas")
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                           conf=args.score_thr, class_names=CLASSES_PERSONALIZADAS)
        else:
            print("No hay detecciones válidas después del filtrado final")
    else:
        print("No se encontraron detecciones después del NMS")

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
    print(f"Resultado guardado en: {output_path}")
