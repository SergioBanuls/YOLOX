#!/usr/bin/env python3
import torch
import cv2
import numpy as np
import onnxruntime
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import preproc

def test_pytorch_model(image_path, checkpoint_path):
    """Prueba el modelo PyTorch original"""
    print("=== TESTING PYTORCH MODEL ===")
    
    # Cargar el modelo
    exp = get_exp("exps/example/custom/yolox_doc_face.py", None)
    exp.num_classes = 2  # IMPORTANTE!
    model = exp.get_model()
    
    # Cargar checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Cargar y procesar imagen
    img = cv2.imread(image_path)
    img_processed, ratio = preproc(img, (640, 640))
    img_tensor = torch.from_numpy(img_processed).unsqueeze(0).float()
    
    # Inferencia
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(outputs, exp.num_classes, 0.3, 0.45)[0]
    
    if outputs is not None:
        print(f"Detecciones PyTorch: {len(outputs)}")
        for i, det in enumerate(outputs):
            x1, y1, x2, y2, obj_conf, cls_conf, cls = det
            cls_name = ['face', 'doc_quad'][int(cls)]
            score = obj_conf * cls_conf
            print(f"  {cls_name}: score={score:.3f}, obj_conf={obj_conf:.3f}, cls_conf={cls_conf:.3f}")
    else:
        print("No se encontraron detecciones")
    
    return outputs

def test_onnx_model(image_path, onnx_path):
    """Prueba el modelo ONNX"""
    print("\n=== TESTING ONNX MODEL ===")
    
    # Cargar modelo
    session = onnxruntime.InferenceSession(onnx_path)
    
    # Cargar y procesar imagen
    img = cv2.imread(image_path)
    img_processed, ratio = preproc(img, (640, 640))
    
    # Inferencia
    ort_inputs = {session.get_inputs()[0].name: img_processed[None, :, :, :]}
    output = session.run(None, ort_inputs)
    
    output_raw = output[0][0]
    print(f"Output shape: {output_raw.shape}")
    
    # Analizar salidas
    objectness = output_raw[:, 4]
    class_probs = output_raw[:, 5:]
    
    face_scores = objectness * class_probs[:, 0]
    doc_scores = objectness * class_probs[:, 1]
    
    # Top detecciones para cada clase
    print(f"\nTop 3 face detections:")
    top_face = np.argsort(face_scores)[-3:][::-1]
    for idx in top_face:
        print(f"  score={face_scores[idx]:.3f}, obj={objectness[idx]:.3f}, prob={class_probs[idx,0]:.3f}")
    
    print(f"\nTop 3 doc_quad detections:")
    top_doc = np.argsort(doc_scores)[-3:][::-1]
    for idx in top_doc:
        print(f"  score={doc_scores[idx]:.3f}, obj={objectness[idx]:.3f}, prob={class_probs[idx,1]:.3f}")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="prueba_doc.jpg")
    parser.add_argument("--checkpoint", default="YOLOX_outputs/yolox_doc_face_20250805/best_ckpt.pth")
    parser.add_argument("--onnx", default="yolox_doc_face_decoded.onnx")  # Cambiar a tu modelo ONNX real
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.image):
        print(f"Error: La imagen {args.image} no existe")
        exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: El checkpoint {args.checkpoint} no existe")
        exit(1)
    
    if not os.path.exists(args.onnx):
        # Buscar el modelo ONNX en varias ubicaciones
        possible_onnx = [
            "yolox_doc_face_decoded.onnx",
            "../yolox_doc_face_decoded.onnx",
            "../../yolox_doc_face_decoded.onnx",
            "demo/ONNXRuntime/yolox_doc_face_decoded.onnx"
        ]
        for path in possible_onnx:
            if os.path.exists(path):
                args.onnx = path
                print(f"Modelo ONNX encontrado en: {path}")
                break
        
        if not os.path.exists(args.onnx):
            print(f"Error: El modelo ONNX {args.onnx} no existe")
            exit(1)
    
    # Comparar ambos modelos
    pytorch_results = test_pytorch_model(args.image, args.checkpoint)
    onnx_results = test_onnx_model(args.image, args.onnx)
