#!/usr/bin/env python3
import torch
import torch.nn as nn
from yolox.exp import get_exp
import numpy as np
import os

def export_correct_onnx():
    # Paths
    checkpoint_path = "YOLOX_outputs/yolox_doc_face_20250805/best_ckpt.pth"
    exp_file = "exps/example/custom/yolox_doc_face.py"
    output_name = "yolox_doc_face_fixed.onnx"
    
    print("=== EXPORTANDO MODELO ONNX CORREGIDO ===")
    
    # Cargar experimento con configuración personalizada
    exp = get_exp(exp_file, None)
    
    # IMPORTANTE: Configurar ANTES de crear el modelo
    exp.num_classes = 2
    exp.test_conf = 0.25  # Bajamos el threshold
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)
    
    print(f"Configuración del modelo:")
    print(f"  num_classes: {exp.num_classes}")
    print(f"  test_conf: {exp.test_conf}")
    print(f"  nmsthre: {exp.nmsthre}")
    print(f"  test_size: {exp.test_size}")
    
    # Crear modelo
    model = exp.get_model()
    
    # Verificar que el modelo tiene la arquitectura correcta
    print(f"\nVerificando arquitectura del modelo:")
    print(f"  Model head num_classes: {model.head.num_classes}")
    
    # Cargar pesos
    print(f"\nCargando checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No se encuentra el checkpoint en {checkpoint_path}")
        return None
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Verificar que el checkpoint es para 2 clases
    if "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt
    
    # Cargar los pesos
    model.load_state_dict(model_state)
    model.eval()
    
    # IMPORTANTE: Configurar decode_in_inference
    model.head.decode_in_inference = True
    
    print(f"\nConfiguración final del head:")
    print(f"  num_classes: {model.head.num_classes}")
    print(f"  decode_in_inference: {model.head.decode_in_inference}")
    
    # Crear dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Test de inferencia con PyTorch antes de exportar
    print("\nProbando inferencia PyTorch con decode_in_inference=True...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
        print(f"  Output shape: {pytorch_output.shape}")
        print(f"  Output range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
        
        # Verificar estructura del output
        if len(pytorch_output.shape) == 3:
            obj_scores = pytorch_output[0, :, 4]
            cls_probs = pytorch_output[0, :, 5:]
            print(f"  Objectness range: [{obj_scores.min():.3f}, {obj_scores.max():.3f}]")
            print(f"  Class probs shape: {cls_probs.shape}")
            print(f"  Face prob range: [{cls_probs[:, 0].min():.3f}, {cls_probs[:, 0].max():.3f}]")
            print(f"  Doc prob range: [{cls_probs[:, 1].min():.3f}, {cls_probs[:, 1].max():.3f}]")
    
    # Exportar a ONNX
    print(f"\nExportando a ONNX: {output_name}")
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✓ Modelo exportado exitosamente a: {output_name}")
    
    # Verificar el modelo ONNX exportado
    print("\nVerificando modelo ONNX...")
    import onnxruntime
    session = onnxruntime.InferenceSession(output_name)
    
    # Test de inferencia con ONNX
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_inputs)[0]
    print(f"  ONNX output shape: {ort_output.shape}")
    print(f"  ONNX output range: [{ort_output.min():.3f}, {ort_output.max():.3f}]")
    
    # Analizar estructura del output ONNX
    if len(ort_output.shape) == 3:
        obj_scores_onnx = ort_output[0, :, 4]
        cls_probs_onnx = ort_output[0, :, 5:]
        print(f"  ONNX Objectness range: [{obj_scores_onnx.min():.3f}, {obj_scores_onnx.max():.3f}]")
        print(f"  ONNX Face prob range: [{cls_probs_onnx[:, 0].min():.3f}, {cls_probs_onnx[:, 0].max():.3f}]")
        print(f"  ONNX Doc prob range: [{cls_probs_onnx[:, 1].min():.3f}, {cls_probs_onnx[:, 1].max():.3f}]")
    
    # Comparar outputs
    pytorch_flat = pytorch_output.numpy().flatten()
    onnx_flat = ort_output.flatten()
    diff = np.abs(pytorch_flat - onnx_flat).mean()
    print(f"\n  Diferencia media PyTorch vs ONNX: {diff:.6f}")
    
    if diff < 0.001:
        print("✓ Los outputs son consistentes entre PyTorch y ONNX")
    else:
        print("⚠ Hay diferencias significativas entre PyTorch y ONNX")
    
    return output_name

if __name__ == "__main__":
    onnx_path = export_correct_onnx()
    
    if onnx_path:
        print("\n" + "="*50)
        print("Para probar el nuevo modelo ONNX, ejecuta:")
        print(f"python compare_models.py --onnx {onnx_path}")
        print("\nO para inferencia:")
        print(f"python demo/ONNXRuntime/onnx_inference_decoded.py -m {onnx_path} -i prueba_doc.jpg")