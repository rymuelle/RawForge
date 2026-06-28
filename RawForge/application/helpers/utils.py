import onnxruntime as ort
import os


def get_best_providers(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    available = ort.get_available_providers()
    return available
    priority = [
        "CUDAExecutionProvider",  # NVIDIA
        "ROCMExecutionProvider",  # AMD (Direct)
        "MIGraphXExecutionProvider",  # AMD (Optimized)
        "CoreMLExecutionProvider",  # Apple Silicon
        "DmlExecutionProvider",  # Windows (AMD/Intel/Generic GPU)
        "CPUExecutionProvider",  # The fallback
    ]

    return [p for p in priority if p in available]
