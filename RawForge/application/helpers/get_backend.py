def get_backend(use_onnx: bool, verbose: int):
    """Handles dynamic loading of backends."""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    try:
        import onnxruntime
        onnx_available = True
    except ImportError:
        onnx_available = False

    if not (onnx_available or torch_available):
        raise RuntimeError("Must have either ONNX or Torch backends installed.")

    if (onnx_available and not torch_available) or use_onnx:
        from RawForge.application.ONNXBackend import ONNXBackend as Backend
        from RawForge.application.ModelHandler import ModelHandler
        if verbose > 1: print("Using ONNX runtime")
        return ModelHandler, Backend, "ONNX"
    else:
        from RawForge.application.TorchBackend import TorchBackend as Backend
        from RawForge.application.ModelHandlerTorch import ModelHandlerTorch as ModelHandler
        if verbose > 1: print("Using Torch runtime")
        return ModelHandler, Backend, "Torch"