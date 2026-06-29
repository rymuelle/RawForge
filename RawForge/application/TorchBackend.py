import torch
import numpy as np
from contextlib import nullcontext

class TorchBackend:
    def __init__(self, model, model_params, device):
        self.model = model
        self.model_params = model_params
        self.device = device
        self.autocast_dtype = torch.float16

    def prepare_conditioning(self, cond):
        return torch.from_numpy(cond).to(self.device)

    def infer(self, batch, cond):
        batch = torch.from_numpy(batch).to(self.device)

        B = batch.shape[0]
        cond = cond.expand(B, -1)

        kwargs = {"cond": cond} if self.model_params["conditioning"] else {}
        is_cpu = self.device == "cpu" or (
            hasattr(self.device, "type") and self.device.type == "cpu"
        )
        autocast_ctx = (
            nullcontext()
            if is_cpu
            else torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        )
        with torch.no_grad(), autocast_ctx:
            output = self.model(batch, **kwargs)

        return output.cpu().numpy().clip(0, 1)