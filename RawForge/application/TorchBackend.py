import torch
import numpy as np

class TorchBackend:
    def __init__(self, model, model_params, device):
        self.model = model
        self.model_params = model_params
        self.device = device

        dtype_map = {
            "cpu": torch.float16,
            "cuda": torch.float16,
            "mps": torch.float16,
        }
        self.autocast_dtype = dtype_map.get(device.type, torch.float32)

    def prepare_conditioning(self, cond):
        return torch.from_numpy(cond).to(self.device)

    def infer(self, batch, cond):
        batch = torch.from_numpy(batch).to(self.device)

        B = batch.shape[0]
        cond = cond.expand(B, -1)

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
            ):
                if self.model_params["conditioning"]:
                    output = self.model(batch, cond)
                else:
                    output = self.model(batch)

        return output.cpu().numpy().clip(0, 1)