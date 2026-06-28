import numpy as np

class ONNXBackend:
    def __init__(self, model):
        self.model = model

        self.model_inputs = {
            inp.name for inp in self.model.get_inputs()
        }

    def prepare_conditioning(self, cond):
        return cond

    def infer(self, batch, cond):
        B = batch.shape[0]

        cond = np.broadcast_to(
            cond,
            (B, cond.shape[-1])
        ).astype(np.float16)

        payload = {
            "input": batch,
            "cond": cond,
        }

        payload = {
            k: v
            for k, v in payload.items()
            if k in self.model_inputs
        }

        return self.model.run(["output"], payload)[0]