import torch
import numpy as np
from blended_tiling_numpy import TilingModule
from tqdm import tqdm


class InferenceWorkerTorch:
    def __init__(
        self,
        model,
        model_params,
        conditioning,
        device="cpu",
        tile_size=512,
        tile_overlap=0.25,
        batch_size=2,
        disable_tqdm=False,
    ):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.conditioning = conditioning
        self.tile_size = tile_size
        self.device = device
        if "tile_size" in model_params:
            self.tile_size = model_params["tile_size"]
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        if "batch_size" in model_params:
            self.batch_size = model_params["batch_size"]
        self._is_cancelled = False
        self.disable_tqdm = disable_tqdm

    def cancel(self):
        self._is_cancelled = True

    def _tile_process(self, image_RGB, model_params, progress_callback=None):
        # Prepare Data
        full_size = [image_RGB.shape[2], image_RGB.shape[3]]
        tile_size = [self.tile_size, self.tile_size]
        overlap = [self.tile_overlap, self.tile_overlap]
        # Tiling Setup
        tiling_module_rgb = TilingModule(
            tile_size=[s for s in tile_size],
            tile_overlap=overlap,
            base_size=[s for s in full_size],
        )

        tiles_rgb = tiling_module_rgb.split_into_tiles(image_RGB)

        batches_rgb = [
            tiles_rgb[i : i + self.batch_size]
            for i in range(0, len(tiles_rgb), self.batch_size)
        ]
        # Conditioning Setup
        cond_tensor = np.array([self.conditioning]).astype(np.float32)
        cond_tensor[:, 0] /= 6400.0
        cond_tensor[:, 1] = 0.0
        cond_tensor = cond_tensor[:, 0:1]
        cond_tensor = cond_tensor.astype(np.float16)
        if "cond_scale" in model_params:
            cond_tensor *= cond_tensor * model_params["cond_scale"]

        processed_batches = []

        cond_tensor = torch.from_numpy(cond_tensor).to(self.device)
        # Determine Dtype
        dtype_map = {"mps": torch.float16, "cuda": torch.float16, "cpu": torch.float16}
        autocast_dtype = dtype_map.get(self.device.type, torch.float32)
        total_batches = len(batches_rgb)
        # Inference Loop
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                for i, (batch_rgb) in tqdm(
                    enumerate(batches_rgb), disable=self.disable_tqdm
                ):
                    if self._is_cancelled:
                        return None, None
                    batch_rgb = torch.from_numpy(batch_rgb).to(self.device)
                    B = batch_rgb.shape[0]
                    # Expand conditioning to match batch size
                    curr_cond = cond_tensor.expand(B, -1)
                    if model_params['conditioning']:
                        output = self.model(batch_rgb, curr_cond)
                    else:
                        output = self.model(batch_rgb)
                    processed_batches.append(output.cpu().numpy())

                    if progress_callback:
                        progress_callback((i + 1) / total_batches)
        # Rebuild
        tiles_out = np.concat(processed_batches, axis=0)
        stitched = tiling_module_rgb.rebuild_with_masks(tiles_out)

        return image_RGB, stitched

    def run(self, model_params, image_RGB):
        img, denoised_img = self._tile_process(image_RGB, model_params)
        return img, denoised_img
