import torch
from blended_tiling import TilingModule
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from RawForge.application.postprocessing import match_colors_linear
from tqdm import tqdm

class InferenceWorker():
    def __init__(self, model, model_params, device, rh, conditioning, dims, tile_size=256, tile_overlap=0.25, batch_size=2, disable_tqdm=False):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.device = device
        self.rh = rh
        self.conditioning = conditioning
        self.dims = dims
        # Quick and dirty hack to force to be even
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._is_cancelled = False
        self.disable_tqdm = disable_tqdm

    def cancel(self):
        self._is_cancelled = True

    def _tile_process(self):
        # Prepare Data
        image_RGGB = self.rh.as_rggb(dims=self.dims, colorspace='lin_rec2020')
        image_RGB = self.rh.as_rgb(dims=self.dims, demosaicing_func=demosaicing_CFA_Bayer_Malvar2004, colorspace='lin_rec2020', clip=True)
        
        tensor_image = torch.from_numpy(image_RGGB).unsqueeze(0).contiguous()
        tensor_RGB = torch.from_numpy(image_RGB).unsqueeze(0).contiguous()

        full_size = [image_RGGB.shape[1], image_RGGB.shape[2]]
        tile_size = [self.tile_size // 2, self.tile_size // 2]
        overlap = [self.tile_overlap, self.tile_overlap]

        # Tiling Setup
        tiling_module = TilingModule(tile_size=tile_size, tile_overlap=overlap, base_size=full_size)
        tiling_module_rgb = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])
        tiling_module_rebuild = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])

        tiles = tiling_module.split_into_tiles(tensor_image).float().to(self.device)
        tiles_rgb = tiling_module_rgb.split_into_tiles(tensor_RGB).float().to(self.device)
        
        batches = torch.split(tiles, self.batch_size)
        batches_rgb = torch.split(tiles_rgb, self.batch_size)

        # Conditioning Setup
        cond_tensor = torch.as_tensor(self.conditioning, device=self.device).float().unsqueeze(0)
        cond_tensor[:, 0] /= 6400
        cond_tensor[:, 1] = 0
        cond_tensor = cond_tensor[:, 0:1]

        processed_batches = []
        
        # Determine Dtype
        dtype_map = {'mps': torch.float16, 'cuda': torch.float16, 'cpu': torch.bfloat16}
        autocast_dtype = dtype_map.get(self.device.type, torch.float32)

        total_batches = len(batches_rgb)
        
        # Inference Loop
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                for i, (batch, batch_rgb) in tqdm(enumerate(zip(batches, batches_rgb)), disable=self.disable_tqdm):
                    if self._is_cancelled: return None, None
                    
                    B = batch.shape[0]
                    # Expand conditioning to match batch size
                    curr_cond = cond_tensor.expand(B, -1)
                    
                    output = self.model(batch_rgb, curr_cond)

                    # Output processing
                    if "affine" in self.model_params:
                        output, _, _ = match_colors_linear(output, batch_rgb)
                    processed_batches.append(output.cpu())
                    
        # Rebuild
        tiles_out = torch.cat(processed_batches, dim=0)
        stitched = tiling_module_rebuild.rebuild_with_masks(tiles_out).detach().cpu().numpy()[0]

        torch.cuda.empty_cache()

        return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)
    
    def run(self):
        try:
            img, denoised_img = self._tile_process()

            # Post-process blending
            blend_alpha = self.conditioning[1] / 100
            final_denoised = (denoised_img * (1 - blend_alpha)) + (img * blend_alpha)
            
            return img, final_denoised
            
        except Exception as e:
            print(str(e))