import numpy as np
from blended_tiling_numpy import TilingModule
from tqdm import tqdm


class InferenceWorker:
    def __init__(
        self,
        backend,
        model_params,
        conditioning,
        disable_tqdm=False,
    ):
        self.backend = backend
        self.model_params = model_params
        self.conditioning = conditioning

        self.tile_size = model_params.get("tile_size", 256)
        self.tile_overlap = model_params.get("tile_overlap", 0.25)
        self.batch_size = model_params.get("batch_size", 2)

        self.disable_tqdm = disable_tqdm
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _build_conditioning(self):
        cond = np.array([self.conditioning], dtype=np.float32)

        if "max_iso" in self.model_params:
            cond[:, 0] = min(cond[:, 0], self.model_params["max_iso"])
        cond[:, 0] /= 6400.0
        cond[:, 1] = 0.0
        cond = cond[:, :1].astype(np.float16)

        if "cond_scale" in self.model_params:
            cond *= cond * self.model_params["cond_scale"]

        return self.backend.prepare_conditioning(cond)

    def run(self, image_RGB, progress_callback=None):

        full_size = image_RGB.shape[2:]

        tiler = TilingModule(
            tile_size=[self.tile_size] * 2,
            tile_overlap=[self.tile_overlap] * 2,
            base_size=list(full_size),
        )

        tiles = tiler.split_into_tiles(image_RGB)

        batches = [
            tiles[i:i+self.batch_size]
            for i in range(0, len(tiles), self.batch_size)
        ]

        cond = self._build_conditioning()

        outputs = []

        for i, batch in tqdm(
            enumerate(batches),
            total=len(batches),
            disable=self.disable_tqdm,
        ):

            if self._is_cancelled:
                return None, None

            outputs.append(
                self.backend.infer(batch, cond)
            )

            if progress_callback:
                progress_callback((i + 1) / len(batches))

        stitched = tiler.rebuild_with_masks(
            np.concatenate(outputs, axis=0)
        )

        return image_RGB, stitched
    
