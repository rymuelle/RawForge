import numpy as np
from RawForge.application.dng_utils import convert_color_matrix, to_dng
import tifffile


class ImageSaver:
    def __init__(self, model_params, rh, dims=None):
        self.rh = rh
        self.model_params = model_params
        self.dims = dims

    def to_tiff(self, image, filename, apply_ccm=True):
        image = image
        if apply_ccm:
            transform_matrix = self.rh.rgb_colorspace_transform(
                colorspace="lin_rec2020"
            )
            image = image[0].transpose(1, 2, 0)
            transformed = image @ transform_matrix.T
        else:
            transformed = image[0].transpose(1, 2, 0)

        transformed = np.clip(transformed, 0, 1)
        transformed = transformed ** (1 / 2.2)
        transformed = transformed * (2**8 - 1)

        uint_img = transformed.astype(np.uint8)

        tifffile.imwrite(
            filename,
            uint_img,
            photometric="rgb",  # Explicitly define the color space
            compression="deflate",  # Optional: Lossless compression supported by darktable
        )

    def to_raw(self, denoised, filename, save_cfa):
        # Compute CFA
        if self.model_params["demosaicing"] == "rawpy" or self.model_params["demosaicing"] == "sixchan":
            if self.dims is None:
                self.dims = [0, 9999999, 0, 9999999]
            _, mask = self.rh.compute_mask_and_sparse(dims=self.dims)
            denoised = denoised[0]
            denoised = denoised.clip(0, 1)

            denoised = np.where(mask, denoised, 0)
            denoised = denoised.sum(axis=0)
            denoised = (
                denoised * (self.rh.core_metadata.white_level)
                + self.rh.core_metadata.black_level_per_channel[0]
            )
            self.rh.to_dng(filename, uint_img=denoised)
        else:
            transform_matrix = np.linalg.inv(
                self.rh.rgb_colorspace_transform(colorspace=self.rh.colorspace)
            )

            CCM = self.rh.rgb_colorspace_transform(colorspace="XYZ")
            CCM = np.linalg.inv(CCM)
            denoised = denoised[0].transpose(1, 2, 0)
            transformed = denoised @ transform_matrix.T
            uint_img = np.clip(transformed * 2**16 - 1, 0, 2**16 - 1).astype(np.uint16)
            ccm1 = convert_color_matrix(CCM)
            to_dng(
                uint_img,
                self.rh,
                filename,
                ccm1,
                save_cfa=save_cfa,
                convert_to_cfa=True,
            )
