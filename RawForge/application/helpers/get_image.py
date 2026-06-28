import numpy as np
import rawpy
from RawHandler.RawHandler import RawHandler
from RawHandler.RawHandlerRawpy import RawHandlerRawpy


def load_rh(path, demosaicing):
    print(path)
    """Loads the raw file handler"""
    if demosaicing == "rawpy":
        rh = RawHandlerRawpy(path)
    if demosaicing == "sixchan":
        rh = RawHandlerRawpy(path)
    elif demosaicing == "Malvar2004":
        rh = RawHandler(path)
    iso = rh.full_metadata.get_ISO()
    return rh, iso


def get_image(path, model_params, dims=None):
    demosaicing = model_params["demosaicing"]
    rh, iso = load_rh(path, demosaicing)
    image_scale = model_params['image_scale'] if 'image_scale' in model_params else 1
    if demosaicing == "Malvar2004":
        from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

        image_RGB = rh.as_rgb(
            dims=dims,
            demosaicing_func=demosaicing_CFA_Bayer_Malvar2004,
            colorspace="lin_rec2020",
            clip=True,
        )
    elif demosaicing == "rawpy":
        image_RGB = (
            rh.rawpy_object.postprocess(
                user_wb=[1, 1, 1, 1],
                output_color=rawpy.ColorSpace.raw,
                demosaic_algorithm=rawpy.DemosaicAlgorithm(3),
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                gamma=(1, 1),
                user_flip=0,
                output_bps=16,
                no_auto_scale=True,
            )
            / rh.rawpy_object.white_level
        )
        if dims is not None:
            image_RGB = image_RGB[dims[0] : dims[1], dims[2] : dims[3]]
        image_RGB = image_RGB.transpose(2, 0, 1)
    elif demosaicing == "sixchan":
        if 'subtract_bl' in model_params:
            subtract_bl = model_params['subtract_bl']
        else:
            subtract_bl = False
        sparse, mask = rh.compute_mask_and_sparse(dims=dims, subtract_bl=subtract_bl)
        sparse = sparse * image_scale
        if 'image_clip' in model_params:
            sparse = sparse.clip(0, model_params['image_clip'])
        image_RGB = np.concatenate([sparse, mask], axis=0)
    

    image_RGB = np.expand_dims(image_RGB, axis=0).astype(np.float16)
    return rh, image_RGB, iso
