# Quiet color import warning
import warnings

warnings.filterwarnings("ignore", message=".*Matplotlib.*not available.*")
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import numpy.typing as npt

from .application.postprocessing import postprocess
from .application.MODEL_REGISTRY import MODEL_REGISTRY
from .application.ImageSaver import ImageSaver
from .application.helpers.get_image import get_image
from .application.helpers.get_backend import get_backend
from .application.helpers.exiftools import copy_with_exiftools
# import glob



def process_img(
    models: List[str],
    image_RGB:  npt.NDArray[np.float16],
    conditioning_str: Optional[str] = None,
    iso:  float = 0,
    disable_tqdm: bool = False,
    tile_size: int = 256,
    tile_overlap: float = 0.25,
    verbose: int = 1,
    use_onnx: bool = False,
    device: Optional[str] = None,
    lumi: float = 0.0,
    chroma: float = 0.0,
    clip_highlights: bool = False,
    affine: bool = False,
    progress_callback = None,
    rh = None,
):
    ModelHandler, InferenceWorker, runtime = get_backend(use_onnx, verbose)

    # Formatting conditioning
    if not conditioning_str:
        conditioning = [iso, 0]
    else:
        conditioning = [int(x) for x in conditioning_str.split(",")]

    inference_kwargs = {
        "disable_tqdm": disable_tqdm or (verbose == 0),
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
    }

    # Processing Loop
    output_img = image_RGB
    for model_name in models:
        handler = ModelHandler(verbose=verbose)
        handler.load_model(model_name)
        
        if device and runtime == "Torch":
            handler.set_device(device)
        if runtime == "Torch":
            inference_kwargs["device"] = handler.device
            
        worker = InferenceWorker(
            handler.model, handler.model_params, conditioning, **inference_kwargs
        )
        _, output_img = worker._tile_process(output_img, handler.model_params, progress_callback=progress_callback)

    # Postprocess
    output = postprocess(
        image_RGB, output_img,
        lumi_blend=lumi, chroma_blend=chroma,
        eps=1e-6, clip_highlights=clip_highlights, affine=affine,
    )
    primary_model_params = MODEL_REGISTRY[models[0]]
    if primary_model_params.get("demosaicing") == "sixchan" and not primary_model_params.get("subtract_bl"):
            if hasattr(rh, 'rawpy_object') and rh.rawpy_object is not None:
                output = output - np.mean(rh.rawpy_object.black_level_per_channel) / rh.rawpy_object.white_level
    return output

def run_pipeline(
    model_names: str,
    in_file: str,
    out_file: str,
    conditioning_str: Optional[str] = None,
    dims: Optional[List[int]] = None,
    cfa: bool = False,
    tile_size: int = 256,
    tile_overlap: float = 0.25,
    lumi: float = 0.0,
    chroma: float = 0.0,
    clip_highlights: bool = False,
    affine: bool = False,
    use_onnx: bool = False,
    device: Optional[str] = None,
    verbose: int = 1,
    disable_tqdm: bool = False,
    progress_callback = None,
    exiftools: bool = True,
):    

    # Initialization
    models = model_names.split(",")
    primary_model_params = MODEL_REGISTRY[models[0]]

    rh, image_RGB, iso = get_image(in_file, primary_model_params, dims=dims)

    output = process_img(models, image_RGB, conditioning_str, iso, disable_tqdm, tile_size, tile_overlap, verbose, use_onnx, device, 
                         lumi, chroma, clip_highlights, affine, progress_callback, rh)
    # Save
    saver = ImageSaver(primary_model_params, rh, dims=dims)
    apply_ccm = primary_model_params["demosaicing"] in ("rawpy", "sixchan")
    
    if Path(out_file).suffix == ".tiff":
        saver.to_tiff(output, out_file, apply_ccm=apply_ccm)
    else:
        saver.to_raw(output, out_file, cfa)

    if exiftools:
        copy_with_exiftools(in_file, out_file, verbose)

    if verbose > 0:
        print(f"{out_file} saved!")


def main():
    parser = argparse.ArgumentParser(
        description="A command line utility for processing raw images."
    )
    parser.add_argument("model", type=str, help="The name of the model to use.")
    parser.add_argument("in_file", type=str, help="The name of the file to open.")
    parser.add_argument("out_file", type=str, help="The name of the file to save.")
    parser.add_argument(
        "--conditioning",
        type=str,
        help="Conditioning array to feed model. Input string of numbers like so: 1,2,3",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=4,
        metavar=("x0", "x1", "y0", "y1"),
        help="Optional crop dimensions.",
    )

    parser.add_argument(
        "--cfa",
        action="store_true",
        help="Save the image as a CFA image (default: False).",
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable the progress bar."
    )
    parser.add_argument(
        "--tile_size", type=int, help="Set tile size. (default: 256)", default=256
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        help="Set tile overlap. (default: 0.25)",
        default=0.25,
    )

    parser.add_argument("--lumi", type=float, help="Lumi noise (0-1).", default=0)
    parser.add_argument("--chroma", type=float, help="Chroma noise (0-1).", default=0)
    parser.add_argument(
        "--clip_highlights",
        action="store_true",
        help="Do not run model on clipped highlights.",
    )
    parser.add_argument(
        "--affine", action="store_true", help="Affine fit the model to the input."
    )

    parser.add_argument(
        "--onnx", action="store_true", help="Run on ONNX backend. (Default Torch)."
    )
    parser.add_argument(
        "--device", type=str, help="Torch only: Set device backend (cuda, cpu, mps)."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        help="Verbose output. 0:Silent 1:Progress bar 2:Verbose (default: 1)",
        default=1,
    )
    parser.add_argument(
        "--disable_exiftools",
        help = "Do not use exiftools to copy exif information.",
        action="store_false"
    )

    args = parser.parse_args()
    run_pipeline(
        model_names=args.model,
        in_file=args.in_file,
        out_file=args.out_file,
        conditioning_str=args.conditioning,
        dims=args.dims,
        cfa=args.cfa,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        lumi=args.lumi,
        chroma=args.chroma,
        clip_highlights=args.clip_highlights,
        affine=args.affine,
        use_onnx=args.onnx,
        device=args.device,
        verbose=args.verbose,
        disable_tqdm=args.disable_tqdm,
        exiftools=args.disable_exiftools,
    )

if __name__ == "__main__":
    main()
