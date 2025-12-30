import sys
import platform
from  RawForge.application.ModelHandler import ModelHandler 
import argparse

def main():
    parser = argparse.ArgumentParser(description='A command line utility for processing raw images.')
    parser.add_argument('model', type=str, help='The name of the model to use.')
    parser.add_argument('in_file', type=str, help='The name of the file to open.')
    parser.add_argument('out_file', type=str, help='The name of the file to save.')
    parser.add_argument('--conditioning', type=str, help='Conditioning array to feed model.')
    parser.add_argument('--dims', type=int, nargs=4, metavar=("x0", "x1", "y0", "y1"), help='Optional crop dimensions.')

    parser.add_argument('--cfa', action='store_true', help='Save the image as a CFA image (default: False).')
    parser.add_argument('--device', type=str, help='Set device backend (cuda, cpu, mps).')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable the progress bar.')
    parser.add_argument('--tile_size', type=int, help='Set tile size. (default: 256)', default=256)

    args = parser.parse_args()

    handler = ModelHandler()

    handler.load_model(args.model)

    iso = handler.load_rh(args.in_file)

    if not args.conditioning:
        conditioning  = [iso, 0]

    if args.device:
        handler.set_device(args.device)

    inference_kwargs = {"disable_tqdm": args.disable_tqdm,
                        "tile_size": args.tile_size}
    img, denoised_image = handler.run_inference(conditioning=conditioning, dims=args.dims, inference_kwargs=inference_kwargs)

    handler.handle_full_image(denoised_image, args.out_file, args.cfa)


if __name__ == '__main__':
    main()
