import sys
import platform
from  RawForge.application.ModelHandler import ModelHandler  # Adjust import based on your structure
import argparse

def main():
    parser = argparse.ArgumentParser(description='A command line utility for processing raw images.')
    parser.add_argument('model', type=str, help='The name of the model to use.')
    parser.add_argument('in_file', type=str, help='The name of the file to open.')
    parser.add_argument('out_file', type=str, help='The name of the file to save.')
    parser.add_argument('--conditioning', type=str, help='Conditioning array to feed model.')
    parser.add_argument('--dims', type=str, help='Dims to process the image.')
    parser.add_argument('--cfa', action='store_true', help='Save the image as a CFA image (default: False)')

    args = parser.parse_args()

    handler = ModelHandler()

    print(args.model)
    handler.load_model(args.model)

    iso = handler.load_rh(args.in_file)
    print(iso)


    if not args.conditioning:
        conditioning  = [iso, 0]

    img, denoised_image = handler.run_inference(conditioning=conditioning, dims=args.dims)

    handler.handle_full_image(denoised_image, args.out_file, args.cfa)


if __name__ == '__main__':
    main()
