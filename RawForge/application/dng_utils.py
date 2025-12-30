import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
import tifffile

def get_ratios(string, rh):
    return [x.as_integer_ratio() for x in rh.full_metadata[string].values]


def get_as_shot_neutral(rh, denominator=10000):

    cam_mul = rh.core_metadata.camera_white_balance
    
    if cam_mul[0] == 0 or cam_mul[2] == 0:
        return [[denominator, denominator], [denominator, denominator], [denominator, denominator]]

    r_neutral = cam_mul[1] / cam_mul[0]
    g_neutral = 1.0 
    b_neutral = cam_mul[1] / cam_mul[2]

    return [
        [int(r_neutral * denominator), denominator],
        [int(g_neutral * denominator), denominator],
        [int(b_neutral * denominator), denominator],
    ]
def convert_ccm_to_rational(matrix_3x3, denominator=10000):

    numerator_matrix = np.round(matrix_3x3 * denominator).astype(int)
    numerators_flat = numerator_matrix.flatten()
    ccm_rational = [[num, denominator] for num in numerators_flat]
    
    return ccm_rational


   
def simulate_CFA(image, pattern="RGGB", cfa_type="bayer"):
    """
    Simulate a CFA image from an RGB image.

    Args:
        image: numpy array (H, W, 3), RGB image.
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer,
                 or ignored if cfa_type="xtrans".
        cfa_type: "bayer" or "xtrans".

    Returns:
        cfa: numpy array (H, W) CFA image.
        sparse_mask:  numpy array (H, W, r), mask of pixels.
    """
    width = image.shape[1]
    height = image.shape[0]
    cfa = np.zeros((height, width, 3), dtype=image.dtype)
    sparse_mask = np.zeros((height, width, 3), dtype=image.dtype)
    if cfa_type == "bayer":
        # 2×2 Bayer masks
        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        if pattern not in masks:
            raise ValueError(f"Unknown Bayer pattern: {pattern}")

        mask = masks[pattern]
        cmap = {"R": 0, "G": 1, "B": 2}
         
        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                cfa[i::2, j::2, ch] = image[i::2, j::2, ch]
                sparse_mask[i::2, j::2, ch] = 1
    elif cfa_type == "xtrans":
        # Fuji X-Trans 6×6 repeating pattern
        xtrans_pattern = np.array([
            ["G","B","R","G","R","B"],
            ["R","G","G","B","G","G"],
            ["B","G","G","R","G","G"],
            ["G","R","B","G","B","R"],
            ["B","G","G","R","G","G"],
            ["R","G","G","B","G","G"],
        ])
        cmap = {"R":0, "G":1, "B":2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                cfa[i::6, j::6, ch] = image[i::6, j::6, ch]
                sparse_mask[i::2, j::2, ch] = 1
    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    return cfa.sum(axis=2), sparse_mask

def to_dng(uint_img, rh, filepath, ccm1, save_cfa=True, convert_to_cfa=True, use_orig_wb_points=False):
    width = uint_img.shape[1]
    height = uint_img.shape[0]
    bpp = 16 

    t = DNGTags()

    if save_cfa:
      if convert_to_cfa:
        cfa, _ = simulate_CFA(uint_img, pattern="RGGB", cfa_type="bayer")
        uint_img = cfa.astype(np.uint16)
      t.set(Tag.BitsPerSample, bpp)
      t.set(Tag.SamplesPerPixel, 1) 
      t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
      t.set(Tag.CFARepeatPatternDim, [2,2])
      t.set(Tag.CFAPattern, CFAPattern.RGGB)
      t.set(Tag.BlackLevelRepeatDim, [2,2])

      # This should not be used except to save testing patches
      if use_orig_wb_points:
        bl = rh.core_metadata.black_level_per_channel
        t.set(Tag.BlackLevel, bl)
        t.set(Tag.WhiteLevel, rh.core_metadata.white_level)
      else:
        t.set(Tag.BlackLevel, [0, 0, 0, 0])
        t.set(Tag.WhiteLevel, 65535)
    else:
      t.set(Tag.BitsPerSample, [bpp, bpp, bpp]) # 3 channels for RGB
      t.set(Tag.SamplesPerPixel, 3) # 3 for RGB
      t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Linear_Raw)
      t.set(Tag.BlackLevel,[0,0,0])
      t.set(Tag.WhiteLevel, [65535, 65535, 65535])

    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.PlanarConfiguration, 1) # 1 for chunky (interleaved RGB)

    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)

    t.set(Tag.ColorMatrix1, ccm1)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    wb = get_as_shot_neutral(rh)
    t.set(Tag.AsShotNeutral, wb)
    t.set(Tag.BaselineExposure, [[0,100]])


    try:
      t.set(Tag.Make, rh.full_metadata['Image Make'].values)
      t.set(Tag.Model, rh.full_metadata['Image Model'].values)
      t.set(Tag.Orientation, rh.full_metadata['Image Orientation'].values[0])
      exposures = get_ratios('EXIF ExposureTime', rh)
      fnumber = get_ratios('EXIF FNumber', rh)
      ExposureBiasValue = get_ratios('EXIF ExposureBiasValue', rh) 
      FocalLength = get_ratios('EXIF FocalLength', rh) 
      t.set(Tag.FocalLength, FocalLength)
      t.set(Tag.EXIFPhotoLensModel, rh.full_metadata['EXIF LensModel'].values)
      t.set(Tag.ExposureBiasValue, ExposureBiasValue)
      t.set(Tag.ExposureTime, exposures)
      t.set(Tag.FNumber, fnumber)
      t.set(Tag.PhotographicSensitivity, rh.full_metadata['EXIF ISOSpeedRatings'].values)
    except:
      print("Could not save EXIF")
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.Adobe_RGB)

    r = RAW2DNG()

    r.options(t, path="", compress=False)

    r.convert(uint_img, filename=filepath)


def to_tiff_dng(uint_img, rh, filepath, ccm1, save_cfa=True, convert_to_cfa=True, use_orig_wb_points=False):
    """
    Saves image as a DNG-compatible TIFF with full metadata.
    Works on Windows using tifffile.
    """
    print("to tiff")
    # 1. Prepare Image Data
    if save_cfa:
        if convert_to_cfa:
            cfa, _ = simulate_CFA(uint_img, pattern="RGGB", cfa_type="bayer")
            data = cfa.astype(np.uint16)
        else:
            data = uint_img.astype(np.uint16)
        samples_per_pixel = 1
        # 32803 is the code for Color Filter Array
        photometric = 32803 
    else:
        data = uint_img.astype(np.uint16)
        samples_per_pixel = 3
        # 34892 for Linear Raw (or 1 for BlackIsZero)
        photometric = 34892 

    # 2. Define DNG/TIFF Tags
    # Tag IDs based on Adobe DNG Specification
    tags = []

    # Basic Geometry
    height, width = data.shape[:2]
    
    # Black and White Levels
    if save_cfa and use_orig_wb_points:
        bl = rh.core_metadata.black_level_per_channel
        wl = rh.core_metadata.white_level
    else:
        bl = [0, 0, 0, 0] if save_cfa else [0, 0, 0]
        wl = 65535

    # DNG Metadata Mapping
    # Format: (code, data_type, count, value, writeonce)
    # Types: 3=short, 4=long, 5=rational, 2=ascii, 12=double
    
    # DNG Version (1.4.0.0) - Type 1 (BYTE)
    tags.append((50706, 1, 4, [1, 4, 0, 0], True)) 

    # ColorMatrix1 (Tag 50721)
    # Type 10 = SRATIONAL. Count = 9 elements (each is a [num, den] pair)
    ccm1_flat = np.array(ccm1).flatten().tolist()
    tags.append((50721, 10, 9, ccm1_flat, True))

    # AsShotNeutral (Tag 50728)
    # Type 5 = RATIONAL. Count = 3 elements
    wb = get_as_shot_neutral(rh)
    wb_flat = np.array(wb).flatten().tolist()
    tags.append((50728, 5, 3, wb_flat, True))

    # BlackLevel (Tag 50714)
    # RATIONAL (5), count is 4 for Bayer
    if any(isinstance(i, list) for i in bl) or (isinstance(bl, np.ndarray) and bl.ndim > 1):
        bl_flat = np.array(bl).flatten().tolist()
    else:
        # Convert simple list [0, 0, 0, 0] to [0, 1, 0, 1, 0, 1, 0, 1]
        bl_flat = []
        for val in bl:
            bl_flat.extend([int(val), 1])
    tags.append((50714, 5, len(bl), bl_flat, True))

    # WhiteLevel (Tag 50717) 
    # LONG (4) or SHORT (3)
    tags.append((50717, 4, 1, int(wl), True))

    if save_cfa:
        # CFARepeatPatternDim: [Rows, Cols] -> Type 3 (SHORT)
        tags.append((33421, 3, 2, [2, 2], True))
        
        # CFAPattern: [0, 1, 1, 2] for RGGB -> Type 1 (BYTE)
        # 0=Red, 1=Green, 2=Blue
        tags.append((33422, 1, 4, [0, 1, 1, 2], True))

    # # 3. EXIF Extraction
    # try:
    #     make = str(rh.full_metadata['Image Make'].values[0])
    #     model = str(rh.full_metadata['Image Model'].values[0])
    #     tags.append((271, 'z', len(make), make, True)) # Make
    #     tags.append((272, 'z', len(model), model, True)) # Model
        
    #     # Exposure values (Rationals)
    #     def to_rat(val): return (int(val * 1000), 1000)
        
    #     tags.append((33434, '5L', 1, to_rat(get_ratios('EXIF ExposureTime', rh)), True))
    #     tags.append((33437, '5L', 1, to_rat(get_ratios('EXIF FNumber', rh)), True))
    #     tags.append((34855, '3L', 1, int(rh.full_metadata['EXIF ISOSpeedRatings'].values[0]), True))
    # except Exception as e:
    #     print(f"Warning: Could not extract all EXIF data: {e}")

    # 4. Save the file
    tifffile.imwrite(
        filepath,
        data,
        photometric=photometric,
        planarconfig=1,
        extrasamples=[],
        extratags=tags
    )

def convert_color_matrix(matrix):
  """
  Converts a 3x3 NumPy matrix of floats into a list of integer pairs.

  Each float value in the matrix is converted to a fractional representation
  with a denominator of 10000. The numerator is calculated by scaling the
  float value by 10000 and rounding to the nearest integer.

  Args:
    matrix: A 3x3 NumPy array with floating-point numbers.

  Returns:
    A list of 9 lists, where each inner list contains two integers
    representing the numerator and denominator.
  """
  # Ensure the input is a NumPy array
  if not isinstance(matrix, np.ndarray):
    raise TypeError("Input must be a NumPy array.")

  # Flatten the 3x3 matrix into a 1D array of 9 elements
  flattened_matrix = matrix.flatten()

  # Initialize the list for the converted matrix
  converted_list = []
  denominator = 10000

  # Iterate over each element in the flattened matrix
  for element in flattened_matrix:
    # Scale the element, round it to the nearest integer, and cast to int
    numerator = int(round(element * denominator))
    # Append the [numerator, denominator] pair to the result list
    converted_list.append([numerator, denominator])

  return converted_list