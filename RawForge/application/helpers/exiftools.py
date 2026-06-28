import shutil

def is_exiftool_installed():
    return shutil.which("exiftool") is not None

def copy_with_exiftools(source_img, target_img, verbose):
    if is_exiftool_installed():
        import exiftool
        with exiftool.ExifToolHelper() as et:
            et.execute("-tagsFromFile", source_img, target_img)
            if verbose > 1: print(f"Metadata copied from {source_img} to {target_img}")    
    else:
        if verbose > 1: print(f"Did not find exiftool")    

