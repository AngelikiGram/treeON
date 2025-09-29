# import numpy as np
# from colormath.color_objects import sRGBColor, LabColor
# from colormath.color_conversions import convert_color
# from colormath.color_diff import delta_e_cie2000

# def patch_asscalar(a):
#     return a.item()

# setattr(np, "asscalar", patch_asscalar)

# def compute_delta_e_cie2000_batch(rgb1, rgb2):
#     """
#     Compute CIEDE2000 color difference (Delta E) between two arrays of RGB colors.
#     Args:
#         rgb1: Nx3 array of RGB colors (0-255 or 0-1)
#         rgb2: Nx3 array of RGB colors (0-255 or 0-1)
#     Returns:
#         delta_e: N-array of Delta E values
#     """
#     rgb1 = np.asarray(rgb1)
#     rgb2 = np.asarray(rgb2)
#     # Convert to 0-1 range if needed
#     if rgb1.max() > 1.0:
#         rgb1 = rgb1 / 255.0
#     if rgb2.max() > 1.0:
#         rgb2 = rgb2 / 255.0
#     delta_e = []
#     for c1, c2 in zip(rgb1, rgb2):
#         color1 = sRGBColor(*c1, is_upscaled=False)
#         color2 = sRGBColor(*c2, is_upscaled=False)
#         lab1 = convert_color(color1, LabColor)
#         lab2 = convert_color(color2, LabColor)
#         de = delta_e_cie2000(lab1, lab2)
#         # Fix for numpy.asscalar removal
#         if hasattr(de, 'item'):
#             de = de.item()
#         delta_e.append(de)
#     return np.array(delta_e)


from sklearn.neighbors import NearestNeighbors
import numpy as np
try:
    from skimage.color import rgb2lab, deltaE_ciede2000
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def srgb_to_lab_numpy_uint8(rgb_uint8):
    """rgb_uint8: (N,3) uint8 sRGB -> Lab (N,3) float"""
    if _HAS_SKIMAGE:
        rgb01 = rgb_uint8.astype(np.float32)/255.0
        return rgb2lab(rgb01)  # D65 handling is fine & consistent here
    else:
        # fallback via colormath (looped)
        out = np.empty((rgb_uint8.shape[0],3), np.float32)
        for i, (r,g,b) in enumerate(rgb_uint8.astype(np.float32)/255.0):
            lab = convert_color(sRGBColor(r,g,b, is_upscaled=False), LabColor)  # D50 default
            out[i] = [lab.lab_l, lab.lab_a, lab.lab_b]
        return out

def deltaE00_numpy(lab1, lab2):
    """lab*: (N,3). Returns (N,) ΔE00."""
    if _HAS_SKIMAGE:
        return deltaE_ciede2000(lab1, lab2)
    else:
        vals = np.empty((lab1.shape[0],), np.float32)
        for i in range(lab1.shape[0]):
            L1,a1,b1 = lab1[i]; L2,a2,b2 = lab2[i]
            de = delta_e_cie2000(LabColor(L1,a1,b1), LabColor(L2,a2,b2))
            vals[i] = float(de)
        return vals

def ensure_uint8_srgb(arr):
    """Accepts (N,3) array in [0,1] or [0,255], returns uint8 sRGB.
       If input looks linear [0,1], gamma-encode."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        # assume linear -> gamma to sRGB
        arr = np.clip(arr, 0.0, 1.0) ** (1.0/2.2)
        arr = np.round(arr * 255.0)
    else:
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)
# Export for use in other modules
__all__ = [
    'ensure_uint8_srgb',
    '_srgb_to_lab_numpy_uint8',
    '_deltaE00_numpy'
]

