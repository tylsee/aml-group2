"""High-contrast Grad-CAM image overlay.

Grad-CAM measures attribution to an image-level class score; it does not produce
segmentation masks, lesion boxes, diagnoses or ground-truth annotations.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, ImageFilter


def vivid_gradcam(image: Image.Image, cam: np.ndarray) -> tuple[Image.Image, bool]:
    """Blend yellow -> orange -> red only where attribution is nonzero.

    The CAM is bilinearly resized to the displayed model crop. Its magnitude is
    normalised by the attribution adapter; absence of usable CAM is not faked.
    Returns (image, has_signal).
    """
    base = ImageOps.exif_transpose(image).convert('RGB')
    w, h = base.size
    a = np.asarray(cam, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError('Grad-CAM must be a 2D activation map')
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.shape != (h, w):
        a = np.array(Image.fromarray(np.uint8(np.clip(a, 0, 1)*255))
                     .resize((w,h), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    a = np.clip(np.nan_to_num(a, nan=0, posinf=0, neginf=0), 0, 1)
    if float(a.max()) < 1e-8:
        return base.copy(), False

    # Contrast curve emphasises truly influential areas while leaving low
    # activation pixels legible and nearly unchanged.
    strength = np.clip((a - .10) / .82, 0, 1)
    opacity = (strength ** 0.78) * .83
    image_arr = np.asarray(base, dtype=np.float32)

    # At low influence: bright yellow; medium: tangerine; high: crimson red.
    t = np.clip((strength - .40) / .60, 0, 1)
    r = np.full_like(a, 255.0)
    g = np.where(strength < .40, 232-72*strength/.40, 160-107*t)
    b = np.where(strength < .40, 55-10*strength/.40, 45-10*t)
    colour = np.stack((r,g,b),axis=-1)
    mix = image_arr*(1-opacity[...,None])+colour*opacity[...,None]

    # A fine white edge around strong activation patches makes them much easier
    # to identify against dark or green leaves. This is a saliency contour,
    # NOT a traced leaf lesion boundary.
    high_mask = Image.fromarray(np.uint8(a >= .64)*255, 'L')
    dilation = np.asarray(high_mask.filter(ImageFilter.MaxFilter(7)),dtype=np.int16)
    erosion = np.asarray(high_mask.filter(ImageFilter.MinFilter(5)),dtype=np.int16)
    edge = (dilation > erosion) & (a >= .25)
    mix[edge] = mix[edge]*.2+np.array([255,253,233],dtype=np.float32)*.8
    return Image.fromarray(np.uint8(np.clip(mix,0,255)), 'RGB'), True
