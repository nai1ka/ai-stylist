"""Provisional approximation; exact author mask/crop has not been recovered."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from stylist.face import detect_single_face
from stylist.segmentation import FaceParsing


@dataclass(frozen=True)
class SeasonalPreprocessing:
    # Retain facial features, ears and hair, exclude neck/clothes/background.
    mask_classes: tuple[str, ...] = (
        "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
        "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair",
    )
    crop_scale: float = 1.6
    center_y_offset: float = -0.12
    preserve_mouth_landmarks: bool = True


# Ordered outer lip contour in MediaPipe's 468-point face topology. Fill the
# entire polygon, including the mouth interior, using the original RGB pixels.
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185)


def prepare_seasonal_image(image: Image.Image, parsing: FaceParsing,
                           settings: SeasonalPreprocessing = SeasonalPreprocessing(),
                           landmarks=None) -> Image.Image:
    if landmarks is None:
        landmarks = detect_single_face(image)
    if parsing.labels.shape != (image.height, image.width):
        raise ValueError("Координаты фотографии и маски не совпадают")
    if settings.crop_scale <= 0 or not math.isfinite(settings.crop_scale):
        raise ValueError("Масштаб обрезки должен быть положительным")
    ids = [i for i, label in parsing.class_names.items()
           if label.lower().replace('.png', '').strip() in settings.mask_classes]
    mask = np.isin(parsing.labels, ids)
    if not mask.any():
        raise ValueError("Маска для определения цветотипа пуста")
    if settings.preserve_mouth_landmarks:
        if len(landmarks) <= max(OUTER_LIPS):
            raise ValueError("Недостаточно точек MediaPipe для сохранения области рта")
        mouth = Image.new("L", image.size, 0)
        contour = [(landmarks[i].x * image.width, landmarks[i].y * image.height)
                   for i in OUTER_LIPS]
        if not np.isfinite(contour).all():
            raise ValueError("Некорректные координаты контура губ")
        ImageDraw.Draw(mouth).polygon(contour, fill=255)
        mask |= np.asarray(mouth) > 0
    xs = [p.x * image.width for p in landmarks]
    ys = [p.y * image.height for p in landmarks]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    side = math.ceil(max(width, height) * settings.crop_scale)
    if side < 1:
        raise ValueError("Не удалось определить область лица")
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2 + height * settings.center_y_offset
    left, top = round(cx - side / 2), round(cy - side / 2)
    box = (left, top, left + side, top + side)
    # Identical box for RGB and binary mask, black padding at image edges.
    masked = np.asarray(image.convert("RGB")).copy()
    masked[~mask] = 0
    crop = Image.fromarray(masked).crop(box).resize((224, 224), Image.Resampling.BILINEAR)
    binary = Image.fromarray(mask.astype(np.uint8) * 255).crop(box)
    binary = binary.resize((224, 224), Image.Resampling.NEAREST)
    keep = np.asarray(binary) > 0
    if not keep.any():
        raise ValueError("В области лица маска пуста")
    pixels = np.asarray(crop).copy()
    pixels[~keep] = 0
    return Image.fromarray(pixels)
