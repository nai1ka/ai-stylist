from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from stylist.segmentation import FaceParsing


@dataclass(frozen=True)
class RegionColor:
    name: str
    label: str
    mask: np.ndarray
    median_rgb: tuple[int, int, int] | None


def _normalize_label(label: str) -> str:
    return label.lower().replace(".png", "").strip()


def build_color_regions(image: Image.Image, parsing: FaceParsing) -> list[RegionColor]:
    """Create skin, hair and central eye-area regions with robust median RGB.

    Eye centers are only a coarse display/sample region. This model does not
    segment iris pixels, so these values must never be described as iris color.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    shape_ids = {_normalize_label(label): index for index, label in parsing.class_names.items()}

    def ids_for(*names: str) -> list[int]:
        return [shape_ids[name] for name in names if name in shape_ids]

    def make(name: str, label: str, classes: list[int], pupil_center: bool = False) -> RegionColor:
        mask = np.isin(parsing.labels, classes) if classes else np.zeros(parsing.labels.shape, bool)
        if pupil_center and mask.any():
            yy, xx = np.where(mask)
            x0, x1 = int(xx.min()), int(xx.max())
            y0, y1 = int(yy.min()), int(yy.max())
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = max(1, (x1 - x0) * 0.18), max(1, (y1 - y0) * 0.2)
            y_grid, x_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
            center_area = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 <= 1
            mask &= center_area
        count = int(mask.sum())
        median = tuple(int(v) for v in np.median(pixels[mask], axis=0)) if count else None
        return RegionColor(name, label, mask, median)

    left_eye = make("left_eye", "", ids_for("l_eye"), True)
    right_eye = make("right_eye", "", ids_for("r_eye"), True)
    eye_mask = left_eye.mask | right_eye.mask
    eye_pixels = pixels[eye_mask]
    eye_median = tuple(int(v) for v in np.median(eye_pixels, axis=0)) if len(eye_pixels) else None

    return [
        make("skin", "Кожа", ids_for("skin")),
        make("hair", "Волосы", ids_for("hair")),
        RegionColor("eye_center", "Цвет глаз", eye_mask, eye_median),
    ]


def render_masks(image: Image.Image, regions: list[RegionColor]) -> Image.Image:
    """Render a compact color mask overlay for a demo, retaining original pixels."""
    colors = {
        "skin": (255, 100, 130),
        "hair": (55, 220, 255),
        "eye_center": (255, 220, 50),
    }
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for region in regions:
        if region.mask.any():
            tint = np.asarray(colors[region.name], dtype=np.float32)
            overlay[region.mask] = base[region.mask] * 0.38 + tint * 0.62
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def render_region_mask(image: Image.Image, region: RegionColor) -> Image.Image:
    """Make a transparent-background colored mask preview."""
    color = {
        "skin": (255, 100, 130),
        "hair": (55, 220, 255),
        "eye_center": (255, 220, 50),
    }[region.name]
    output = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    output[region.mask] = (*color, 230)
    return Image.fromarray(output, mode="RGBA")
