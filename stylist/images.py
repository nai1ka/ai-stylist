from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from stylist.config import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE


class InvalidImageError(ValueError):
    pass


Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def load_image(file_bytes: bytes) -> Image.Image:
    if not file_bytes or len(file_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Файл пустой или превышает ограничение 10 МБ.")
    try:
        with Image.open(BytesIO(file_bytes)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise InvalidImageError("Поддерживаются только JPEG и PNG.")
            image = ImageOps.exif_transpose(source)
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Изображение слишком большое.")
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                image = background.convert("RGB")
            else:
                image = image.convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Не удалось прочитать изображение. Выберите корректный JPEG или PNG.") from exc
