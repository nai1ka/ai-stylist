from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

from stylist.config import FACE_SHAPE_DIR, LANDMARKER_PATH
from stylist.schemas import FaceAnalysis


class NoFaceError(ValueError):
    pass


class MultipleFacesError(ValueError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


_landmarker_lock = Lock()


def detect_single_face(image: Image.Image):
    """Return MediaPipe landmarks after requiring exactly one detected face."""
    if not LANDMARKER_PATH.is_file():
        raise ModelUnavailableError("Face Landmarker не найден. Выполните: python scripts/download_models.py")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
        )
        pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        with _landmarker_lock, vision.FaceLandmarker.create_from_options(options) as detector:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels))
        count = len(result.face_landmarks)
        if count == 0:
            raise NoFaceError("Лицо не найдено. Загрузите портрет анфас с хорошо видимым лицом.")
        if count > 1:
            raise MultipleFacesError("Найдено несколько лиц. Оставьте на фото одного человека.")
        return result.face_landmarks[0]
    except (NoFaceError, MultipleFacesError, ModelUnavailableError):
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"Не удалось обнаружить лицо: {exc}") from exc


def _label_to_shape(label: str) -> str | None:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "heart": "heart", "heart shape": "heart", "heart shaped": "heart",
        "oblong": "oblong", "long": "oblong", "long face": "oblong",
        "oval": "oval", "oval face": "oval", "round": "round", "round face": "round",
        "square": "square", "square face": "square",
    }
    return aliases.get(normalized)


def _crop_face(image: Image.Image, landmarks) -> tuple[Image.Image, Image.Image]:
    from PIL import ImageDraw

    width, height = image.size
    xs = [min(1.0, max(0.0, point.x)) * width for point in landmarks]
    ys = [min(1.0, max(0.0, point.y)) * height for point in landmarks]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    pad_x, pad_y = (right - left) * 0.2, (bottom - top) * 0.2
    box = (max(0, int(left - pad_x)), max(0, int(top - pad_y)), min(width, int(right + pad_x)), min(height, int(bottom + pad_y)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise NoFaceError("Не удалось определить границы лица.")
    crop = image.crop(box)
    marked = crop.copy()
    draw = ImageDraw.Draw(marked)
    radius = max(2, round(min(crop.size) / 170))
    for x, y in zip(xs, ys):
        px, py = round(x - box[0]), round(y - box[1])
        draw.ellipse((px-radius-1, py-radius-1, px+radius+1, py+radius+1), fill="#10212b")
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill="#62f5cf")
    return crop, marked


def analyze_face(image: Image.Image) -> tuple[FaceAnalysis, Image.Image]:
    if not LANDMARKER_PATH.is_file() or not any(FACE_SHAPE_DIR.glob("config.json")):
        raise ModelUnavailableError("Модели не найдены. Выполните: python scripts/download_models.py")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        landmarks = detect_single_face(image)
        crop, marked_crop = _crop_face(image, landmarks)
        # The model metadata uses the legacy ViTFeatureExtractor name, which
        # newer Auto classes no longer resolve. The ViT processor still reads
        # resize and normalization values from the downloaded config.
        processor = ViTImageProcessor.from_pretrained(FACE_SHAPE_DIR, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(FACE_SHAPE_DIR, local_files_only=True).to("cpu").eval()
        encoded = processor(images=crop, return_tensors="pt")
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits[0], dim=-1).tolist()
        labels = model.config.id2label
        scores = {}
        for index, score in enumerate(probabilities):
            label = labels.get(index, labels.get(str(index), ""))
            shape = _label_to_shape(label)
            if shape is None:
                raise ModelUnavailableError(f"Не удалось сопоставить метку модели: {label!r}.")
            scores[shape] = float(score)
        expected = {"heart", "oblong", "oval", "round", "square"}
        if set(scores) != expected:
            raise ModelUnavailableError("Модель должна содержать ровно пять ожидаемых форм лица.")
        shape = max(scores, key=scores.get)
        return FaceAnalysis(shape, scores, "metadome/face_shape_classification", "local snapshot"), marked_crop
    except (NoFaceError, MultipleFacesError, ModelUnavailableError):
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"Не удалось загрузить или выполнить модели: {exc}") from exc
