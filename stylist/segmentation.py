from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from stylist.config import FACE_PARSING_DIR


class SegmentationUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceParsing:
    labels: np.ndarray
    class_names: dict[int, str]


_model_lock = Lock()


def load_face_parser():
    if not (FACE_PARSING_DIR / "config.json").is_file():
        raise SegmentationUnavailableError(
            "Модель сегментации не найдена. Скачайте jonathandinu/face-parsing в models/face_parsing."
        )
    try:
        processor = AutoImageProcessor.from_pretrained(FACE_PARSING_DIR, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            FACE_PARSING_DIR, local_files_only=True
        ).to("cpu").eval()
        return processor, model
    except Exception as exc:
        raise SegmentationUnavailableError(f"Не удалось загрузить SegFormer: {exc}") from exc


def segment_face(image: Image.Image, processor, model) -> FaceParsing:
    """Segment the full prepared portrait and resize class IDs to its dimensions."""
    try:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with _model_lock, torch.inference_mode():
            logits = model(**inputs).logits
            logits = F.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        id2label = model.config.id2label
        class_names = {int(index): str(label) for index, label in id2label.items()}
        return FaceParsing(labels, class_names)
    except Exception as exc:
        raise SegmentationUnavailableError(f"Ошибка сегментации портрета: {exc}") from exc
