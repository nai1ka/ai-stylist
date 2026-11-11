"""CPU-only, offline inference for the author's four-season checkpoint."""
import json
from pathlib import Path

import torch
from torch import nn
from PIL import Image

from stylist.config import MODELS_DIR
from stylist.face import ModelUnavailableError
from stylist.schemas import SeasonalAnalysis

CLASSES = ["autumn", "spring", "summer", "winter"]
SEASON_LABELS = dict(zip(CLASSES, ["Осень", "Весна", "Лето", "Зима"]))
MODEL_ID = "menes000/armocromia-4-season-effnetv2s"
MODEL_DIR = MODELS_DIR / "seasonal"
WEIGHTS_NAME = "armocromia_4season_effnetv2s.pth"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_seasonal_model(directory: Path = MODEL_DIR):
    try:
        import timm
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["model_id"] != MODEL_ID or not metadata["revision"]:
            raise ValueError("Некорректная ревизия модели")
        backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0)
        head = nn.Sequential(nn.Linear(backbone.num_features, 512), nn.BatchNorm1d(512),
                             nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4))
        model = nn.Sequential(backbone, head)
        state = torch.load(directory / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("Ожидался непустой state_dict")
        # The published checkpoint is a timm model with its classifier replaced:
        # backbone keys are at the root (conv_stem.*, blocks.*, ...), while the
        # head uses classifier.*. Also accept the documented wrapped backbone.
        backbone_keys = set(backbone.state_dict())
        head_keys = {"classifier." + key for key in head.state_dict()}
        bare_keys = backbone_keys | head_keys
        wrapped_keys = {"backbone." + key for key in backbone_keys} | head_keys
        actual_keys = set(state)
        if actual_keys == bare_keys:
            prefix = ""
        elif actual_keys == wrapped_keys:
            prefix = "backbone."
        else:
            expected = wrapped_keys if any(k.startswith("backbone.") for k in state) else bare_keys
            raise ValueError(
                f"Ключи checkpoint не совпадают: отсутствуют {sorted(expected - actual_keys)[:8]}; "
                f"лишние {sorted(actual_keys - expected)[:8]}"
            )
        remapped = {
            ("1." + key[len("classifier."):] if key.startswith("classifier.")
             else "0." + key[len(prefix):]): value
            for key, value in state.items()
        }
        model.load_state_dict(remapped, strict=True)
        return model.to("cpu").eval(), metadata["revision"]
    except Exception as exc:
        raise ModelUnavailableError(
            f"Цветотип недоступен: {exc}. Подготовьте модель: python scripts/download_models.py --seasonal-only"
        ) from exc


def predict_batch(model, images):
    transform = image_transform()
    inputs = torch.stack([transform(image.convert("RGB")) for image in images])
    with torch.inference_mode():
        logits = model(inputs)
        if logits.shape != (len(images), 4) or not torch.isfinite(logits).all():
            raise ValueError("Некорректный выход сезонной модели: ожидался конечный тензор (N, 4)")
        probabilities = logits.softmax(dim=1)
        if not torch.allclose(probabilities.sum(1), torch.ones(len(images)), atol=1e-5):
            raise ValueError("Некорректная сумма оценок модели")
    return probabilities.cpu().tolist()


def analyze_seasonal(image: Image.Image, model, revision: str) -> SeasonalAnalysis:
    scores = dict(zip(CLASSES, predict_batch(model, [image])[0]))
    return SeasonalAnalysis(max(scores, key=scores.get), scores, MODEL_ID, revision)
