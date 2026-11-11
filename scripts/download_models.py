from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import argparse
import json
import shutil

# When invoked as ``python scripts/download_models.py``, Python puts only the
# scripts directory on sys.path. Add the project root for local package imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stylist.config import FACE_SHAPE_DIR, LANDMARKER_PATH, ROOT
from stylist.config import FACE_PARSING_DIR

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def download_file(url: str, destination: Path) -> None:
    from urllib.request import urlopen

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        return
    temp = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=60) as response, temp.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if temp.stat().st_size <= 1024:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Скачанный файл подозрительно мал: {destination}")
    temp.replace(destination)


def download_seasonal() -> None:
    from stylist.seasonal import MODEL_DIR, MODEL_ID, WEIGHTS_NAME
    revision = HfApi().model_info(MODEL_ID).sha
    cached = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHTS_NAME, revision=revision,
                             cache_dir=str(ROOT / "models" / ".hf_cache"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, MODEL_DIR / WEIGHTS_NAME)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "revision": revision}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasonal-only", action="store_true")
    args = parser.parse_args()
    download_seasonal()
    if args.seasonal_only:
        return
    ROOT.mkdir(exist_ok=True)
    download_file(LANDMARKER_URL, LANDMARKER_PATH)
    snapshot_download(
        repo_id="metadome/face_shape_classification",
        local_dir=str(FACE_SHAPE_DIR),
        allow_patterns=["config.json", "preprocessor_config.json", "processor_config.json", "*.safetensors", "pytorch_model.bin"],
    )
    snapshot_download(
        repo_id="jonathandinu/face-parsing",
        revision="8ee285cb01328e6b6d0d33f2cdbc6d8505ac069d",
        local_dir=str(FACE_PARSING_DIR),
        allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "pytorch_model.bin"],
    )
    print(f"Модели подготовлены в {ROOT / 'models'}")


if __name__ == "__main__":
    main()
