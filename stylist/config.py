from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
FACE_SHAPE_DIR = MODELS_DIR / "face_shape"
FACE_PARSING_DIR = MODELS_DIR / "face_parsing"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 1600
