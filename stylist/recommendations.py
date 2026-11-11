import json
from pathlib import Path

from stylist.config import ROOT
from stylist.schemas import Recommendation

DATA_DIR = ROOT / "data"


def _read(name: str) -> list[dict]:
    with (DATA_DIR / name).open(encoding="utf-8") as file:
        return json.load(file)


def get_hairstyles(face_shape: str, length_preference: str = "any") -> list[Recommendation]:
    return [
        Recommendation(row["id"], row["title"], row["description"], row["reason"], row["source_ids"])
        for row in _read("hairstyles.json")
        if face_shape in row["face_shapes"] and (length_preference == "any" or row["length"] == length_preference)
    ][:3]


def get_clothing(body_shape: str, style: str) -> list[Recommendation]:
    return [
        Recommendation(row["id"], row["title"], row["description"], row["reason"], row["source_ids"])
        for row in _read("clothing.json")
        if body_shape in row["body_shapes"] and style in row["styles"]
    ][:3]
