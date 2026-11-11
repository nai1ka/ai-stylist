from dataclasses import dataclass


@dataclass(frozen=True)
class BodyMeasurements:
    bust_cm: float
    waist_cm: float
    hips_cm: float


@dataclass(frozen=True)
class BodyAnalysis:
    predicted_shape: str | None
    matched_rule_ids: list[str]
    explanation: str


@dataclass(frozen=True)
class FaceAnalysis:
    predicted_shape: str
    scores: dict[str, float]
    model_id: str
    model_revision: str


@dataclass(frozen=True)
class Recommendation:
    id: str
    title: str
    description: str
    reason: str
    source_ids: list[str]


@dataclass(frozen=True)
class SeasonalAnalysis:
    predicted_season: str
    scores: dict[str, float]
    model_id: str
    model_revision: str
