import math

from stylist.schemas import BodyAnalysis, BodyMeasurements


SHAPES = {
    "hourglass": "Песочные часы",
    "pear": "Груша",
    "inverted_triangle": "Перевёрнутый треугольник",
    "rectangle": "Прямоугольник",
    "apple": "Яблоко",
}


class InvalidMeasurementsError(ValueError):
    """Measurements must be finite positive numbers."""


def classify_body(measurements: BodyMeasurements) -> BodyAnalysis:
    values = (measurements.bust_cm, measurements.waist_cm, measurements.hips_cm)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
        raise InvalidMeasurementsError("Все мерки должны быть конечными числами больше нуля.")

    bust, waist, hips = values
    balanced = max(bust, hips) / min(bust, hips) <= 1.05
    rules = {
        "hourglass": balanced and waist <= 0.75 * bust and waist <= 0.75 * hips,
        "rectangle": balanced and 0.75 * bust < waist < min(bust, hips),
        "pear": hips >= 1.05 * bust and waist < min(bust, hips),
        "inverted_triangle": bust > 1.05 * hips and hips < waist and hips < bust,
        "apple": 0.95 * bust <= hips < bust and waist >= bust,
    }
    matched = [shape for shape, applies in rules.items() if applies]
    if len(matched) == 1:
        shape = matched[0]
        explanation = f"Мерки соответствуют правилу «{SHAPES[shape]}» по выбранным пропорциям."
        return BodyAnalysis(shape, matched, explanation)
    if matched:
        labels = ", ".join(SHAPES[shape] for shape in matched)
        explanation = f"Мерки подходят нескольким категориям ({labels}); выберите тип фигуры вручную."
    else:
        explanation = "Правила не дают однозначной категории; выберите тип фигуры вручную."
    return BodyAnalysis(None, matched, explanation)
