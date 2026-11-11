"""Evaluate prepared ImageFolder data without segmentation or training."""
import argparse
import csv
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_ID = "menes000/armocromia-12-season"


def collect(directory):
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        subtype = path.parent.name
        season = subtype.split("_", 1)[0]
        if "_" not in subtype or season not in CLASSES:
            raise ValueError(f"Неизвестное имя класса: {subtype}")
        rows.append((path, season))
    if not rows:
        raise ValueError(f"Нет изображений ImageFolder: {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/armocromia-12-season")
    parser.add_argument("--download", action="store_true", help="Download pinned snapshot and record its revision")
    parser.add_argument("--revision", default="main", help="Dataset revision to resolve when downloading")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/seasonal")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size и threads должны быть положительными")
    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from stylist.seasonal import load_seasonal_model, predict_batch, MODEL_ID, MEAN, STD
    from stylist.seasonal_preprocessing import SeasonalPreprocessing

    torch.set_num_threads(args.threads)
    if args.download:
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
        snapshot_download(DATASET_ID, repo_type="dataset", revision=revision,
                          local_dir=str(args.dataset_dir),
                          cache_dir=str(ROOT / "datasets/.hf_cache"))
        (args.dataset_dir / "dataset_revision.json").write_text(json.dumps({"revision": revision}))
    metadata_path = args.dataset_dir / "dataset_revision.json"
    if not metadata_path.is_file():
        raise ValueError("Нет dataset_revision.json. Используйте --download для фиксации ревизии")
    dataset_revision = json.loads(metadata_path.read_text())["revision"]
    train = collect(args.dataset_dir / "train")
    test = collect(args.dataset_dir / "test")
    print(f"Train: {len(train)}, test: {len(test)} (ожидалось 912)", flush=True)
    model, revision = load_seasonal_model()
    # Smoke test on actual prepared images before full evaluation.
    for path, _ in test[:3]:
        with Image.open(path) as source:
            predict_batch(model, [source.convert("RGB")])
    majority = Counter(label for _, label in train).most_common(1)[0][0]
    truth, predicted, output = [], [], []
    for start in range(0, len(test), args.batch_size):
        batch = test[start:start + args.batch_size]
        images = []
        for path, _ in batch:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        for (path, label), scores in zip(batch, predict_batch(model, images)):
            prediction = CLASSES[max(range(4), key=lambda i: scores[i])]
            truth.append(label)
            predicted.append(prediction)
            output.append([path.relative_to(args.dataset_dir).as_posix(), label, prediction, *scores])
        print(f"{min(start + args.batch_size, len(test))}/{len(test)}", flush=True)
    matrix = confusion_matrix(truth, predicted, labels=CLASSES)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "timm", "huggingface_hub", "Pillow", "numpy", "scikit-learn", "matplotlib", "streamlit"]}
    metrics = {
        "model_id": MODEL_ID, "model_revision": revision,
        "dataset_id": DATASET_ID, "dataset_revision": dataset_revision,
        "train_count": len(train), "test_count": len(test), "expected_test_count": 912,
        "classes": CLASSES, "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
        "per_class": classification_report(truth, predicted, labels=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(), "predicted_distribution": {k: predicted.count(k) for k in CLASSES},
        "majority_train_class": majority, "majority_baseline_accuracy": truth.count(majority) / len(truth),
        "versions": versions, "device": "cpu", "threads": args.threads,
        "preprocessing": {"size": [224, 224], "resize": "bilinear", "mean": MEAN, "std": STD,
                          "segmentation": False, "augmentation": False},
        "user_photo_approximation": asdict(SeasonalPreprocessing()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    with (args.output / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_season", "predicted_season", *CLASSES])
        writer.writerows(output)
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(cmap="Blues")
    plt.ylabel("True season")
    plt.xlabel("Predicted season")
    plt.tight_layout()
    plt.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close()
    report = f'''# Оценка сезонного классификатора

Оценка готового классификатора на предоставленной автором тестовой части. Собственного обучения не проводилось.

Модель: {MODEL_ID}, ревизия {revision}.
Датасет: {DATASET_ID}, ревизия {dataset_revision}.
Train: {len(train)}, test: {len(test)}; ожидаемое число test — 912, совпадение: {len(test) == 912}.

Accuracy: {metrics['accuracy']:.6f}. Macro-F1: {metrics['macro_f1']:.6f}.
Базовое решение: всегда {majority}, определённый по train; accuracy на test: {metrics['majority_baseline_accuracy']:.6f}.

Имена 12 подтипов сведены к сезону по префиксу имени. Порядок классов фиксирован: autumn, spring, summer, winter.
Готовые изображения не сегментировались: RGB → Resize 224×224 (bilinear) → ToTensor → Normalize ImageNet.
CPU, eval, inference_mode, один softmax; аугментаций нет. Первые три изображения прошли проверку формы (1,4), конечности и суммы softmax.
Версии, precision/recall/F1 и распределение предсказаний — в metrics.json; построчные оценки — в predictions.csv.

Автор использовал оценочную выборку для восстановления порядка классов. Поэтому это воспроизведение опубликованного эксперимента, а не новая полностью независимая проверка.
Показатель автора «правильно распознанные объекты сезона / все объекты сезона» соответствует recall.
Значение 64,36% из карточки не является нашим результатом. Сравнения с исследованиями на других разбиениях не доказывают превосходства.

Обработка обычных фотографий — отдельная, пока не проверенная аппроксимация. Эта оценка её не проверяет.
Освещение, субъективность меток и сдвиг распределения ограничивают переносимость результата. Модель и данные предназначены для исследовательского использования.

Источники: https://huggingface.co/{MODEL_ID} и https://huggingface.co/datasets/{DATASET_ID}.
'''
    (args.output / "report.md").write_text(report)
    (args.output / "requirements-validated.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n")


if __name__ == "__main__":
    from stylist.seasonal import CLASSES
    main()
