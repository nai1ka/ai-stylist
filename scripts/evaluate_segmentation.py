#!/usr/bin/env python3
"""Evaluate the app's local face parser against a reproducible LaPa test sample."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stylist.segmentation import load_face_parser, segment_face  # noqa: E402

MODEL_ID = "jonathandinu/face-parsing"
MODEL_REVISION = "8ee285cb01328e6b6d0d33f2cdbc6d8505ac069d"
OFFICIAL_LABELMAP_URL = "https://github.com/jd-opensource/lapa-dataset/blob/master/labelmap.txt"
LAPA_LABELS = {
    0: "background", 1: "skin", 2: "left eyebrow", 3: "right eyebrow",
    4: "left eye", 5: "right eye", 6: "nose", 7: "upper lip",
    8: "inner mouth", 9: "lower lip", 10: "hair",
}
# Shared presentation colors for the three evaluated regions (RGB).
REGION_COLORS = {"skin": (64, 190, 160), "hair": (220, 80, 160), "eyes": (250, 205, 70)}
REGIONS = ("skin", "hair", "eyes")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".tif", ".tiff", ".bmp"}


def counts_metrics(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    union = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    return {
        "iou": (tp / union) if union else None,
        "dice": ((2 * tp) / denom_dice) if denom_dice else None,
    }


def check_metric_functions() -> None:
    # identical non-empty, disjoint, both empty, and one-sided empty cases
    assert counts_metrics(4, 0, 0) == {"iou": 1.0, "dice": 1.0}
    assert counts_metrics(0, 3, 2) == {"iou": 0.0, "dice": 0.0}
    assert counts_metrics(0, 0, 0) == {"iou": None, "dice": None}
    assert counts_metrics(0, 2, 0) == {"iou": 0.0, "dice": 0.0}


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode not in {"1", "L", "P", "I", "I;16", "I;16B", "I;16L"}:
            raise ValueError(f"Ожидалась целочисленная одноканальная маска, получен режим {im.mode!r}")
        arr = np.asarray(im).copy()
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Маска должна быть двумерной целочисленной: shape={arr.shape}, dtype={arr.dtype}")
    values = set(map(int, np.unique(arr)))
    invalid = sorted(values - set(LAPA_LABELS))
    if invalid:
        raise ValueError(f"Неизвестные ID классов LaPa: {invalid}")
    return arr.astype(np.uint8, copy=False)


def image_id(path: Path) -> str:
    # Pairing uses exact filename stem; no positional pairing or fuzzy matching.
    return path.stem


def split_members(root: Path, split: str) -> tuple[set[str], bool]:
    ids: set[str] = set()
    found = False
    # Common extraction layouts either put split in path components or provide a split list.
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = {part.lower() for part in p.relative_to(root).parts[:-1]}
        if split.lower() in parts:
            ids.add(p.stem)
            found = True
    for list_path in root.rglob(f"{split}.txt"):
        if not list_path.is_file():
            continue
        found = True
        for line in list_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Official split files may contain tab-separated relative image/mask/landmark paths.
            for token in line.split("\t"):
                candidate = Path(token.strip())
                if candidate.suffix.lower() in IMAGE_EXTS | MASK_EXTS:
                    ids.add(candidate.stem)
                elif candidate.name:
                    ids.add(candidate.stem)
    return ids, found


def discover_pairs(root: Path, split: str) -> list[dict[str, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root не найден: {root}\nРаспакуйте архив LaPa сюда или укажите --dataset-root.")
    test_ids, split_found = split_members(root, split)
    files = [p for p in root.rglob("*") if p.is_file()]
    mask_candidates = [p for p in files if p.suffix.lower() in MASK_EXTS and any(k in p.as_posix().lower() for k in ("mask", "label", "parsing", "seg"))]
    # If directory names are unusual, use grayscale/integer PNG files as mask candidates later.
    by_stem: dict[str, list[Path]] = {}
    for p in mask_candidates:
        by_stem.setdefault(image_id(p), []).append(p)
    pairs: list[dict[str, str]] = []
    used_ids: Counter[str] = Counter()
    for img in files:
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        relparts = {part.lower() for part in img.relative_to(root).parts[:-1]}
        if any(k in part for part in relparts for k in ("mask", "label", "landmark", "point", "annot")):
            continue
        # The archive can reuse a filename ID in more than one split. Never let
        # a matching ID pull a train/val file into the requested split.
        known_splits = {"train", "val", "test"}
        if any(part in known_splits and part != split.lower() for part in relparts):
            continue
        iid = image_id(img)
        if split_found and iid not in test_ids:
            # Explicit test directory wins even when the split file carries paths to other assets.
            if split.lower() not in relparts:
                continue
        candidates = by_stem.get(iid, [])
        # Resolve duplicate stems only within the same split subtree where possible.
        same_split = [m for m in candidates if split.lower() in {x.lower() for x in m.relative_to(root).parts[:-1]}]
        if same_split:
            candidates = same_split
        if len(candidates) != 1:
            if len(candidates) > 1:
                raise ValueError(f"Несколько масок для ID {iid!r}: {[p.relative_to(root).as_posix() for p in candidates]}")
            continue
        mask = candidates[0]
        if split_found and iid not in test_ids and split.lower() not in relparts:
            continue
        used_ids[iid] += 1
        pairs.append({"image_id": iid, "image": img.relative_to(root).as_posix(), "mask": mask.relative_to(root).as_posix()})
    if not split_found:
        raise ValueError(f"Не найден способ определить split={split!r}: ожидается каталог с этим именем или файл {split}.txt")
    if not pairs:
        raise ValueError(f"Не найдено ни одной пары изображение/маска для split={split!r}. Осмотренные корневые каталоги: {[p.name for p in root.iterdir()]}")
    duplicates = sorted(k for k, v in used_ids.items() if v > 1)
    if duplicates:
        raise ValueError(f"Повторяющиеся ID файлов в split {split}: {duplicates[:20]}")
    pairs.sort(key=lambda x: (x["image_id"], x["image"], x["mask"]))
    return pairs


def validate_pair(root: Path, pair: dict[str, str]) -> tuple[int, int, list[int]]:
    ip, mp = root / pair["image"], root / pair["mask"]
    with Image.open(ip) as im:
        im.verify()
    with Image.open(ip) as im:
        image_size = im.size
        im.convert("RGB").load()
    mask = read_mask(mp)
    if mask.shape[::-1] != image_size:
        raise ValueError(f"Размеры изображения {image_size} и маски {mask.shape[::-1]} не совпадают; маску автоматически не изменяем")
    return image_size[0], image_size[1], sorted(map(int, np.unique(mask)))


def region_masks(labels: np.ndarray, names: dict[int, str], source: str) -> dict[str, np.ndarray]:
    normalized = {int(k): str(v).lower().replace(".png", "").replace(" ", "_") for k, v in names.items()}
    def ids_for(wanted: set[str]) -> set[int]:
        found = {i for i, name in normalized.items() if name in wanted}
        if not found:
            raise ValueError(f"Не найдены ожидаемые метки {sorted(wanted)} в {source}: {normalized}")
        return found
    ids = {
        "skin": ids_for({"skin"}),
        "hair": ids_for({"hair"}),
        "eyes": ids_for({"left_eye", "right_eye", "l_eye", "r_eye"}),
    }
    return {region: np.isin(labels, tuple(values)) for region, values in ids.items()}


def save_triptych(root: Path, pair: dict[str, str], gt: np.ndarray, pred: np.ndarray, out: Path, title: str) -> None:
    image = Image.open(root / pair["image"]).convert("RGB")
    font = None
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    def colorize(masks: dict[str, np.ndarray]) -> Image.Image:
        shape = next(iter(masks.values())).shape
        rgb = np.zeros((*shape, 3), dtype=np.uint8)
        for region in REGIONS:
            rgb[masks[region]] = REGION_COLORS[region]
        return Image.fromarray(rgb, "RGB")
    panels = [image, colorize(gt), colorize(pred)]
    pad, label_h, legend_h = 14, 32, 42
    width = sum(p.width for p in panels) + pad * 4
    max_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, max_h + label_h + legend_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    x = pad
    for panel, label in zip(panels, ("Исходное фото", "Эталон LaPa", "Предсказание")):
        draw.text((x, pad), label, font=font, fill=(20, 20, 20))
        canvas.paste(panel, (x, pad + label_h))
        x += panel.width + pad
    y = pad + label_h + max_h + 8
    x = pad
    for region in REGIONS:
        draw.rectangle((x, y, x + 16, y + 16), fill=REGION_COLORS[region])
        draw.text((x + 22, y), {"skin": "Кожа", "hair": "Волосы", "eyes": "Глаза"}[region], font=font, fill=(20, 20, 20))
        x += 110
    canvas.save(out)


def load_manifest(path: Path, root: Path, split: str, limit: int, seed: int) -> list[dict[str, str]]:
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("split") != split:
            raise ValueError(f"Manifest содержит split={doc.get('split')!r}, ожидался {split!r}")
        selected = doc.get("selected", [])
        if not selected:
            raise ValueError(f"Пустой manifest: {path}")
        for pair in selected:
            if not (root / pair["image"]).is_file() or not (root / pair["mask"]).is_file():
                raise FileNotFoundError(f"Файл из manifest отсутствует: {pair}")
        return selected
    pairs = discover_pairs(root, split)
    # Validate all candidate test pairs before treating them as eligible for sampling.
    bad = []
    for pair in pairs:
        try:
            validate_pair(root, pair)
        except Exception as exc:
            bad.append({**pair, "error": f"{type(exc).__name__}: {exc}"})
    if bad:
        path.parent.mkdir(parents=True, exist_ok=True)
        diag_path = path.with_name(path.stem + "_dataset_diagnostics.json")
        diag_path.write_text(json.dumps({"split": split, "invalid_pairs": bad}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Обнаружены нечитаемые/несогласованные пары ({len(bad)}). Остановлено без замены; диагностика: {diag_path}")
    if len(pairs) < limit:
        raise ValueError(f"Корректных test-пар {len(pairs)}, требуется {limit}; список не будет урезан.")
    rng = random.Random(seed)
    selected = rng.sample(pairs, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": "LaPa", "split": split, "seed": seed, "limit": limit,
        "official_labelmap": OFFICIAL_LABELMAP_URL,
        "selected": selected,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected


def metadata_metrics(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float | None]]:
    result = {}
    for region, vals in counts.items():
        result[region] = {**vals, **counts_metrics(vals["tp"], vals["fp"], vals["fn"])}
    return result


def get_processor_size(processor) -> object:
    size = getattr(processor, "size", None)
    if size is None:
        size = getattr(processor, "crop_size", None)
    if isinstance(size, (dict, tuple, list)):
        return size
    return str(size)


def write_plots(metrics: dict, rows: list[dict[str, str]], output_dir: Path) -> list[tuple[str, str]]:
    """Create presentation-ready PNG charts from completed evaluation results."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    os.environ.setdefault("FC_CACHEDIR", str(output_dir / ".fontconfig-cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    metrics.setdefault("runtime", {})["matplotlib"] = matplotlib.__version__

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.edgecolor": "#d8dee9",
        "grid.color": "#e7ebf0",
        "axes.unicode_minus": False,
    })
    labels = ["Кожа", "Волосы", "Глаза"]
    regions = ["skin", "hair", "eyes"]
    colors = ["#40BEA0", "#DC50A0", "#FACD46"]
    figures: list[tuple[str, str]] = []

    # Dataset-level micro metrics (TP/FP/FN pooled over all 100 images).
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    x = np.arange(len(regions))
    width = 0.32
    ious = [metrics["metrics_micro_aggregated"][r]["iou"] for r in regions]
    dices = [metrics["metrics_micro_aggregated"][r]["dice"] for r in regions]
    bars_iou = ax.bar(x - width / 2, ious, width, label="IoU", color=colors, edgecolor="white", linewidth=1.2)
    bars_dice = ax.bar(x + width / 2, dices, width, label="Dice", color=colors, alpha=0.58, edgecolor="white", linewidth=1.2)
    for bars in (bars_iou, bars_dice):
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_title("Качество сегментации по выбранным областям", loc="left", weight="bold", pad=14)
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    miou3 = metrics.get("miou_3_selected_regions")
    if miou3 is not None:
        ax.axhline(miou3, color="#5577CC", linewidth=1.8, linestyle="--", label=f"Среднее IoU по трём областям: {miou3:.3f}")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    fig.savefig(plot_dir / "metrics_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(("Сравнение суммарных IoU и Dice по областям", "plots/metrics_overview.png"))

    # Distribution of per-image IoU; empty-on-both-sides records remain absent/N/A.
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    distributions = []
    counts = []
    for region in regions:
        vals = []
        for row in rows:
            value = row.get(f"{region}_iou", "")
            if value not in (None, "", "N/A"):
                vals.append(float(value))
        distributions.append(vals)
        counts.append(len(vals))
    boxes = ax.boxplot(distributions, positions=x, widths=0.48, patch_artist=True, showfliers=False, medianprops={"color": "#17212b", "linewidth": 2}, whiskerprops={"color": "#8492a6"}, capprops={"color": "#8492a6"})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.72)
        box.set_edgecolor("white")
    jitter = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(distributions, colors)):
        xs = i + jitter.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(xs, vals, s=13, alpha=0.26, color=color, edgecolors="none", zorder=3)
    ax.set_title("Разброс IoU между изображениями", loc="left", weight="bold", pad=14)
    ax.set_ylabel("IoU на одном изображении")
    ax.set_xticks(x, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(plot_dir / "per_image_iou.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(("Распределение IoU по отдельным изображениям", "plots/per_image_iou.png"))

    # Per-image module timing, excluding image file IO and report writing.
    timings = np.asarray([float(row["processing_seconds"]) for row in rows], dtype=float)
    timing = metrics["timing"]
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    ax.hist(timings, bins=16, color="#5577CC", alpha=0.82, edgecolor="white", linewidth=1.2)
    ax.axvline(timing["mean_seconds"], color="#DC50A0", linewidth=2, label=f"Среднее: {timing['mean_seconds']:.2f} с")
    ax.axvline(timing["median_seconds"], color="#40A987", linewidth=2, linestyle="--", label=f"Медиана: {timing['median_seconds']:.2f} с")
    ax.axvline(timing["p95_seconds"], color="#D99B00", linewidth=2, linestyle=":", label=f"p95: {timing['p95_seconds']:.2f} с")
    ax.set_title("Время обработки одного изображения", loc="left", weight="bold", pad=14)
    ax.set_xlabel("Секунды на изображение")
    ax.set_ylabel("Количество изображений")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(plot_dir / "processing_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(("Распределение времени сегментации", "plots/processing_time.png"))
    return figures


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def graph_markdown(figures: list[tuple[str, str]]) -> str:
    body = ["## Графики", ""]
    for caption, relpath in figures:
        body.extend((f"### {caption}", "", f"![{caption}]({relpath})", ""))
    return "\n".join(body).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/LaPa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/lapa_test_100_seed42.json")
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="Оценщик использует CPU, как существующий модуль приложения")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    parser.add_argument("--preflight-only", action="store_true", help="Проверить первые два изображения по manifest и сохранить сравнительные картинки")
    parser.add_argument("--plots-only", action="store_true", help="Построить графики из готовых metrics.json и per_image.csv без инференса")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    check_metric_functions()

    try:
        if args.plots_only:
            metrics_path = args.output_dir / "metrics.json"
            rows_path = args.output_dir / "per_image.csv"
            summary_path = args.output_dir / "summary.md"
            if not metrics_path.is_file() or not rows_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError("Для --plots-only нужны готовые metrics.json, per_image.csv и summary.md")
            plot_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            plot_rows = read_csv_rows(rows_path)
            if plot_metrics.get("dataset", {}).get("processed") != 100 or len(plot_rows) != 100:
                raise ValueError("Графики строятся только по завершённому прогону всех 100 изображений")
            figures = write_plots(plot_metrics, plot_rows, args.output_dir)
            metrics_path.write_text(json.dumps(plot_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            existing = summary_path.read_text(encoding="utf-8").split("\n## Графики", 1)[0].rstrip()
            summary_path.write_text(existing + "\n\n" + graph_markdown(figures) + "\n", encoding="utf-8")
            print(f"Графики сохранены: {args.output_dir / 'plots'}; summary.md обновлён.")
            return 0
        selected = load_manifest(args.manifest, args.dataset_root, args.split, args.limit, args.seed)
        if len(selected) != args.limit:
            raise ValueError(f"В manifest {len(selected)} изображений, но --limit={args.limit}; для manifest укажите согласованный limit")
        # Validate the frozen sample every run, and explicitly verify the first two before inference.
        pair_info = {}
        for pair in selected:
            pair_info[pair["image_id"]] = validate_pair(args.dataset_root, pair)
        print(f"Проверено пар из фиксированной выборки: {len(selected)}; первые два будут визуально сохранены.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        examples_dir = args.output_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        start_load = time.perf_counter()
        processor, model = load_face_parser()
        model.eval()
        load_seconds = time.perf_counter() - start_load
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        pred_region_ids = {
            "skin": {i for i, n in id2label.items() if n.lower().replace(".png", "") == "skin"},
            "hair": {i for i, n in id2label.items() if n.lower().replace(".png", "") == "hair"},
            "eyes": {i for i, n in id2label.items() if n.lower().replace(".png", "") in {"l_eye", "r_eye"}},
        }
        if any(not ids for ids in pred_region_ids.values()):
            raise ValueError(f"Model id2label не содержит нужные классы: {id2label}")
        if not model.training:
            pass
        first_image = Image.open(args.dataset_root / selected[0]["image"]).convert("RGB")
        # One warm-up, outside per-image timings.
        segment_face(first_image, processor, model)
        aggregate = {region: {"tp": 0, "fp": 0, "fn": 0} for region in REGIONS}
        rows = []
        per_image_scores = []
        times = []
        fixed_examples = set(pair["image_id"] for pair in selected[:2])
        run_pairs = selected[:2] if args.preflight_only else selected
        for index, pair in enumerate(run_pairs, start=1):
            stage = "read_image"
            try:
                with Image.open(args.dataset_root / pair["image"]) as im:
                    im.load()  # decode/read before starting the processing timer
                    image = im.copy()
                truth = read_mask(args.dataset_root / pair["mask"])
                if image.size != truth.shape[::-1]:
                    raise ValueError(f"Координаты не совпали: image={image.size}, mask={truth.shape[::-1]}")
                # Timing excludes image/mask file IO and output serialization.
                t0 = time.perf_counter()
                stage = "inference"
                parsed = segment_face(image, processor, model)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                pred = parsed.labels
                if pred.shape != truth.shape:
                    raise ValueError(f"Карта классов {pred.shape} не совпала с эталоном {truth.shape}")
                gt_regions = {
                    "skin": truth == 1,
                    "hair": truth == 10,
                    "eyes": np.isin(truth, (4, 5)),
                }
                pred_regions = {region: np.isin(pred, tuple(ids)) for region, ids in pred_region_ids.items()}
                row: dict[str, object] = {"image_id": pair["image_id"], "image_rel_path": pair["image"], "mask_rel_path": pair["mask"]}
                available_ious = []
                for region in REGIONS:
                    g, p = gt_regions[region], pred_regions[region]
                    tp = int(np.count_nonzero(g & p)); fp = int(np.count_nonzero(~g & p)); fn = int(np.count_nonzero(g & ~p))
                    aggregate[region]["tp"] += tp; aggregate[region]["fp"] += fp; aggregate[region]["fn"] += fn
                    metric = counts_metrics(tp, fp, fn)
                    row[f"{region}_iou"] = metric["iou"]
                    row[f"{region}_dice"] = metric["dice"]
                    if metric["iou"] is not None:
                        available_ious.append(metric["iou"])
                row["processing_seconds"] = elapsed
                rows.append(row)
                per_image_scores.append((sum(available_ious) / len(available_ious) if available_ious else None, pair, gt_regions, pred_regions))
                if pair["image_id"] in fixed_examples:
                    save_triptych(args.dataset_root, pair, gt_regions, pred_regions, examples_dir / f"fixed_{index:03d}_{pair['image_id']}.png", "fixed")
                print(f"[{index}/{len(selected)}] {pair['image_id']} {elapsed:.4f}s")
            except Exception as exc:
                diagnostic = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                    "image_id": pair.get("image_id"), "image": pair.get("image"), "mask": pair.get("mask"),
                    "processed_before_error": len(rows), "selected_count": len(selected),
                    "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                }
                diag = args.output_dir / "diagnostics.json"
                diag.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Ошибка; прогон остановлен. Диагностика: {diag}", file=sys.stderr)
                return 2
        if args.preflight_only:
            (args.output_dir / "diagnostics.json").unlink(missing_ok=True)
            print(f"Preflight готов: изображения/маски и две сравнительные картинки в {examples_dir}. Полный прогон не выполнялся.")
            return 0
        totals = metadata_metrics(aggregate)
        miou3 = sum(totals[r]["iou"] for r in REGIONS) / len(REGIONS)
        p95 = float(np.percentile(np.asarray(times), 95))
        summary_metrics = {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "config_id2label": id2label},
            "dataset": {"name": "LaPa", "split": args.split, "root": str(args.dataset_root), "seed": args.seed, "manifest": str(args.manifest), "sampled": len(selected), "processed": len(rows)},
            "class_mapping": {
                "skin": {"lapa_ids": [1], "lapa_labels": ["skin"], "model_ids": sorted(pred_region_ids["skin"]), "model_labels": [id2label[i] for i in sorted(pred_region_ids["skin"])]},
                "hair": {"lapa_ids": [10], "lapa_labels": ["hair"], "model_ids": sorted(pred_region_ids["hair"]), "model_labels": [id2label[i] for i in sorted(pred_region_ids["hair"])]},
                "eyes": {"lapa_ids": [4, 5], "lapa_labels": ["left eye", "right eye"], "model_ids": sorted(pred_region_ids["eyes"]), "model_labels": [id2label[i] for i in sorted(pred_region_ids["eyes"])]},
                "note": "Схемы разметки различаются; оцениваются только перечисленные области. Все остальные ID являются отрицательными пикселями.",
            },
            "metrics_micro_aggregated": totals, "miou_3_selected_regions": miou3,
            "timing": {"weights_load_seconds": load_seconds, "warmup_passes": 1, "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times), "p95_seconds": p95, "device": args.device, "processor_size": get_processor_size(processor), "timed_scope": "RGB input preparation, AutoImageProcessor, forward pass, logits resize, argmax and CPU class-map materialization; excludes file I/O and report writes"},
            "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": __import__("transformers").__version__, "pillow": PIL.__version__, "numpy": np.__version__},
            "evaluation": {"inference": "full original image; same stylist.segmentation.segment_face preprocessing and output resizing as app", "training_or_finetuning": False, "limitations": ["Небольшая выборка: 100 изображений.", "Схемы разметки LaPa и CelebAMask-HQ различаются; использовано указанное сопоставление областей.", "Радужки и точность определения цвета не оценивались.", "Проверка на другом датасете сама по себе не доказывает отсутствие пересечений изображений с обучением модели."], "created_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        if len(rows) != 100:
            raise RuntimeError(f"Внутренняя проверка: обработано {len(rows)} изображений, ожидалось 100; metrics не сохраняются")
        (args.output_dir / "metrics.json").write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        columns = ["image_id", "image_rel_path", "mask_rel_path", "skin_iou", "skin_dice", "hair_iou", "hair_dice", "eyes_iou", "eyes_dice", "processing_seconds"]
        with (args.output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        # Worst example is selected only by mean of available per-image region IoUs.
        valid_worst = [x for x in per_image_scores if x[0] is not None]
        if valid_worst:
            worst = min(valid_worst, key=lambda x: x[0])
            save_triptych(args.dataset_root, worst[1], worst[2], worst[3], examples_dir / f"lowest_iou_{worst[1]['image_id']}.png", "lowest IoU")
        def fmt(value): return "N/A" if value is None else f"{value:.4f}"
        md = [
            "# Оценка сегментации LaPa",
            "", f"Модель: {MODEL_ID} (ревизия `{MODEL_REVISION}`)", "Датасет: LaPa, test", "Выборка: 100 изображений, seed=42", "Обучение и дообучение не проводились.",
            "", "| Область | IoU | Dice |", "|---|---:|---:|",
        ]
        for region, ru in (("skin", "Кожа"), ("hair", "Волосы"), ("eyes", "Глаза")):
            md.append(f"| {ru} | {fmt(totals[region]['iou'])} | {fmt(totals[region]['dice'])} |")
        figures = write_plots(summary_metrics, rows, args.output_dir)
        (args.output_dir / "metrics.json").write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md += ["", f"Среднее IoU по трём выбранным областям: {miou3:.4f}", f"Среднее время сегментации: {statistics.mean(times):.4f} с (медиана {statistics.median(times):.4f} с, p95 {p95:.4f} с)", f"Устройство: {args.device}", f"Загрузка весов: {load_seconds:.3f} с", f"Размер входа processor: {get_processor_size(processor)}", "", "Полное изображение обработано через тот же `segment_face` и `AutoImageProcessor`, что использует приложение. Метрики рассчитаны по суммарным TP/FP/FN на всех 100 изображениях. LaPa skin сопоставлен только с skin модели; уши, нос и шея к коже не добавлялись. Глаза — объединение левого и правого глаза.", "", "## Ограничения", "", *[f"- {x}" for x in summary_metrics["evaluation"]["limitations"]], "", "Иллюстрации `fixed_001` и `fixed_002` соответствуют первым двум позициям сохранённого manifest. Третий пример — минимальное среднее IoU по доступным областям среди этих 100 изображений. Перед публичным использованием фотографий проверьте условия датасета: [официальная страница LaPa](https://github.com/jd-opensource/lapa-dataset).", ""]
        md.extend(("", graph_markdown(figures), ""))
        (args.output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
        (args.output_dir / "diagnostics.json").unlink(missing_ok=True)
        print(f"Готово: обработано {len(rows)} изображений. Результаты: {args.output_dir}")
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        diag = args.output_dir / "diagnostics.json"
        diag.write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Оценка остановлена: {exc}\nДиагностика: {diag}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
