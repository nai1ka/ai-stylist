"""Presentation figures from saved evaluation outputs; no inference or training."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stylist-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = "#23323F"
MUTED = "#627382"
BG = "#F5F3EE"
GRID = "#E5E7E7"
COLORS = ["#BC7147", "#85A76A", "#809EBD", "#575B86"]
NAMES = ["Осень", "Весна", "Лето", "Зима"]
CLASSES = ["autumn", "spring", "summer", "winter"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "figure.facecolor": BG,
        "axes.facecolor": "white", "savefig.facecolor": BG,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def title(ax, heading, subtitle):
    ax.set_title(heading, loc="left", fontsize=15, weight="bold", pad=34)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)


def performance(ax, m):
    title(ax, "01  /  Общее качество", "Accuracy на test • базовый класс выбран по train")
    values = [m["majority_baseline_accuracy"], m["accuracy"]]
    bars = ax.barh([1, 0], values, height=.48, color=["#CDD3D6", COLORS[3]])
    ax.set_yticks([1, 0], ["Всегда «Осень»", "EfficientNetV2-S"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-.7, 1.7)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=.7)
    for bar, value in zip(bars, values):
        ax.text(value + .025, bar.get_y() + bar.get_height()/2,
                f"{value:.2%}".replace('.', ','), va="center", fontsize=14, weight="bold")
    gain = (values[1] - values[0]) * 100
    ax.text(.99, .90, f"+{gain:.2f} п.п.".replace('.', ','), ha="right",
            transform=ax.transAxes, color=COLORS[3], weight="bold", fontsize=12)


def confusion(ax, m):
    title(ax, "02  /  Какие сезоны путает модель", "В ячейке: число фото и доля от истинного сезона")
    counts = np.array(m["confusion_matrix"])
    ratios = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seasonal", ["#F3F4F8", "#B6BED6", "#454D78"])
    ax.imshow(ratios, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), NAMES)
    ax.set_yticks(range(4), NAMES)
    ax.set_xlabel("Предсказанный сезон", labelpad=9)
    ax.set_ylabel("Истинный сезон", labelpad=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{counts[i,j]}\n{ratios[i,j]:.1%}".replace('.', ','),
                    ha="center", va="center", color="white" if ratios[i,j] > .48 else INK,
                    fontsize=11, weight="bold" if i == j else "normal", linespacing=1.5)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)


def class_metrics(ax, m):
    title(ax, "03  /  Качество каждого сезона", "Precision — точность • Recall — полнота • F1 — баланс")
    x = np.arange(4)
    for index, (key, label, color) in enumerate([
        ("precision", "Precision", "#B2C3C6"),
        ("recall", "Recall", "#667F98"),
        ("f1-score", "F1", "#3E496D"),
    ]):
        values = [m["per_class"][k][key] for k in CLASSES]
        positions = x + (index-1)*.24
        ax.bar(positions, values, width=.21, color=color, label=label)
        for pos, value in zip(positions, values):
            ax.text(pos, value+.018, f"{100*value:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=9)


def distribution(ax, m):
    title(ax, "04  /  Распределение сезонов", "Контур — истинные метки • заливка — предсказания")
    x = np.arange(4)
    actual = np.array(m["confusion_matrix"]).sum(axis=1)
    predicted = [m["predicted_distribution"][k] for k in CLASSES]
    ax.bar(x-.18, actual, width=.32, edgecolor=COLORS, facecolor="white", linewidth=2)
    ax.bar(x+.18, predicted, width=.32, color=COLORS)
    for pos, a, p in zip(x, actual, predicted):
        ax.text(pos-.18, a+9, str(a), ha="center", fontsize=11)
        ax.text(pos+.18, p+9, str(p), ha="center", fontsize=11, weight="bold")
    ax.set_xticks(x, NAMES)
    ax.set_ylabel("Количество фото")
    ax.set_ylim(0, max(max(actual), max(predicted))*1.2)
    ax.grid(axis="y", color=GRID, linewidth=.7)


def save(fig, output, name):
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(output / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "evaluation/seasonal")
    args = parser.parse_args()
    m = json.loads((args.results / "metrics.json").read_text())
    if m["classes"] != CLASSES:
        raise ValueError("Unexpected class order")
    # Validate that plots reflect saved predictions, not model-card numbers.
    matrix = np.zeros((4,4), dtype=int)
    with (args.results / "predictions.csv").open() as file:
        for row in csv.DictReader(file):
            matrix[CLASSES.index(row['true_season']), CLASSES.index(row['predicted_season'])] += 1
    if matrix.tolist() != m["confusion_matrix"] or int(matrix.sum()) != m["test_count"]:
        raise ValueError("Predictions and metrics disagree")
    if not np.isclose(np.trace(matrix)/matrix.sum(), m['accuracy']):
        raise ValueError("Accuracy and predictions disagree")
    output = args.results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    fig.subplots_adjust(left=.09, right=.96, top=.77, bottom=.13, hspace=.62, wspace=.32)
    fig.text(.055, .952, "AI-STYLIST  /  ОЦЕНКА МОДЕЛИ", color=MUTED, fontsize=11, weight="bold")
    fig.text(.055, .903, "Четыре сезона: результаты классификатора", fontsize=25, weight="bold")
    fig.text(.055, .855,
             f"{m['test_count']} фото в test     /     Accuracy  {m['accuracy']:.2%}     /     Macro-F1  {m['macro_f1']:.2%}".replace('.', ','),
             fontsize=15, color=COLORS[3])
    for ax, draw in zip(axes.flat, [performance, confusion, class_metrics, distribution]):
        draw(ax, m)
    fig.text(.055, .055, "Готовый классификатор • тестовая часть автора • собственного обучения не проводилось", fontsize=10, color=MUTED)
    fig.text(.055, .032, "Порядок классов автор восстановил по оценочной выборке. Воспроизведение эксперимента, не независимая проверка.", fontsize=9, color=MUTED)
    save(fig, output, "seasonal_overview")
    for name, draw in [("accuracy_baseline",performance), ("confusion_matrix",confusion),
                       ("per_season_metrics",class_metrics), ("class_distribution",distribution)]:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.subplots_adjust(left=.19, right=.94, top=.80, bottom=.19)
        draw(ax, m)
        fig.text(.08, .065, f"AI-Stylist • test: {m['test_count']} фото • готовая модель, без собственного обучения", fontsize=10, color=MUTED)
        save(fig, output, name)
    print(f"Saved 5 figures in PNG, SVG, PDF: {output}")


if __name__ == "__main__":
    main()
