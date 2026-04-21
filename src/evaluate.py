"""
评估模块 - 生成详细的评估报告和可视化图表
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
DEFECT_CLASSES_ZH = ["龟裂", "夹杂物", "斑点", "麻面", "压入氧化皮", "划痕"]


def run_full_evaluation(weights: str, dataset_yaml: str, output_dir: str = "results/eval"):
    """
    完整评估流程，输出：
    - per-class mAP 表格
    - 混淆矩阵热力图
    - PR 曲线对比图
    - 置信度分布图
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    metrics = model.val(data=dataset_yaml, split="test", plots=True,
                        save_dir=str(output_path))

    _plot_per_class_map(metrics, output_path)
    _plot_confidence_distribution(model, dataset_yaml, output_path)

    print(f"\n评估报告已保存至: {output_dir}")
    return metrics


def _plot_per_class_map(metrics, output_path: Path):
    """绘制各类别 mAP50 对比柱状图。"""
    names = DEFECT_CLASSES_ZH
    maps = metrics.box.maps if hasattr(metrics.box, "maps") else [0] * len(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, maps, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mAP50", fontsize=12)
    ax.set_title("各缺陷类别 mAP50 对比", fontsize=14, fontweight="bold")
    ax.axhline(y=np.mean(maps), color="red", linestyle="--", alpha=0.7,
               label=f"均值: {np.mean(maps):.3f}")
    ax.legend()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig(output_path / "per_class_map.png", dpi=150)
    plt.close()
    print("已生成: per_class_map.png")


def _plot_confidence_distribution(model: YOLO, dataset_yaml: str, output_path: Path):
    """绘制各类别预测置信度分布。"""
    import yaml
    with open(dataset_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    test_img_dir = Path(cfg["path"]) / cfg.get("test", "images/test")
    if not test_img_dir.exists():
        print("测试集目录不存在，跳过置信度分布图")
        return

    results = model.predict(str(test_img_dir), conf=0.1, verbose=False, stream=True)

    class_confs = {c: [] for c in DEFECT_CLASSES}
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in class_confs:
                class_confs[cls_name].append(float(box.conf[0]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (cls_name, cls_zh) in zip(axes.flat, zip(DEFECT_CLASSES, DEFECT_CLASSES_ZH)):
        confs = class_confs[cls_name]
        if confs:
            ax.hist(confs, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
            ax.axvline(np.mean(confs), color="red", linestyle="--",
                       label=f"均值={np.mean(confs):.2f}")
            ax.legend(fontsize=8)
        ax.set_title(cls_zh, fontsize=11)
        ax.set_xlabel("置信度")
        ax.set_ylabel("频次")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.suptitle("各类别预测置信度分布", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "confidence_distribution.png", dpi=150)
    plt.close()
    print("已生成: confidence_distribution.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/eval")
    args = parser.parse_args()
    run_full_evaluation(args.weights, args.data, args.output)
