"""
数据集准备模块 - 支持 NEU-DET 钢铁表面缺陷数据集格式
数据集下载: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
或使用 Roboflow 上的公开工业缺陷数据集
"""

import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


DEFECT_CLASSES = [
    "crazing",      # 龟裂
    "inclusion",    # 夹杂物
    "patches",      # 斑点
    "pitted_surface",  # 麻面
    "rolled-in_scale", # 压入氧化皮
    "scratches",    # 划痕
]


def prepare_neu_det_dataset(raw_dir: str, output_dir: str, split_ratio=(0.7, 0.2, 0.1)):
    """
    将 NEU-DET 数据集转换为 YOLO 格式。

    Args:
        raw_dir: 原始数据集根目录（包含 IMAGES/ 和 ANNOTATIONS/ 子目录）
        output_dir: 输出目录
        split_ratio: (train, val, test) 比例
    """
    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    raw_path = Path(raw_dir)
    image_dir = raw_path / "IMAGES"
    ann_dir = raw_path / "ANNOTATIONS"

    all_samples = []
    for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
        images = list((image_dir / cls_name).glob("*.jpg")) if (image_dir / cls_name).exists() else []
        for img_path in images:
            ann_path = ann_dir / cls_name / img_path.with_suffix(".xml").name
            if ann_path.exists():
                all_samples.append((img_path, ann_path, cls_idx))

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    for split_name, samples in splits.items():
        for img_path, ann_path, cls_idx in tqdm(samples, desc=f"处理 {split_name}"):
            shutil.copy(img_path, output_path / "images" / split_name / img_path.name)
            yolo_label = _parse_xml_to_yolo(ann_path, cls_idx)
            label_file = output_path / "labels" / split_name / img_path.with_suffix(".txt").name
            with open(label_file, "w") as f:
                f.write(yolo_label)

    _write_yaml(output_path)
    print(f"数据集准备完成，共 {n} 张图片 -> {output_dir}")


def _parse_xml_to_yolo(xml_path: Path, cls_idx: int) -> str:
    """将 Pascal VOC XML 标注转为 YOLO 格式（归一化中心坐标）。"""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(lines)


def _write_yaml(output_path: Path):
    """生成 YOLO 训练所需的 dataset.yaml 文件。"""
    config = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(DEFECT_CLASSES),
        "names": DEFECT_CLASSES,
    }
    with open(output_path / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def generate_synthetic_samples(output_dir: str, n_per_class: int = 50):
    """
    生成合成演示数据（无需下载真实数据集即可运行演示）。
    使用简单的纹理生成模拟缺陷外观。
    """
    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = {"train": 0.7, "val": 0.2, "test": 0.1}
    sample_id = 0

    for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
        for i in range(n_per_class):
            img = _generate_defect_image(cls_idx)
            label = _random_yolo_label(cls_idx)

            split_name = random.choices(list(splits.keys()), weights=list(splits.values()))[0]
            fname = f"{cls_name}_{i:04d}.jpg"

            cv2.imwrite(str(output_path / "images" / split_name / fname), img)
            with open(output_path / "labels" / split_name / fname.replace(".jpg", ".txt"), "w") as f:
                f.write(label)
            sample_id += 1

    _write_yaml(output_path)
    print(f"合成演示数据生成完成：{sample_id} 张图片 -> {output_dir}")


def _generate_defect_image(cls_idx: int) -> np.ndarray:
    """生成模拟缺陷图像（200x200 灰度图）。"""
    base = np.random.randint(160, 200, (200, 200), dtype=np.uint8)
    noise = np.random.normal(0, 10, (200, 200)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    defect_funcs = [
        lambda m: cv2.line(m, (50, 50), (150, 150), 40, 2),
        lambda m: cv2.rectangle(m, (70, 70), (130, 130), 30, -1),
        lambda m: cv2.circle(m, (100, 100), 30, 20, -1),
        lambda m: _draw_pits(m),
        lambda m: cv2.ellipse(m, (100, 100), (40, 20), 45, 0, 360, 35, -1),
        lambda m: cv2.line(m, (40, 100), (160, 100), 25, 3),
    ]
    defect_funcs[cls_idx % len(defect_funcs)](img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _draw_pits(img: np.ndarray):
    for _ in range(10):
        cx, cy = random.randint(30, 170), random.randint(30, 170)
        cv2.circle(img, (cx, cy), random.randint(3, 8), 25, -1)


def _random_yolo_label(cls_idx: int) -> str:
    cx = random.uniform(0.3, 0.7)
    cy = random.uniform(0.3, 0.7)
    bw = random.uniform(0.2, 0.5)
    bh = random.uniform(0.2, 0.5)
    return f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
