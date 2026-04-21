"""
真实 NEU-DET 数据集准备脚本
将下载的数据集转换为 YOLOv8 训练格式

数据集实际结构（Kaggle 版本）:
  data/NEU-DET/
  ├── train/
  │   ├── images/{class_name}/{class_name}_{num}.jpg
  │   └── annotations/{class_name}_{num}.xml
  └── validation/
      ├── images/{class_name}/{class_name}_{num}.jpg
      └── annotations/{class_name}_{num}.xml

用法: python scripts/setup_real_dataset.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR    = Path("data/NEU-DET")
OUTPUT_DIR = Path("data/neu_det_yolo")

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
CLASS2ID = {cls: i for i, cls in enumerate(DEFECT_CLASSES)}

IMG_W, IMG_H = 200, 200  # NEU-DET 固定 200x200


def voc_xml_to_yolo(xml_path: Path) -> list[str]:
    """将单个 Pascal VOC XML 转换为 YOLO 格式行列表。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 尝试从 XML 读取尺寸，否则使用默认值
    size = root.find("size")
    w = int(size.findtext("width", default=str(IMG_W))) if size else IMG_W
    h = int(size.findtext("height", default=str(IMG_H))) if size else IMG_H
    if w == 0: w = IMG_W
    if h == 0: h = IMG_H

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name", "").strip()
        if cls_name not in CLASS2ID:
            continue
        cls_id = CLASS2ID[cls_name]
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))

        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def process_split(split_name: str, yolo_split: str):
    """处理 train 或 validation split，复制图片并生成 YOLO 标签。"""
    src_img_root = RAW_DIR / split_name / "images"
    src_ann_dir  = RAW_DIR / split_name / "annotations"

    dst_img_dir = OUTPUT_DIR / "images" / yolo_split
    dst_lbl_dir = OUTPUT_DIR / "labels" / yolo_split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    total, skipped = 0, 0
    for cls_name in DEFECT_CLASSES:
        cls_img_dir = src_img_root / cls_name
        if not cls_img_dir.exists():
            print(f"  [警告] 未找到目录: {cls_img_dir}")
            continue

        imgs = sorted(cls_img_dir.glob("*.jpg")) + sorted(cls_img_dir.glob("*.png"))
        for img_path in imgs:
            xml_path = src_ann_dir / img_path.with_suffix(".xml").name
            if not xml_path.exists():
                skipped += 1
                continue

            yolo_lines = voc_xml_to_yolo(xml_path)
            if not yolo_lines:
                skipped += 1
                continue

            # 复制图片
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            # 写标签
            lbl_path = dst_lbl_dir / img_path.with_suffix(".txt").name
            lbl_path.write_text("\n".join(yolo_lines))
            total += 1

    print(f"  {yolo_split:<8}: {total} 张图片转换完成" +
          (f"，{skipped} 张跳过（无标注）" if skipped else ""))
    return total


def write_dataset_yaml():
    """生成 YOLOv8 dataset.yaml。"""
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(DEFECT_CLASSES)}
names: {DEFECT_CLASSES}
"""
    yaml_path.write_text(content)
    print(f"\n  已生成: {yaml_path}")


if __name__ == "__main__":
    print("=" * 58)
    print("  NEU-DET → YOLO 格式转换")
    print("=" * 58)

    if not RAW_DIR.exists():
        print(f"\n[错误] 未找到数据集目录: {RAW_DIR.resolve()}")
        print("请先解压数据集到 data/NEU-DET/")
        exit(1)

    print(f"\n来源: {RAW_DIR.resolve()}")
    print(f"输出: {OUTPUT_DIR.resolve()}\n")

    n_train = process_split("train",      "train")
    n_val   = process_split("validation", "val")

    # 从验证集切 10% 作为 test（可选）
    val_imgs = sorted((OUTPUT_DIR / "images" / "val").glob("*.jpg"))
    test_n = max(1, len(val_imgs) // 5)
    dst_test_img = OUTPUT_DIR / "images" / "test"
    dst_test_lbl = OUTPUT_DIR / "labels" / "test"
    dst_test_img.mkdir(parents=True, exist_ok=True)
    dst_test_lbl.mkdir(parents=True, exist_ok=True)

    for img in val_imgs[:test_n]:
        shutil.move(str(img), dst_test_img / img.name)
        lbl = OUTPUT_DIR / "labels" / "val" / img.with_suffix(".txt").name
        if lbl.exists():
            shutil.move(str(lbl), dst_test_lbl / lbl.name)
    print(f"  test    : {test_n} 张（从 val 切分）")

    write_dataset_yaml()

    print("\n" + "=" * 58)
    print("  转换完成！")
    print("=" * 58)
    print(f"\n  训练集: {n_train} 张")
    print(f"  验证集: {n_val - test_n} 张")
    print(f"  测试集: {test_n} 张")
    print(f"\n  下一步，开始训练：")
    print(f"  python src/train.py --data data/neu_det_yolo/dataset.yaml --model s --epochs 100 --batch 32")
