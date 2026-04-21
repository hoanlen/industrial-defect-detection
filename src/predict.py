"""
推理模块 - 单张图片 / 批量目录推理
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_COLORS = {
    "crazing":         (0, 255, 0),
    "inclusion":       (255, 0, 0),
    "patches":         (0, 0, 255),
    "pitted_surface":  (255, 255, 0),
    "rolled-in_scale": (255, 0, 255),
    "scratches":       (0, 255, 255),
}

CLASS_NAMES_ZH = {
    "crazing":         "龟裂",
    "inclusion":       "夹杂物",
    "patches":         "斑点",
    "pitted_surface":  "麻面",
    "rolled-in_scale": "压入氧化皮",
    "scratches":       "划痕",
}


def predict_image(model: YOLO, image_path: str, conf: float = 0.25) -> tuple[np.ndarray, list]:
    """
    对单张图片进行推理。

    Returns:
        annotated_img: 标注后的图像（BGR）
        detections: 检测结果列表，每项为 dict
    """
    results = model.predict(image_path, conf=conf, verbose=False)
    result = results[0]

    img = cv2.imread(image_path)
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES_ZH.get(cls_name, cls_name)} {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        detections.append({
            "class": cls_name,
            "class_zh": CLASS_NAMES_ZH.get(cls_name, cls_name),
            "confidence": conf_score,
            "bbox": [x1, y1, x2, y2],
        })

    return img, detections


def batch_predict(weights: str, input_dir: str, output_dir: str, conf: float = 0.25):
    """批量推理目录下所有图片。"""
    model = YOLO(weights)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"共找到 {len(image_files)} 张图片，开始推理...")

    summary = []
    for img_file in image_files:
        annotated, detections = predict_image(model, str(img_file), conf)
        out_file = output_path / img_file.name
        cv2.imwrite(str(out_file), annotated)

        for det in detections:
            summary.append({"file": img_file.name, **det})

    import pandas as pd
    df = pd.DataFrame(summary)
    df.to_csv(output_path / "detection_results.csv", index=False, encoding="utf-8-sig")
    print(f"推理完成！结果保存至 {output_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业缺陷检测推理")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="图片路径或目录")
    parser.add_argument("--output", type=str, default="results/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source_path = Path(args.source)
    if source_path.is_dir():
        batch_predict(args.weights, args.source, args.output, args.conf)
    else:
        model = YOLO(args.weights)
        img, dets = predict_image(model, args.source, args.conf)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        out = Path(args.output) / source_path.name
        cv2.imwrite(str(out), img)
        print(f"检测到 {len(dets)} 个缺陷：")
        for d in dets:
            print(f"  [{d['class_zh']}] 置信度: {d['confidence']:.3f}  位置: {d['bbox']}")
