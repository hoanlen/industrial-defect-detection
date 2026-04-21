"""
训练模块 - 基于 YOLOv8 的工业缺陷检测训练脚本
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    dataset_yaml: str,
    model_size: str = "s",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "defect_v1",
    pretrained: bool = True,
):
    """
    启动 YOLOv8 训练。

    Args:
        dataset_yaml: dataset.yaml 路径
        model_size: YOLOv8 模型尺寸，可选 n/s/m/l/x
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: batch size
        project: 结果保存根目录
        name: 实验名称
        pretrained: 是否使用预训练权重
    """
    model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
    model = YOLO(model_name)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        patience=20,
        augment=True,
    )

    print(f"\n训练完成！结果保存至: {project}/{name}")
    return results


def evaluate(weights: str, dataset_yaml: str, split: str = "test"):
    """在测试集上评估模型性能。"""
    model = YOLO(weights)
    metrics = model.val(data=dataset_yaml, split=split)

    print("\n===== 评估结果 =====")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业缺陷检测训练")
    parser.add_argument("--data", type=str, required=True, help="dataset.yaml 路径")
    parser.add_argument("--model", type=str, default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="defect_v1")
    parser.add_argument("--eval-only", type=str, default=None, help="仅评估，传入 weights 路径")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.eval_only, args.data)
    else:
        train(
            dataset_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
        )
