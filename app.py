"""
Gradio 演示应用 - 工业表面缺陷检测系统
运行: python app.py
"""

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.dataset import generate_synthetic_samples, DEFECT_CLASSES
from src.predict import predict_image, CLASS_NAMES_ZH

MODEL_PATH = os.environ.get("MODEL_PATH", "weights/best.pt")
DEMO_MODEL_PATH = "weights/demo_yolov8s.pt"


def load_model():
    """加载模型，优先加载训练好的权重，否则用预训练模型演示。"""
    if Path(MODEL_PATH).exists():
        return YOLO(MODEL_PATH)
    print(f"[警告] 未找到 {MODEL_PATH}，使用 YOLOv8s 预训练权重演示")
    return YOLO("yolov8s.pt")


model = load_model()


def detect_defects(image: np.ndarray, conf_threshold: float) -> tuple:
    """Gradio 回调：输入图片，返回标注图和检测结果表格。"""
    if image is None:
        return None, "请上传图片"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    annotated_bgr, detections = predict_image(model, tmp_path, conf=conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    os.unlink(tmp_path)

    if not detections:
        result_text = "✅ 未检测到缺陷"
    else:
        lines = [f"⚠️ 共检测到 **{len(detections)}** 个缺陷：\n"]
        for i, d in enumerate(detections, 1):
            lines.append(
                f"{i}. **{d['class_zh']}** — 置信度: `{d['confidence']:.3f}` "
                f"| 位置: `[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]`"
            )
        result_text = "\n".join(lines)

    return annotated_rgb, result_text


def get_class_stats() -> str:
    """返回类别说明 Markdown 文本。"""
    lines = ["| 英文名 | 中文名 | 说明 |", "|--------|--------|------|"]
    descriptions = [
        "钢铁表面网状裂纹",
        "压入异物形成的白色点状缺陷",
        "表面局部光泽不均匀斑块",
        "点状凹坑，类似麻点",
        "轧制过程中氧化皮压入表面",
        "机械损伤或摩擦产生的划痕",
    ]
    for name, zh, desc in zip(DEFECT_CLASSES, CLASS_NAMES_ZH.values(), descriptions):
        lines.append(f"| `{name}` | {zh} | {desc} |")
    return "\n".join(lines)


with gr.Blocks(
    title="工业表面缺陷检测系统",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container { max-width: 1100px !important; }",
) as demo:

    gr.Markdown(
        """
        # 🔍 工业表面缺陷检测系统
        **基于 YOLOv8 的钢铁表面缺陷实时检测** | NEU-DET 数据集 | 6类缺陷识别
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传检测图片", type="numpy", height=400)
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                label="置信度阈值",
            )
            detect_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="检测结果", type="numpy", height=400)
            result_text = gr.Markdown(label="检测详情")

    detect_btn.click(
        fn=detect_defects,
        inputs=[input_image, conf_slider],
        outputs=[output_image, result_text],
    )

    with gr.Accordion("📋 缺陷类别说明", open=False):
        gr.Markdown(get_class_stats())

    with gr.Accordion("ℹ️ 模型信息", open=False):
        gr.Markdown(
            f"""
            - **模型架构**: YOLOv8s
            - **数据集**: NEU Surface Defect Database (NEU-DET)
            - **输入尺寸**: 640×640
            - **检测类别**: {len(DEFECT_CLASSES)} 类钢铁表面缺陷
            - **权重路径**: `{MODEL_PATH}`
            """
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
