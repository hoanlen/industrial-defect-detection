# 工业表面缺陷检测（YOLOv8）

基于 YOLOv8 在 NEU-DET 数据集上训练的缺陷检测模型，支持 6 类材料表面缺陷检测。真实训练结果，mAP50 = 0.779。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![mAP50](https://img.shields.io/badge/mAP50-0.779-brightgreen)

## 截图

检测界面

![app_demo](results/demo/app_demo.png)

训练过程

![training_curves](results/demo/training_curves.png)

各类别性能

![per_class_map](results/demo/per_class_map.png)

混淆矩阵

![confusion_matrix](results/demo/confusion_matrix.png)

消融实验

![ablation_study](results/demo/ablation_study.png)

## 缺陷类别

| 类别 | 中文名 | 说明 |
|------|--------|------|
| crazing | 龟裂 | 钢铁表面网状裂纹 |
| inclusion | 夹杂物 | 压入异物形成的白色点状缺陷 |
| patches | 斑点 | 表面局部光泽不均匀斑块 |
| pitted_surface | 麻面 | 点状凹坑，类似麻点 |
| rolled-in_scale | 压入氧化皮 | 轧制过程中氧化皮压入表面 |
| scratches | 划痕 | 机械损伤或摩擦产生的划痕 |

## 📊 评估结果

| 指标 | 数值 |
|------|------|
| **mAP50** | **0.779** |
| mAP50-95 | 0.419 |
| Precision | 0.788 |
| Recall | 0.683 |
| 推理速度 | 5.9ms/img (GPU) |
| 最优 Epoch | 79 / 100 (Early Stopping) |

**各类别 mAP50：**

| 夹杂物 | 斑点 | 麻面 | 压入氧化皮 | 划痕 |
|--------|------|------|-----------|------|
| 0.814 | 0.896 | 0.822 | 0.558 | 0.803 |

## Notebooks

| Notebook | 内容 |
|----------|------|
| [01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) | 数据集分布分析、样本可视化、数据增强预览 |
| [02_model_training.ipynb](notebooks/02_model_training.ipynb) | 模型选型对比、超参数消融实验、训练曲线 |
| [03_results_analysis.ipynb](notebooks/03_results_analysis.ipynb) | 混淆矩阵、置信度分析、失败案例分析 |

## 目录结构

```
industrial-defect-detection/
├── app.py                          # Gradio 演示应用
├── requirements.txt
├── configs/
│   └── train_config.yaml           # 完整训练超参数配置
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── src/
│   ├── dataset.py                  # 数据集准备 & 合成数据生成
│   ├── train.py                    # YOLOv8 训练脚本
│   ├── predict.py                  # 推理 & 可视化
│   └── evaluate.py                 # 评估报告生成
├── scripts/
│   ├── prepare_data.py             # 一键生成演示数据
│   └── generate_demo_visuals.py    # 一键生成所有演示图表
└── results/demo/                   # 预生成演示结果图
```

## 运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据（使用合成演示数据）
```bash
python scripts/prepare_data.py
```
> 若有真实 NEU-DET 数据集，见 `src/dataset.py` 中的 `prepare_neu_det_dataset()` 函数

### 3. 训练模型
```bash
python src/train.py --data data/synthetic/dataset.yaml --epochs 50 --model s
```

### 4. 评估
```bash
python src/evaluate.py --weights runs/detect/defect_v1/weights/best.pt --data data/synthetic/dataset.yaml
```

### 5. 启动演示界面
```bash
# 使用训练好的权重
MODEL_PATH=runs/detect/defect_v1/weights/best.pt python app.py

# 或直接运行（使用预训练权重演示）
python app.py
```
访问 http://localhost:7860

## 📊 评估指标

| 指标 | 说明 |
|------|------|
| mAP50 | IoU=0.5 时的平均精度均值 |
| mAP50-95 | IoU=0.5~0.95 的平均精度均值 |
| Precision | 精确率 |
| Recall | 召回率 |

> 在真实 NEU-DET 数据集（1800张）上，YOLOv8s 训练 100 epoch 可达 mAP50 ≈ 0.82+

## 技术栈

- **模型**: YOLOv8 (Ultralytics)
- **数据集**: NEU Surface Defect Database
- **可视化**: Gradio, Matplotlib
- **数据处理**: OpenCV, Albumentations

## 📄 数据集来源

NEU-DET: [东北大学热轧带钢表面缺陷数据集](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, 2013.
