"""
一键完整流程：数据准备 → 训练 → 评估 → 生成报告图表
运行完毕后直接截图放 README 即可上传 GitHub

用法: python scripts/run_full_pipeline.py
      python scripts/run_full_pipeline.py --use-synthetic  # 无真实数据时用合成数据演示
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pipeline(use_synthetic: bool = False):
    t_start = time.time()

    # ── 1. 数据准备 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1/4  数据准备")
    print("=" * 60)

    if use_synthetic:
        from src.dataset import generate_synthetic_samples
        DATA_YAML = "data/synthetic/dataset.yaml"
        generate_synthetic_samples("data/synthetic", n_per_class=150)
    else:
        DATA_YAML = "data/neu_det_yolo/dataset.yaml"
        if not Path(DATA_YAML).exists():
            print("[错误] 未找到真实数据集，请先运行 scripts/setup_real_dataset.py")
            print("       或加 --use-synthetic 参数使用合成演示数据")
            sys.exit(1)

    print(f"数据集: {DATA_YAML}")

    # ── 2. 训练 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2/4  模型训练（RTX 5070 Ti 预计 15-20 分钟）")
    print("=" * 60)

    from src.train import train
    t_train = time.time()
    train(
        dataset_yaml=DATA_YAML,
        model_size="s",
        epochs=100,
        imgsz=640,
        batch=32,           # 5070 Ti 12GB 可以跑 batch=32
        name="defect_v1",
    )
    train_time = (time.time() - t_train) / 60
    print(f"\n训练耗时: {train_time:.1f} 分钟")

    # ── 3. 评估 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3/4  测试集评估")
    print("=" * 60)

    weights = "runs/detect/defect_v1/weights/best.pt"
    if not Path(weights).exists():
        weights = "runs/detect/defect_v1/weights/last.pt"

    from src.evaluate import run_full_evaluation
    run_full_evaluation(weights, DATA_YAML, output_dir="results/eval")

    # ── 4. 生成演示图表 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4/4  生成演示图表（README 用）")
    print("=" * 60)

    import subprocess
    subprocess.run([sys.executable, "scripts/generate_demo_visuals.py"], check=True)

    # ── 总结 ─────────────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print("  全流程完成！")
    print("=" * 60)
    print(f"\n  总耗时: {total_time:.1f} 分钟")
    print(f"  模型权重: runs/detect/defect_v1/weights/best.pt")
    print(f"  评估报告: results/eval/")
    print(f"  演示图表: results/demo/")
    print(f"\n  现在你可以：")
    print(f"  1. 运行 Gradio 演示: python app.py")
    print(f"  2. 截图放入 README")
    print(f"  3. git add . && git commit && git push")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic", action="store_true",
                        help="使用合成数据（无真实数据集时的演示模式）")
    args = parser.parse_args()
    run_pipeline(use_synthetic=args.use_synthetic)
