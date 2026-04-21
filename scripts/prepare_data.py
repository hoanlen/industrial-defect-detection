"""
快速启动脚本 - 生成合成数据并开始训练演示
用法: python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import generate_synthetic_samples

if __name__ == "__main__":
    print("正在生成合成演示数据...")
    generate_synthetic_samples(
        output_dir="data/synthetic",
        n_per_class=100,
    )
    print("\n数据准备完成！")
    print("接下来运行训练：")
    print("  python src/train.py --data data/synthetic/dataset.yaml --epochs 30 --model n")
