# simulator/main.py
"""主程序入口：加载配置、运行仿真、输出报告"""

import argparse
import sys
import os

# 项目根目录：4DGaussians（simulator 的上级）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator.simulator import Simulator
from simulator.analyzer import Analyzer


def main():
    parser = argparse.ArgumentParser(description="4DGS ASIC Simulator")
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="统计结果输出 JSON 路径（覆盖 config 中的 output.stats_file）",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(_ROOT, "simulator", config_path)
        if os.path.isfile(candidate):
            config_path = candidate
        else:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    if not os.path.isfile(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    sim = Simulator(config_path)
    sim.start()
    analyzer = Analyzer(sim.stats, sim.config)
    analyzer.run(save_path=args.output)


if __name__ == "__main__":
    main()
