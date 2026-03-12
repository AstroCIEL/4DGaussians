import argparse
import os
import sys

# 允许从项目根运行：python -m simulator.main --config simulator/configs/default.yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator.simulator import run_simulator


def parse_args():
    parser = argparse.ArgumentParser(description="4DGS ASIC simulator (simpy)")
    parser.add_argument(
        "--config",
        type=str,
        default="simulator/configs/neo.yaml",
        help="配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = run_simulator(args.config)
    print(f"[main] done. total_cycles={stats.total_cycles:.2f}")


if __name__ == "__main__":
    main()
