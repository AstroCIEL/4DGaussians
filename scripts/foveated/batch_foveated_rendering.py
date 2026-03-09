#!/usr/bin/env python3
"""
批量处理数据集中的多个场景，对每个场景的渲染图片进行 foveated rendering 模拟。

用法示例:
    python batch_foveated_rendering.py \
        --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \
        --pattern "*/test/*/renders" \
        --tile_size 32 \
        --fov_x 90.0
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

# 导入 foveated_renderer 中的处理函数
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from foveated_renderer import _process_folder


def find_render_folders(base_dir: str, pattern: str = "*/test/*/renders") -> List[Path]:
    """
    查找所有符合模式的渲染文件夹。
    
    Args:
        base_dir: 基础目录，例如 /DISK1/home/rh_xu30/4DGaussians/output/dynerf
        pattern: 查找模式，例如 "*/test/*/renders" 或 "*/test/ours_*/renders"
    
    Returns:
        找到的所有渲染文件夹路径列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"基础目录不存在: {base_dir}")
    
    # 使用 glob 查找所有匹配的文件夹
    render_folders = list(base_path.glob(pattern))
    
    # 过滤出实际存在的目录
    render_folders = [f for f in render_folders if f.is_dir()]
    
    return sorted(render_folders)


def process_dataset(
    base_dir: str,
    pattern: str = "*/test/*/renders",
    output_suffix: str = "_foveated",
    tile_size: int = 32,
    fov_x: float = 90.0,
    foveated_enabled: bool = True,
    fixation: Optional[Tuple[float, float]] = None,
    dry_run: bool = False,
) -> None:
    """
    批量处理数据集中的所有场景。
    
    Args:
        base_dir: 基础目录
        pattern: 查找模式
        output_suffix: 输出文件夹后缀
        tile_size: tile 大小
        fov_x: 水平视场角
        foveated_enabled: 是否启用 foveated rendering
        fixation: 注视点坐标 (x, y)，归一化到 [0, 1]
        dry_run: 如果为 True，只打印将要处理的文件夹，不实际处理
    """
    if fixation is None:
        fixation = (0.5, 0.5)
    
    # 查找所有渲染文件夹
    print(f"在 {base_dir} 中查找渲染文件夹（模式: {pattern}）...")
    render_folders = find_render_folders(base_dir, pattern)
    
    if not render_folders:
        print(f"未找到匹配的渲染文件夹")
        return
    
    print(f"找到 {len(render_folders)} 个渲染文件夹:")
    for folder in render_folders:
        print(f"  - {folder}")
    
    if dry_run:
        print("\n[DRY RUN] 仅显示将要处理的文件夹，不实际处理")
        return
    
    # 处理每个文件夹
    print(f"\n开始处理...")
    for render_folder in tqdm(render_folders, desc="处理场景"):
        output_folder = render_folder.parent / (render_folder.name + output_suffix)
        
        try:
            _process_folder(
                str(render_folder),
                str(output_folder),
                tile_size=tile_size,
                fov_x=fov_x,
                foveated_enabled=foveated_enabled,
                fixation=fixation,
            )
        except Exception as e:
            print(f"\n处理 {render_folder} 时出错: {e}")
            continue
    
    print(f"\n完成！共处理 {len(render_folders)} 个场景")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量处理数据集中的多个场景，对渲染图片进行 foveated rendering 模拟。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 dynerf 数据集中的所有场景
  python batch_foveated_rendering.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --pattern "*/test/*/renders"
  
  # 只处理特定方法的渲染结果
  python batch_foveated_rendering.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --pattern "*/test/ours_*/renders"
  
  # 自定义参数
  python batch_foveated_rendering.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --tile_size 64 \\
      --fov_x 120.0 \\
      --fixation_x 0.6 \\
      --fixation_y 0.4
        """,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="基础目录路径，例如 /DISK1/home/rh_xu30/4DGaussians/output/dynerf",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/test/*/renders",
        help='查找模式（glob 模式），例如 "*/test/*/renders" 或 "*/test/ours_*/renders"',
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_foveated",
        help="输出文件夹后缀（默认: _foveated）",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=32,
        help="tile 大小（像素），需与仿真配置保持一致",
    )
    parser.add_argument(
        "--fov_x",
        type=float,
        default=90.0,
        help="水平视场角（度），影响区域划分",
    )
    parser.add_argument(
        "--disable_foveated",
        action="store_true",
        help="关闭 foveated 模拟（全部视为 fovea）",
    )
    parser.add_argument(
        "--fixation_x",
        type=float,
        default=0.5,
        help="注视点横向归一化坐标（0~1，0.5 表示画面水平中心）",
    )
    parser.add_argument(
        "--fixation_y",
        type=float,
        default=0.5,
        help="注视点纵向归一化坐标（0~1，0.5 表示画面垂直中心）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅显示将要处理的文件夹，不实际处理",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    process_dataset(
        base_dir=args.base_dir,
        pattern=args.pattern,
        output_suffix=args.output_suffix,
        tile_size=args.tile_size,
        fov_x=args.fov_x,
        foveated_enabled=not args.disable_foveated,
        fixation=(args.fixation_x, args.fixation_y),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
