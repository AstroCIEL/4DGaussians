#!/usr/bin/env python3
"""
计算模拟了 foveated rendering 的图片和 ground truth 之间的 HSVQ 指标。

用法示例:
    # 计算单个场景的 HSVQ
    python calc_foveated_hsvq.py \
        --base_dir /DISK1/home/rh_xu30/4DGaussians/output/hypernerf/interp \
        --scene cut-lemon1 \
        --pattern "*/test/*/renders" \
        --method ours_14000 \
        --output /DISK1/home/rh_xu30/4DGaussians/output/hypernerf/interp/cut-lemon1/hsvq.json
    
    # 批量计算多个场景的 HSVQ
    python calc_foveated_hsvq.py \
        --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \
        --pattern "*/test/*/renders_foveated" \
        --output /DISK1/home/rh_xu30/4DGaussians/output/dynerf/hsvq_fov.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

# 导入必要的模块
import sys
# 添加项目根目录到 Python 路径，以便导入 hvs_loss_calc
# 脚本位置: 4DGaussians/scripts/foveated/calc_foveated_hsvq.py
# 项目根目录: 4DGaussians/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from hvs_loss_calc import HVSLoss


def read_images(foveated_dir: Path, gt_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    读取 foveated rendering 图片和对应的 ground truth 图片。
    
    Args:
        foveated_dir: foveated rendering 图片文件夹路径
        gt_dir: ground truth 图片文件夹路径
    
    Returns:
        (foveated_images, gt_images, image_names) 元组
    """
    foveated_images = []
    gt_images = []
    image_names = []
    
    if not foveated_dir.exists():
        raise FileNotFoundError(f"Foveated rendering 文件夹不存在: {foveated_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth 文件夹不存在: {gt_dir}")
    
    # 获取所有图片文件名
    foveated_files = sorted([f for f in foveated_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    for foveated_file in foveated_files:
        gt_file = gt_dir / foveated_file.name
        if not gt_file.exists():
            print(f"警告: 未找到对应的 GT 图片: {gt_file}")
            continue
        
        try:
            foveated_img = Image.open(foveated_file)
            gt_img = Image.open(gt_file)
            
            # 确保两张图片尺寸一致（以 GT 图片尺寸为准）
            if foveated_img.size != gt_img.size:
                print(f"警告: 图片 {foveated_file.name} 尺寸不匹配 (foveated: {foveated_img.size}, gt: {gt_img.size})，将 foveated 图片 resize 到 GT 尺寸")
                foveated_img = foveated_img.resize(gt_img.size, Image.BILINEAR)
            
            # 转换为 tensor 并移到 GPU
            foveated_tensor = tf.to_tensor(foveated_img).unsqueeze(0)[:, :3, :, :].cuda()
            gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
            
            foveated_images.append(foveated_tensor)
            gt_images.append(gt_tensor)
            image_names.append(foveated_file.name)
        except Exception as e:
            print(f"警告: 读取图片 {foveated_file.name} 时出错: {e}")
            continue
    
    return foveated_images, gt_images, image_names


def calculate_hsvq(
    foveated_dir: Path,
    gt_dir: Path,
    hvs_loss_calc: HVSLoss,
    gaze: Tuple[float, float] = (0.5, 0.5),
) -> Tuple[float, Dict[str, float]]:
    """
    计算 foveated rendering 图片和 ground truth 之间的 HSVQ 指标。
    
    Args:
        foveated_dir: foveated rendering 图片文件夹路径
        gt_dir: ground truth 图片文件夹路径
        hvs_loss_calc: HVSLoss 计算器实例
        gaze: 注视点坐标 (x, y)，归一化到 [0, 1]
    
    Returns:
        (平均 HSVQ 值, 每张图片的 HSVQ 值字典)
    """
    foveated_images, gt_images, image_names = read_images(foveated_dir, gt_dir)
    
    if not foveated_images:
        raise ValueError(f"未找到可用的图片对")
    
    hsvqs = []
    per_image_hsvq = {}
    
    for idx in tqdm(range(len(foveated_images)), desc="计算 HSVQ"):
        hsvq_value = hvs_loss_calc.calc_fov_loss(
            foveated_images[idx],
            gt_images[idx],
            gaze=list(gaze)
        )
        hsvq_value_item = hsvq_value.item() if isinstance(hsvq_value, torch.Tensor) else float(hsvq_value)
        hsvqs.append(hsvq_value_item)
        per_image_hsvq[image_names[idx]] = hsvq_value_item
    
    mean_hsvq = torch.tensor(hsvqs).mean().item()
    
    return mean_hsvq, per_image_hsvq


def find_foveated_folders(base_dir: str, pattern: str = "*/test/*/renders_foveated") -> List[Path]:
    """
    查找所有符合模式的 foveated rendering 文件夹。
    
    Args:
        base_dir: 基础目录
        pattern: 查找模式（glob 模式）
    
    Returns:
        找到的所有 foveated rendering 文件夹路径列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"基础目录不存在: {base_dir}")
    
    foveated_folders = list(base_path.glob(pattern))
    foveated_folders = [f for f in foveated_folders if f.is_dir()]
    
    return sorted(foveated_folders)


def process_single_scene(
    base_dir: str,
    scene: str,
    method: str,
    hvs_loss_calc: HVSLoss,
    gaze: Tuple[float, float] = (0.5, 0.5),
    output_file: Optional[str] = None,
) -> Dict:
    """
    处理单个场景的 HSVQ 计算。
    
    Args:
        base_dir: 基础目录
        scene: 场景名称
        method: 方法名称
        hvs_loss_calc: HVSLoss 计算器实例
        gaze: 注视点坐标
        output_file: 输出 JSON 文件路径（可选）
    
    Returns:
        包含结果的字典
    """
    scene_path = Path(base_dir) / scene / "test" / method
    foveated_dir = scene_path / "renders_foveated"
    gt_dir = scene_path / "gt"
    
    if not foveated_dir.exists():
        raise FileNotFoundError(f"Foveated rendering 文件夹不存在: {foveated_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth 文件夹不存在: {gt_dir}")
    
    print(f"计算场景 {scene} 方法 {method} 的 HSVQ...")
    mean_hsvq, per_image_hsvq = calculate_hsvq(foveated_dir, gt_dir, hvs_loss_calc, gaze)
    
    result = {
        "scene": scene,
        "method": method,
        "mean_hsvq": mean_hsvq,
        "per_image_hsvq": per_image_hsvq,
    }
    
    print(f"场景 {scene} 方法 {method} 的平均 HSVQ: {mean_hsvq:.7f}")
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {output_path}")
    
    return result


def process_batch(
    base_dir: str,
    hvs_loss_calc: HVSLoss,
    pattern: str = "*/test/*/renders_foveated",
    gaze: Tuple[float, float] = (0.5, 0.5),
    output_file: Optional[str] = None,
    dry_run: bool = False,
) -> Dict:
    """
    批量处理多个场景的 HSVQ 计算。
    
    Args:
        base_dir: 基础目录
        pattern: 查找模式
        hvs_loss_calc: HVSLoss 计算器实例
        gaze: 注视点坐标
        output_file: 输出 JSON 文件路径（可选）
        dry_run: 如果为 True，只打印将要处理的文件夹，不实际处理
    
    Returns:
        包含所有结果的字典
    """
    print(f"在 {base_dir} 中查找 foveated rendering 文件夹（模式: {pattern}）...")
    foveated_folders = find_foveated_folders(base_dir, pattern)
    
    if not foveated_folders:
        print(f"未找到匹配的 foveated rendering 文件夹")
        return {}
    
    print(f"找到 {len(foveated_folders)} 个 foveated rendering 文件夹:")
    for folder in foveated_folders:
        print(f"  - {folder}")
    
    if dry_run:
        print("\n[DRY RUN] 仅显示将要处理的文件夹，不实际处理")
        return {}
    
    all_results = {}
    
    print(f"\n开始计算 HSVQ...")
    for foveated_folder in tqdm(foveated_folders, desc="处理场景"):
        # 从路径中提取场景和方法信息
        # 路径格式: base_dir/scene/test/method/renders_foveated
        method_dir = foveated_folder.parent
        method = method_dir.name
        test_dir = method_dir.parent
        scene = test_dir.parent.name
        
        gt_dir = method_dir / "gt"
        
        if not gt_dir.exists():
            print(f"警告: 未找到 GT 文件夹: {gt_dir}，跳过")
            continue
        
        try:
            mean_hsvq, per_image_hsvq = calculate_hsvq(foveated_folder, gt_dir, hvs_loss_calc, gaze)
            
            if scene not in all_results:
                all_results[scene] = {}
            all_results[scene][method] = {
                "mean_hsvq": mean_hsvq,
                "per_image_hsvq": per_image_hsvq,
            }
            
            print(f"场景 {scene} 方法 {method} 的平均 HSVQ: {mean_hsvq:.7f}")
        except Exception as e:
            print(f"处理 {foveated_folder} 时出错: {e}")
            continue
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n所有结果已保存到: {output_path}")
    
    return all_results


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算模拟了 foveated rendering 的图片和 ground truth 之间的 HSVQ 指标。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算单个场景的 HSVQ
  python calc_foveated_hsvq.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --scene coffee_martini \\
      --method ours_14000
  
  # 批量计算多个场景的 HSVQ
  python calc_foveated_hsvq.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --pattern "*/test/*/renders_foveated"
  
  # 自定义注视点
  python calc_foveated_hsvq.py \\
      --base_dir /DISK1/home/rh_xu30/4DGaussians/output/dynerf \\
      --pattern "*/test/*/renders_foveated" \\
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
    
    # 单场景模式
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="场景名称（单场景模式）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ours_14000",
        help="方法名称（单场景模式）",
    )
    
    # 批量模式
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/test/*/renders",
        help='查找模式（glob 模式），用于批量处理，例如 "*/test/*/renders_foveated"',
    )
    
    # 其他参数
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
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅显示将要处理的文件夹，不实际处理（仅批量模式）",
    )
    
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    
    # 设置 CUDA 设备
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # 初始化 HVS loss 计算器
    hvs_loss_calc = HVSLoss(device="cuda")
    
    gaze = (args.fixation_x, args.fixation_y)
    
    # 判断是单场景模式还是批量模式
    if args.scene is not None and args.method is not None:
        # 单场景模式
        process_single_scene(
            base_dir=args.base_dir,
            scene=args.scene,
            method=args.method,
            hvs_loss_calc=hvs_loss_calc,
            gaze=gaze,
            output_file=args.output,
        )
    else:
        # 批量模式
        process_batch(
            base_dir=args.base_dir,
            pattern=args.pattern,
            hvs_loss_calc=hvs_loss_calc,
            gaze=gaze,
            output_file=args.output,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
