import math
import os
from argparse import ArgumentParser
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_image(image: ArrayLike) -> np.ndarray:
    """
    将输入统一转换为 HxWxC 的 numpy.uint8 或 float32 图像。
    支持:
    - numpy: HxW, HxWxC
    - torch: CxHxW, HxWxC, HxW
    """
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
    else:
        img = np.asarray(image)

    # 去掉多余维度
    if img.ndim == 2:
        img = img[:, :, None]
    elif img.ndim == 3:
        # 如果是 CxHxW，则转为 HxWxC
        if img.shape[0] in (1, 3) and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 统一类型为 float32 或 uint8，保持原始动态范围
    if not np.issubdtype(img.dtype, np.floating) and not np.issubdtype(img.dtype, np.uint8):
        img = img.astype(np.float32)
    return img


def _downsample_then_upsample(tile: np.ndarray, factor: int) -> np.ndarray:
    """
    先按 factor 做缩小，再用双线性插值放大回原尺寸。
    对应：
    - factor=2: 每 2 个像素算 1 个，其余用双线性插值
    - factor=4: 每 2x2 block 只算 1 个像素，其余 3 个双线性插值
    """
    h, w = tile.shape[:2]
    if h <= 0 or w <= 0:
        return tile

    if factor == 2:
        small_h = max(1, h // factor)
        small_w = w
    elif factor == 4:
        small_h = max(1, h // 2)
        small_w = max(1, w // 2)
    else:
        raise ValueError(f"Unsupported factor: {factor}")

    # 下采样使用 AREA 近似“真实计算”结果，上采样使用 LINEAR 实现双线性插值
    small = cv2.resize(tile, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def _classify_region_with_fixation(
    tx: int,
    ty: int,
    tile_size: int,
    width: int,
    height: int,
    fov_x: float,
    foveated_enabled: bool,
    fixation_xy: Optional[Tuple[float, float]] = None,
) -> str:
    """
    带可指定注视点的区域划分，与 simulator.structures._classify_region 保持一致：
    - 默认注视点在画面中心 (width/2, height/2)
    - fixation_xy 若不为 None，则使用给定像素坐标作为注视点
    """
    if not foveated_enabled:
        return "fovea"

    px = (tx + 0.5) * tile_size
    py = (ty + 0.5) * tile_size

    if fixation_xy is None:
        cx = width / 2.0
        cy = height / 2.0
    else:
        cx, cy = fixation_xy

    dist_pixel = math.hypot(px - cx, py - cy)
    focal_length = (width / 2.0) / math.tan(math.radians(fov_x / 2.0))
    eccentricity_angle = math.degrees(math.atan(dist_pixel / focal_length))
    if eccentricity_angle <= 18.0:
        return "fovea"
    if eccentricity_angle <= 30.0:
        return "transition"
    return "periphery"


def apply_foveated_rendering(
    image: ArrayLike,
    tile_size: int = 32,
    fov_x: float = 90.0,
    foveated_enabled: bool = True,
    fixation: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    对输入整图进行 foveated rendering 模拟。

    - 中心 fovea 区域：tile 保持不变
    - transition 区域：tile 模拟 2x 降采样（factor=2）
    - periphery 区域：tile 模拟 4x 降采样（factor=4）

    Args:
        image: 输入图像（正常渲染结果），支持 numpy 或 torch。
        tile_size: tile 边长，像素。
        fov_x: 水平视场角，度。
        foveated_enabled: 若为 False，则整幅图视为 fovea，不做降采样。
        fixation: 注视点归一化坐标 (fx, fy)，取值大致在 [0, 1]。
                  fx=0.5, fy=0.5 表示画面中心（默认）。

    Returns:
        numpy.ndarray: 模拟 foveated rendering 后的 HxWxC 图像。
    """
    img_np = _to_numpy_image(image)
    h, w = img_np.shape[:2]

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    # 只在对齐到 tile 的有效区域内做处理
    eff_w = (w // tile_size) * tile_size
    eff_h = (h // tile_size) * tile_size
    if eff_w <= 0 or eff_h <= 0:
        return img_np.copy()

    # 计算注视点像素坐标（在有效区域范围内）
    if fixation is not None:
        fx = float(fixation[0]) * eff_w
        fy = float(fixation[1]) * eff_h
        fixation_xy: Optional[Tuple[float, float]] = (fx, fy)
    else:
        fixation_xy = None  # 使用画面中心

    out = img_np.copy()
    ntx = eff_w // tile_size
    nty = eff_h // tile_size

    for ty in range(nty):
        for tx in range(ntx):
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            region = _classify_region_with_fixation(
                tx=tx,
                ty=ty,
                tile_size=tile_size,
                width=eff_w,
                height=eff_h,
                fov_x=fov_x,
                foveated_enabled=foveated_enabled,
                fixation_xy=fixation_xy,
            )

            tile = img_np[y0:y1, x0:x1, ...]

            if region == "fovea" or not foveated_enabled:
                processed = tile
            elif region == "transition":
                processed = _downsample_then_upsample(tile, factor=2)
            else:
                processed = _downsample_then_upsample(tile, factor=4)

            out[y0:y1, x0:x1, ...] = processed

    return out


__all__ = ["apply_foveated_rendering"]


def _build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Simulate foveated rendering on a single image.")
    parser.add_argument("input", type=str, help="输入图片路径")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（默认在文件名后加 _foveated）",
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
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取输入图片: {args.input}")

    fixation = (args.fixation_x, args.fixation_y)
    foveated = apply_foveated_rendering(
        img,
        tile_size=args.tile_size,
        fov_x=args.fov_x,
        foveated_enabled=not args.disable_foveated,
        fixation=fixation,
    )

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_foveated" + (ext if ext else ".png")
    else:
        out_path = args.output

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, foveated)
    print(f"保存 foveated 渲染结果到: {out_path}")


if __name__ == "__main__":
    main()
