"""
从算法侧加载真实 workload：参考 render.py 与 Scene/GaussianModel，将高斯投影到屏幕并按 tile 聚合。
若失败则返回 None，调用方应回退到合成 workload。
"""

import os
import sys
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator.structures import (
    GaussianAttr,
    TileWorkload,
    WorkloadFrame,
    _classify_region,
)
from simulator.generate_labels import generate_motion_labels


def _extract_gaussian_attrs(gaussians, gaussian_labels: Optional[np.ndarray] = None) -> Dict[int, GaussianAttr]:
    """从 GaussianModel 提取关键属性（位置/尺度/旋转/不透明度/SH/标签）。"""
    attrs: Dict[int, GaussianAttr] = {}
    xyz = gaussians._xyz.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy() if hasattr(gaussians, "_scaling") else np.zeros_like(xyz)
    rotation = gaussians._rotation.detach().cpu().numpy() if hasattr(gaussians, "_rotation") else np.zeros((xyz.shape[0], 4))
    opacity = gaussians._opacity.detach().cpu().numpy() if hasattr(gaussians, "_opacity") else np.ones((xyz.shape[0], 1))
    sh_dc = gaussians._features_dc.detach().cpu().numpy() if hasattr(gaussians, "_features_dc") else None
    for i in range(xyz.shape[0]):
        sh_values = sh_dc[i].flatten().tolist() if sh_dc is not None else None
        label = None
        if gaussian_labels is not None and i < len(gaussian_labels):
            lb = int(gaussian_labels[i])
            if lb in (0, 1, 2):
                label = lb
        attrs[i] = GaussianAttr(
            idx=i,
            position=tuple(map(float, xyz[i])),
            scale=tuple(map(float, scaling[i])) if len(scaling.shape) > 1 else (0.0, 0.0, 0.0),
            rotation=tuple(map(float, rotation[i])) if len(rotation.shape) > 1 else (0.0, 0.0, 0.0, 0.0),
            opacity=float(opacity[i][0]) if opacity.ndim > 1 else float(opacity[i]),
            sh=sh_values,
            label=label,
        )
    return attrs


def _project_3d_to_2d(means3D: torch.Tensor, projmatrix: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    将 3D 点投影到 2D 屏幕空间。
    参考 CUDA 代码中的实现：forward.cu 的 preprocessCUDA 函数。
    
    Args:
        means3D: (N, 3) 3D 点坐标
        projmatrix: (4, 4) 投影矩阵
        width: 图像宽度
        height: 图像高度
    
    Returns:
        means2D: (N, 3) 2D 屏幕空间坐标（像素坐标），第三列为 0，保持与原始 viewspace_points 形状一致
    """
    device = means3D.device
    N = means3D.shape[0]
    
    # 转换为齐次坐标
    ones = torch.ones((N, 1), device=device, dtype=means3D.dtype)
    p_orig_h = torch.cat([means3D, ones], dim=-1)  # (N, 4)
    
    # 投影到齐次屏幕空间
    p_hom = p_orig_h @ projmatrix.T  # (N, 4)
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)  # (N, 1)
    p_proj = p_hom[:, :3] * p_w  # (N, 3) - NDC 坐标
    
    # NDC 到像素坐标转换 (参考 auxiliary.h 的 ndc2Pix)
    # ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5
    x_pix = ((p_proj[:, 0] + 1.0) * width - 1.0) * 0.5
    y_pix = ((p_proj[:, 1] + 1.0) * height - 1.0) * 0.5
    
    # 返回 (N, 3) 形状，第三列为 0，与原始 viewspace_points 形状一致
    z_zero = torch.zeros((N, 1), device=device, dtype=means3D.dtype)
    return torch.cat([x_pix.unsqueeze(1), y_pix.unsqueeze(1), z_zero], dim=1)  # (N, 3)


def _compute_tile_cover(
    means2d,
    radii,
    effective_width: int,
    effective_height: int,
    tile_size: int,
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[Tuple[int, int]]], int, int]:
    """
    计算每个 tile 覆盖的高斯列表。
    返回 (tile -> [gaussian_ids], gaussian -> [(tx,ty)], num_tiles_x, num_tiles_y)。
    """
    if hasattr(means2d, "cpu"):
        means2d = means2d.detach().cpu().numpy()
    if hasattr(radii, "cpu"):
        radii = radii.detach().cpu().numpy()
    if means2d.shape[1] == 3:
        means2d = means2d[:, :2]
    visible_mask = radii > 0
    visible_means2d = means2d[visible_mask]
    visible_radii = radii[visible_mask]
    visible_indices = np.nonzero(visible_mask)[0]

    num_tiles_x = effective_width // tile_size
    num_tiles_y = effective_height // tile_size
    if num_tiles_x <= 0 or num_tiles_y <= 0:
        return defaultdict(list), defaultdict(list), 0, 0
    tile_to_gaussians: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    gaussian_to_tiles: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for i, gid in enumerate(visible_indices):
        x, y = float(visible_means2d[i, 0]), float(visible_means2d[i, 1])
        radius = float(visible_radii[i])
        min_x = max(0, min(effective_width, x - radius))
        max_x = max(0, min(effective_width, x + radius))
        min_y = max(0, min(effective_height, y - radius))
        max_y = max(0, min(effective_height, y + radius))
        tile_min_x = max(0, min(int(min_x // tile_size), num_tiles_x - 1))
        tile_max_x = max(0, min(int(max_x // tile_size), num_tiles_x - 1))
        tile_min_y = max(0, min(int(min_y // tile_size), num_tiles_y - 1))
        tile_max_y = max(0, min(int(max_y // tile_size), num_tiles_y - 1))
        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                tile_to_gaussians[(tx, ty)].append(int(gid))
                gaussian_to_tiles[int(gid)].append((tx, ty))
    return tile_to_gaussians, gaussian_to_tiles, num_tiles_x, num_tiles_y


def _build_workload_frame(
    frame_id: int,
    tile_to_gauss: Dict[Tuple[int, int], List[int]],
    num_tiles_x: int,
    num_tiles_y: int,
    gaussian_attrs: Dict[int, GaussianAttr],
    tile_size: int,
    eff_width: int,
    eff_height: int,
    chunk_size: int,
    fov_x: float,
    foveated_enabled: bool,
    total_gaussians_in_scene: int = 0,
) -> WorkloadFrame:
    tiles: Dict[int, TileWorkload] = {}
    
    # 收集所有不重复的可见高斯 ID（跨 tile 去重）
    all_visible_gaussian_ids = set()
    for ids in tile_to_gauss.values():
        all_visible_gaussian_ids.update(ids)
    
    # 统计不重复的高斯数量
    total_visible_gaussians = len(all_visible_gaussian_ids)
    
    # 过滤 gaussian_attrs，只保留视锥内可见的高斯
    visible_gaussian_attrs = {
        gid: attr for gid, attr in gaussian_attrs.items() 
        if gid in all_visible_gaussian_ids
    }
    
    # 计算 frame 级别的 label_counts（去重后）
    frame_label_counts = {0: 0, 1: 0, 2: 0}
    for gid in all_visible_gaussian_ids:
        attr = visible_gaussian_attrs.get(gid)
        if attr is not None and attr.label is not None:
            lb = attr.label
            if lb in (0, 1, 2):
                frame_label_counts[lb] = frame_label_counts.get(lb, 0) + 1
    
    # 计算 visible_ratio
    visible_ratio = total_visible_gaussians / total_gaussians_in_scene if total_gaussians_in_scene > 0 else 0.0
    
    # 计算 culling_rate：需要知道每种标签的总数和可见数
    # 统计场景中所有高斯的标签分布
    scene_label_counts = {0: 0, 1: 0, 2: 0}
    for attr in gaussian_attrs.values():
        if attr.label is not None:
            lb = attr.label
            if lb in (0, 1, 2):
                scene_label_counts[lb] = scene_label_counts.get(lb, 0) + 1
    
    # 计算每种标签的 culling_rate
    culling_rate = {0: 0.0, 1: 0.0, 2: 0.0}
    for label in (0, 1, 2):
        total_label_count = scene_label_counts.get(label, 0)
        visible_label_count = frame_label_counts.get(label, 0)
        if total_label_count > 0:
            culling_rate[label] = 1.0 - (visible_label_count / total_label_count)
        else:
            culling_rate[label] = 0.0
    
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_id = ty * num_tiles_x + tx
            ids = tile_to_gauss.get((tx, ty), [])
            chunks: List[int] = []
            chunk_label_counts: List[Dict[int, int]] = []
            label_counts = {0: 0, 1: 0, 2: 0}
            # 按顺序分 chunk，计算每个 chunk 内的标签计数
            if len(ids) > 0:
                for gid in ids:
                    attr = visible_gaussian_attrs.get(gid)
                    if attr is not None and attr.label is not None:
                        lb = attr.label
                        if lb in (0, 1, 2):
                            label_counts[lb] = label_counts.get(lb, 0) + 1
                pos = 0
                while pos < len(ids):
                    take = min(chunk_size, len(ids) - pos)
                    chunk_ids = ids[pos:pos + take]
                    pos += take
                    chunks.append(take)
                    c_counts = {0: 0, 1: 0, 2: 0}
                    for gid in chunk_ids:
                        attr = visible_gaussian_attrs.get(gid)
                        if attr is not None and attr.label is not None:
                            lb = attr.label
                            if lb in (0, 1, 2):
                                c_counts[lb] = c_counts.get(lb, 0) + 1
                    chunk_label_counts.append(c_counts)
            tiles[tile_id] = TileWorkload(
                tile_id=tile_id,
                gaussian_ids=ids,
                chunk_sizes=chunks,
                chunk_label_counts=chunk_label_counts,
                label_counts=label_counts,
                region=_classify_region(tx, ty, tile_size, eff_width, eff_height, fov_x=fov_x, foveated_enabled=foveated_enabled),
            )

    return WorkloadFrame(
        frame_id=frame_id,
        width=eff_width,
        height=eff_height,
        tile_size=tile_size,
        num_gaussians=total_visible_gaussians,
        num_tiles=num_tiles_x * num_tiles_y,
        tiles=tiles,
        gaussian_attrs=visible_gaussian_attrs,
        visible_ratio=visible_ratio,
        label_counts=frame_label_counts,
        culling_rate=culling_rate,
    )


def load_workload_from_scene(
    config: dict,
    config_path: str = None,
    tile_size: int = 32,
    chunk_size: int = 256,
    verbose: bool = True,
) -> Optional[List[WorkloadFrame]]:
    """
    根据 config 中 dataset/scene/frames 加载真实 4DGS 模型，按 tile 汇总高斯属性。
    若模型缺失或出错，返回 None。
    """
    sim_cfg = config.get("simulation", {})
    algo = config.get("algorithm", {})
    use_synthetic = config.get("workload", {}).get("use_synthetic", False)
    if use_synthetic:
        return None

    dataset = sim_cfg.get("dataset")
    scene = sim_cfg.get("scene")
    model_path = sim_cfg.get("model_path") or config.get("model_path")
    if not model_path and dataset and scene:
        base = sim_cfg.get("base_output", "output")
        model_path = os.path.join(base, dataset, scene)
    if not model_path or not os.path.isdir(model_path):
        if verbose:
            print(f"[workload_loader] model_path 不存在: {model_path}")
        return None

    source_path = sim_cfg.get("source_path") or config.get("source_path")
    if not source_path and dataset and scene:
        candidate = os.path.join("data", dataset, scene)
        if os.path.isdir(candidate):
            source_path = os.path.abspath(candidate)
    if not source_path or not os.path.isdir(source_path):
        if verbose:
            print(f"[workload_loader] source_path 不存在: {source_path}")
        return None

    iteration = sim_cfg.get("iteration", config.get("iteration", -1))
    frames_cfg = sim_cfg.get("frames", 0)
    frame_ids = [frames_cfg] if isinstance(frames_cfg, int) else [int(x) for x in str(frames_cfg).replace(" ", "").split(",")]
    camera_split = sim_cfg.get("camera_split", "video")

    try:
        from argparse import ArgumentParser, Namespace
        from scene import Scene
        from gaussian_renderer import render
        from gaussian_renderer import GaussianModel
        from arguments import ModelParams, PipelineParams, ModelHiddenParams
    except ImportError as e:
        if verbose:
            print(f"[workload_loader] 导入算法模块失败: {e}")
        return None

    parser = ArgumentParser()
    model_param_group = ModelParams(parser, sentinel=True)
    pipeline_param_group = PipelineParams(parser)
    hidden_param_group = ModelHiddenParams(parser)
    args = Namespace()
    args.model_path = model_path
    args.source_path = source_path
    args.images = "images"
    args.resolution = -1
    args.white_background = True
    args.data_device = "cuda"
    args.eval = True
    args.render_process = False
    args.add_points = False
    args.extension = ".png"
    args.llffhold = 8
    args.sh_degree = 3
    cfg_file = os.path.join(model_path, "cfg_args")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg_str = f.read()
            cfg = eval(cfg_str)
            for k, v in vars(cfg).items():
                if k not in ("model_path", "source_path"):
                    setattr(args, k, v)
        except Exception:
            pass

    try:
        model_params = model_param_group.extract(args)
        pipeline_params = pipeline_param_group.extract(args)
        hyperparam = hidden_param_group.extract(args)
    except Exception as e:
        if verbose:
            print(f"[workload_loader] 参数提取失败: {e}")
        return None

    try:
        gaussians = GaussianModel(model_params.sh_degree, hyperparam)
        scene_obj = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        if verbose:
            print(f"[workload_loader] 加载模型失败: {e}")
        return None

    if camera_split == "video":
        cameras = scene_obj.getVideoCameras()
    elif camera_split == "test":
        cameras = scene_obj.getTestCameras()
    else:
        cameras = scene_obj.getTrainCameras()
    if not cameras:
        if verbose:
            print("[workload_loader] 相机列表为空")
        return None

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = getattr(scene_obj, "dataset_type", None)

    # 读取或生成动静标签
    gaussian_labels = None
    label_cfg = config.get("labeling", {})
    label_npy = label_cfg.get("output_npy", "motion_labels.npy")
    label_path = os.path.join(model_path, label_npy)
    if os.path.isfile(label_path):
        try:
            gaussian_labels = np.load(label_path)
            if verbose:
                print(f"[workload_loader] loaded labels from {label_path}")
        except Exception as e:
            if verbose:
                print(f"[workload_loader] 读取标签失败: {e}")
    if gaussian_labels is None and config_path:
        try:
            if verbose:
                print("[workload_loader] 标签缺失，自动生成...")
            labels, _ = generate_motion_labels(config_path)
            gaussian_labels = labels
            if gaussian_labels is None and os.path.isfile(label_path):
                gaussian_labels = np.load(label_path)
        except Exception as e:
            if verbose:
                print(f"[workload_loader] 自动生成标签失败: {e}")
            gaussian_labels = None
    if gaussian_labels is not None and len(gaussian_labels) != gaussians._xyz.shape[0]:
        if verbose:
            print(f"[workload_loader] 标签长度与高斯数不匹配，忽略标签 ({len(gaussian_labels)} vs {gaussians._xyz.shape[0]})")
        gaussian_labels = None
    
    # 提取高斯属性（包含标签）
    gaussian_attrs = _extract_gaussian_attrs(gaussians, gaussian_labels)
    workloads: List[WorkloadFrame] = []
    for fid in frame_ids:
        if fid < 0 or fid >= len(cameras):
            if verbose:
                print(f"[workload_loader] 跳过无效 frame {fid}")
            continue
        view = cameras[fid]
        try:
            with torch.no_grad():
                rendering_results = render(
                    view,
                    gaussians,
                    pipeline_params,
                    background,
                    stage="fine",
                    cam_type=cam_type,
                )
        except Exception as e:
            if verbose:
                print(f"[workload_loader] 渲染 frame {fid} 失败: {e}")
            continue
        viewspace_points = rendering_results.get("viewspace_points")
        radii = rendering_results.get("radii")
        if viewspace_points is None or radii is None:
            if verbose:
                print(f"[workload_loader] frame {fid} 缺少 viewspace_points/radii")
            continue

        full_w, full_h = int(view.image_width), int(view.image_height)
        eff_w = (full_w // tile_size) * tile_size
        eff_h = (full_h // tile_size) * tile_size
        if eff_w <= 0 or eff_h <= 0:
            if verbose:
                print(f"[workload_loader] frame {fid} 有效尺寸为0，跳过")
            continue
        
        # 检查 viewspace_points 是否全零（CUDA 代码不会修改传入的 means2D）
        # 如果是全零，则需要手动从 means3D 计算 2D 投影
        # 只检查前两列（x, y），因为第三列本来就是 0
        if viewspace_points.shape[1] >= 2:
            xy_zeros = torch.allclose(viewspace_points[:, :2], torch.zeros_like(viewspace_points[:, :2]), atol=1e-6)
        else:
            xy_zeros = torch.allclose(viewspace_points, torch.zeros_like(viewspace_points), atol=1e-6)
        
        if xy_zeros:
            if verbose:
                print(f"[workload_loader] frame {fid} viewspace_points 全零，手动计算 2D 投影")
            # 获取 3D 点坐标
            means3D = gaussians.get_xyz  # (N, 3)
            # 获取投影矩阵
            projmatrix = view.full_proj_transform.cuda()  # (4, 4)
            # 手动计算 2D 投影
            viewspace_points = _project_3d_to_2d(means3D, projmatrix, full_w, full_h)
        
        tile_map, _, ntx, nty = _compute_tile_cover(viewspace_points, radii, eff_w, eff_h, tile_size)
        # 获取场景中总高斯数
        total_gaussians_in_scene = gaussians._xyz.shape[0]
        wl = _build_workload_frame(
            frame_id=fid,
            tile_to_gauss=tile_map,
            num_tiles_x=ntx,
            num_tiles_y=nty,
            gaussian_attrs=gaussian_attrs,
            tile_size=tile_size,
            eff_width=eff_w,
            eff_height=eff_h,
            chunk_size=chunk_size,
            fov_x=algo.get("fov_x", 90.0),
            foveated_enabled=algo.get("foveated_enabled", True),
            total_gaussians_in_scene=total_gaussians_in_scene,
        )
        workloads.append(wl)
        if verbose:
            max_tile = max((len(v) for v in tile_map.values()), default=0)
            print(f"[workload_loader] frame {fid}: tiles={ntx*nty}, visible_gaussians={wl.num_gaussians}, max_per_tile={max_tile}")

    return workloads if workloads else None

if __name__ == "__main__":
    import yaml
    from simulator.utils.visualize_tiles import visualize_tile_regions
    config_path = "simulator/configs/default.yaml"  
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    workloads = load_workload_from_scene(config, config_path=config_path, tile_size=32, chunk_size=256, verbose=True)
    if workloads is None:
        print("failed to load workloads (returned None)")
    else:
        print("\n" + "=" * 80)
        print(f"成功加载 {len(workloads)} 个 WorkloadFrame")
        print("=" * 80)
        
        # 为每个 frame 打印详细信息
        for frame_idx, wl in enumerate(workloads):
            print(f"\n{'='*80}")
            print(f"Frame {frame_idx} (frame_id={wl.frame_id})")
            print(f"{'='*80}")
            
            # 基本信息
            print(f"\n[基本信息]")
            print(f"  分辨率: {wl.width} x {wl.height}")
            print(f"  Tile 大小: {wl.tile_size}")
            print(f"  Tile 数量: {wl.num_tiles} ({wl.width // wl.tile_size} x {wl.height // wl.tile_size})")
            
            # 高斯统计
            total_gaussians_in_scene = int(wl.num_gaussians / wl.visible_ratio) if wl.visible_ratio > 0 else 0
            print(f"\n[高斯统计]")
            print(f"  场景总高斯数: {total_gaussians_in_scene}")
            print(f"  可见高斯数: {wl.num_gaussians}")
            print(f"  可见比例 (visible_ratio): {wl.visible_ratio:.4f} ({wl.visible_ratio*100:.2f}%)")
            print(f"  高斯属性数量: {len(wl.gaussian_attrs)}")
            
            # 标签分布
            print(f"\n[标签分布 (label_counts)]")
            label_names = {0: "静止 (Static)", 1: "微动 (Quasi-static)", 2: "巨变 (Dynamic)"}
            total_labeled = sum(wl.label_counts.values())
            if total_labeled > 0:
                for label_id in (0, 1, 2):
                    count = wl.label_counts.get(label_id, 0)
                    ratio = count / total_labeled if total_labeled > 0 else 0.0
                    print(f"  {label_names[label_id]}: {count:6d} ({ratio*100:5.2f}%)")
                if total_labeled < wl.num_gaussians:
                    unlabeled = wl.num_gaussians - total_labeled
                    print(f"  未标记: {unlabeled:6d} ({(unlabeled/wl.num_gaussians)*100:5.2f}%)")
            else:
                print("  无标签信息")
            
            # Culling 率
            print(f"\n[Culling 率 (culling_rate)]")
            if any(rate > 0 for rate in wl.culling_rate.values()):
                for label_id in (0, 1, 2):
                    rate = wl.culling_rate.get(label_id, 0.0)
                    print(f"  {label_names[label_id]}: {rate*100:5.2f}%")
            else:
                print("  所有高斯都可见（无 culling）")
            
            # Tile 区域分布
            print(f"\n[Tile 区域分布]")
            region_counts = {"fovea": 0, "transition": 0, "periphery": 0}
            region_gauss = {"fovea": 0, "transition": 0, "periphery": 0}
            region_chunks = {"fovea": 0, "transition": 0, "periphery": 0}
            max_gauss_per_tile = 0
            min_gauss_per_tile = float('inf')
            
            for tile in wl.tiles.values():
                r = tile.region
                region_counts[r] = region_counts.get(r, 0) + 1
                region_gauss[r] = region_gauss.get(r, 0) + tile.num_gaussians
                region_chunks[r] = region_chunks.get(r, 0) + tile.num_chunks
                max_gauss_per_tile = max(max_gauss_per_tile, tile.num_gaussians)
                if tile.num_gaussians > 0:
                    min_gauss_per_tile = min(min_gauss_per_tile, tile.num_gaussians)
            
            for region in ["fovea", "transition", "periphery"]:
                tile_count = region_counts.get(region, 0)
                gauss_count = region_gauss.get(region, 0)
                chunk_count = region_chunks.get(region, 0)
                if tile_count > 0:
                    avg_gauss = gauss_count / tile_count
                    avg_chunks = chunk_count / tile_count
                    print(f"  {region:12s}: {tile_count:3d} tiles, {gauss_count:8d} gaussians (avg: {avg_gauss:6.1f}/tile), {chunk_count:4d} chunks (avg: {avg_chunks:4.1f}/tile)")
            
            print(f"\n  Tile 高斯数范围: {min_gauss_per_tile if min_gauss_per_tile != float('inf') else 0} - {max_gauss_per_tile}")
            
            # Chunk 统计
            total_chunks = sum(tile.num_chunks for tile in wl.tiles.values())
            if total_chunks > 0:
                print(f"\n[Chunk 统计]")
                print(f"  总 Chunk 数: {total_chunks}")
                print(f"  平均每 Tile: {total_chunks / wl.num_tiles:.2f} chunks")
        
        # 可视化第一个帧的 tile 区域及高斯数
        if len(workloads) > 0:
            print(f"\n{'='*80}")
            print("生成可视化...")
            print(f"{'='*80}")
            out_img = "simulator/results/workload_tile_regions.png"
            visualize_tile_regions(workloads[0], out_img)
            print(f"[viz] 已保存 tile 区域图到: {out_img}")
            
            # 打印第一个帧的汇总
            print(f"\n[第一个 Frame 汇总]")
            wl0 = workloads[0]
            region_counts = {"fovea": 0, "transition": 0, "periphery": 0}
            region_gauss = {"fovea": 0, "transition": 0, "periphery": 0}
            for tile in wl0.tiles.values():
                r = tile.region
                region_counts[r] = region_counts.get(r, 0) + 1
                region_gauss[r] = region_gauss.get(r, 0) + tile.num_gaussians
            print(f"  Tile 数量: {region_counts}")
            print(f"  高斯数量: {region_gauss}")
            print(f"  可见比例: {wl0.visible_ratio*100:.2f}%")
            print(f"  标签分布: Static={wl0.label_counts.get(0,0)}, Quasi={wl0.label_counts.get(1,0)}, Dynamic={wl0.label_counts.get(2,0)}")
        
        print(f"\n{'='*80}")
        print("分析完成")
        print(f"{'='*80}\n")