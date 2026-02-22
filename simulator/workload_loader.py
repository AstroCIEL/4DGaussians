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


def _extract_gaussian_attrs(gaussians) -> Dict[int, GaussianAttr]:
    """从 GaussianModel 提取关键属性（位置/尺度/旋转/不透明度/SH）。"""
    attrs: Dict[int, GaussianAttr] = {}
    xyz = gaussians._xyz.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy() if hasattr(gaussians, "_scaling") else np.zeros_like(xyz)
    rotation = gaussians._rotation.detach().cpu().numpy() if hasattr(gaussians, "_rotation") else np.zeros((xyz.shape[0], 4))
    opacity = gaussians._opacity.detach().cpu().numpy() if hasattr(gaussians, "_opacity") else np.ones((xyz.shape[0], 1))
    sh_dc = gaussians._features_dc.detach().cpu().numpy() if hasattr(gaussians, "_features_dc") else None
    for i in range(xyz.shape[0]):
        sh_values = sh_dc[i].flatten().tolist() if sh_dc is not None else None
        attrs[i] = GaussianAttr(
            idx=i,
            position=tuple(map(float, xyz[i])),
            scale=tuple(map(float, scaling[i])) if len(scaling.shape) > 1 else (0.0, 0.0, 0.0),
            rotation=tuple(map(float, rotation[i])) if len(rotation.shape) > 1 else (0.0, 0.0, 0.0, 0.0),
            opacity=float(opacity[i][0]) if opacity.ndim > 1 else float(opacity[i]),
            sh=sh_values,
        )
    return attrs


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
    gaussian_labels: Optional[np.ndarray],
    tile_size: int,
    eff_width: int,
    eff_height: int,
    chunk_size: int,
) -> WorkloadFrame:
    tiles: Dict[int, TileWorkload] = {}
    total_gaussians = 0
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_id = ty * num_tiles_x + tx
            ids = tile_to_gauss.get((tx, ty), [])
            total_gaussians += len(ids)
            chunks: List[int] = []
            chunk_label_counts: List[Dict[int, int]] = []
            label_counts = {0: 0, 1: 0, 2: 0}
            # 按顺序分 chunk，计算每个 chunk 内的标签计数
            if len(ids) > 0:
                if gaussian_labels is not None:
                    for gid in ids:
                        lb = int(gaussian_labels[gid]) if gid < len(gaussian_labels) else None
                        if lb is not None and lb in (0, 1, 2):
                            label_counts[lb] = label_counts.get(lb, 0) + 1
                pos = 0
                while pos < len(ids):
                    take = min(chunk_size, len(ids) - pos)
                    chunk_ids = ids[pos:pos + take]
                    pos += take
                    chunks.append(take)
                    if gaussian_labels is not None:
                        c_counts = {0: 0, 1: 0, 2: 0}
                        for gid in chunk_ids:
                            lb = int(gaussian_labels[gid]) if gid < len(gaussian_labels) else None
                            if lb is not None and lb in (0, 1, 2):
                                c_counts[lb] = c_counts.get(lb, 0) + 1
                        chunk_label_counts.append(c_counts)
                    else:
                        chunk_label_counts.append({})
            tiles[tile_id] = TileWorkload(
                tile_id=tile_id,
                gaussian_ids=ids,
                chunk_sizes=chunks,
                chunk_label_counts=chunk_label_counts,
                label_counts=label_counts,
                region=_classify_region(tx, ty, num_tiles_x, num_tiles_y),
            )
    labels_dict = {}
    if gaussian_labels is not None:
        labels_dict = {int(i): int(gaussian_labels[i]) for i in range(len(gaussian_labels))}

    return WorkloadFrame(
        frame_id=frame_id,
        width=eff_width,
        height=eff_height,
        tile_size=tile_size,
        num_gaussians=total_gaussians,
        num_tiles=num_tiles_x * num_tiles_y,
        tiles=tiles,
        gaussian_attrs=gaussian_attrs,
        gaussian_labels=labels_dict,
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
    use_synthetic = config.get("workload", {}).get("use_synthetic", False)
    if use_synthetic:
        return None

    dataset = sim_cfg.get("dataset")
    scene = sim_cfg.get("scene")
    model_path = sim_cfg.get("model_path") or config.get("model_path")
    if not model_path and dataset and scene:
        base = config.get("base_output", "output")
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

    gaussian_attrs = _extract_gaussian_attrs(gaussians)
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
        tile_map, _, ntx, nty = _compute_tile_cover(viewspace_points, radii, eff_w, eff_h, tile_size)
        wl = _build_workload_frame(
            frame_id=fid,
            tile_to_gauss=tile_map,
            num_tiles_x=ntx,
            num_tiles_y=nty,
            gaussian_attrs=gaussian_attrs,
            gaussian_labels=gaussian_labels,
            tile_size=tile_size,
            eff_width=eff_w,
            eff_height=eff_h,
            chunk_size=chunk_size,
        )
        workloads.append(wl)
        if verbose:
            max_tile = max((len(v) for v in tile_map.values()), default=0)
            print(f"[workload_loader] frame {fid}: tiles={ntx*nty}, visible_gaussians={wl.num_gaussians}, max_per_tile={max_tile}")

    return workloads if workloads else None

if __name__ == "__main__":
    import yaml
    config_path = "simulator/configs/default.yaml"  
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    workloads = load_workload_from_scene(config, config_path=config_path, tile_size=32, chunk_size=256, verbose=True)
    if workloads is None:
        print("failed to load workloads (returned None)")
    else:
        print(f"successfully loaded {len(workloads)} workloads of {config['simulation']['scene']}")