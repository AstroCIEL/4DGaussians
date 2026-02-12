# simulator/workload_loader.py
"""从算法侧加载真实 workload：根据 config 的 dataset/scene/frame 加载训练好的高斯模型，渲染得到每 tile 高斯数以构建 WorkloadFrame。"""

import os
import sys
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

# 保证从 4DGaussians 项目根可导入 scene / gaussian_renderer / arguments
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .structures import WorkloadFrame


def _compute_tile_gaussians(
    means2D,
    radii,
    image_width: int,
    image_height: int,
    tile_size: int,
):
    """
    计算每个 tile 需要处理的高斯球数量（与 scripts/analyze_tile_gaussians.py 中逻辑一致）。
    means2D/radii 可为 torch.Tensor 或 numpy，返回 (tile_gaussians_dict, visible_count)。
    """
    if hasattr(means2D, "cpu"):
        means2D = means2D.detach().cpu().numpy()
    if hasattr(radii, "cpu"):
        radii = radii.detach().cpu().numpy()
    if means2D.shape[1] == 3:
        means2D = means2D[:, :2]
    visible_mask = radii > 0
    visible_means2D = means2D[visible_mask]
    visible_radii = radii[visible_mask]
    visible_count = int(visible_mask.sum())
    if len(visible_means2D) == 0:
        num_tiles_x = math.ceil(image_width / tile_size)
        num_tiles_y = math.ceil(image_height / tile_size)
        return defaultdict(int), visible_count, num_tiles_x * num_tiles_y

    num_tiles_x = math.ceil(image_width / tile_size)
    num_tiles_y = math.ceil(image_height / tile_size)
    tile_gaussians = defaultdict(int)
    for i in range(len(visible_means2D)):
        x, y = float(visible_means2D[i, 0]), float(visible_means2D[i, 1])
        radius = float(visible_radii[i])
        min_x = max(0, x - radius)
        max_x = min(image_width, x + radius)
        min_y = max(0, y - radius)
        max_y = min(image_height, y + radius)
        tile_min_x = max(0, min(int(min_x // tile_size), num_tiles_x - 1))
        tile_max_x = max(0, min(int(max_x // tile_size), num_tiles_x - 1))
        tile_min_y = max(0, min(int(min_y // tile_size), num_tiles_y - 1))
        tile_max_y = max(0, min(int(max_y // tile_size), num_tiles_y - 1))
        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                tile_gaussians[(tx, ty)] += 1
    num_tiles = num_tiles_x * num_tiles_y
    return dict(tile_gaussians), visible_count, num_tiles


def _tile_gaussians_to_workload_frame(
    tile_gaussians: Dict[Tuple[int, int], int],
    num_tiles_x: int,
    num_tiles_y: int,
    visible_gaussians: int,
    frame_id: int,
    chunk_size: int,
) -> WorkloadFrame:
    """将 (tx,ty)->count 转为 WorkloadFrame（含 chunk 划分）。"""
    num_tiles = num_tiles_x * num_tiles_y
    tile_info = {}
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_id = ty * num_tiles_x + tx
            n = tile_gaussians.get((tx, ty), 0)
            if n <= 0:
                chunk_list = []
                num_chunks = 0
            else:
                num_chunks = (n + chunk_size - 1) // chunk_size
                chunk_list = [min(chunk_size, n - i * chunk_size) for i in range(num_chunks)]
            tile_info[tile_id] = (n, num_chunks, chunk_list)
    return WorkloadFrame(
        frame_id=frame_id,
        num_gaussians=visible_gaussians,
        num_tiles=num_tiles,
        tile_info=tile_info,
    )


def load_workload_from_scene(
    config: dict,
    tile_size: int,
    chunk_size: int = 256,
    verbose: bool = True,
) -> Optional[List[WorkloadFrame]]:
    """
    根据 config 中的 dataset / scene / frames（及可选的 model_path、source_path、iteration）
    加载训练好的 4DGS 模型与对应 view，渲染得到每帧的 tile 高斯分布，构建 WorkloadFrame 列表。
    若未配置或加载失败则返回 None（caller 应回退到合成 workload）。
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
            print(f"[workload_loader] 未配置或不存在 model_path: {model_path}，将使用合成 workload")
        return None

    source_path = sim_cfg.get("source_path") or config.get("source_path")
    if not source_path:
        cfg_path = os.path.join(model_path, "cfg_args")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg_content = f.read()
                if "source_path" in cfg_content:
                    import re
                    m = re.search(r"source_path['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", cfg_content)
                    if m:
                        source_path = m.group(1)
            except Exception:
                pass
        if not source_path and dataset and scene:
            candidate = os.path.join("data", dataset, scene)
            if os.path.isdir(candidate):
                source_path = os.path.abspath(candidate)
    if not source_path or not os.path.isdir(source_path):
        if verbose:
            print(f"[workload_loader] 未找到 source_path: {source_path}，将使用合成 workload")
        return None

    iteration = sim_cfg.get("iteration", config.get("iteration", -1))
    frames_cfg = sim_cfg.get("frames", 0)
    if isinstance(frames_cfg, int):
        frame_ids = [frames_cfg]
    else:
        frame_ids = [int(x) for x in str(frames_cfg).replace(" ", "").split(",")]

    camera_split = sim_cfg.get("camera_split", "video")  # "video" | "test" | "train"

    try:
        from argparse import ArgumentParser, Namespace
        from scene import Scene
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        from arguments import ModelParams, PipelineParams, ModelHiddenParams
    except ImportError as e:
        if verbose:
            print(f"[workload_loader] 无法导入算法模块: {e}，将使用合成 workload")
        return None

    parser = ArgumentParser()
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    ModelHiddenParams(parser)
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
    if os.path.isfile(os.path.join(model_path, "cfg_args")):
        try:
            with open(os.path.join(model_path, "cfg_args"), "r") as f:
                cfg_str = f.read()
            cfg = eval(cfg_str)
            for k, v in vars(cfg).items():
                if k not in ("model_path", "source_path"):
                    setattr(args, k, v)
        except Exception:
            pass

    try:
        model_params = ModelParams(parser, sentinel=True).extract(args)
        pipeline_params = PipelineParams(parser).extract(args)
        hyperparam = ModelHiddenParams(parser).extract(args)
    except Exception as e:
        if verbose:
            print(f"[workload_loader] 参数提取失败: {e}，将使用合成 workload")
        return None

    try:
        gaussians = GaussianModel(model_params.sh_degree, hyperparam)
        scene_obj = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        if verbose:
            print(f"[workload_loader] 场景/模型加载失败: {e}，将使用合成 workload")
        return None

    if camera_split == "video":
        cameras = scene_obj.getVideoCameras()
    elif camera_split == "test":
        cameras = scene_obj.getTestCameras()
    else:
        cameras = scene_obj.getTrainCameras()
    if not cameras:
        if verbose:
            print("[workload_loader] 相机列表为空，将使用合成 workload")
        return None

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = getattr(scene_obj, "dataset_type", None)

    workloads: List[WorkloadFrame] = []
    for fid in frame_ids:
        if fid < 0 or fid >= len(cameras):
            if verbose:
                print(f"[workload_loader] 跳过无效 frame_id={fid}（共 {len(cameras)} 个视图）")
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
                print(f"[workload_loader] frame {fid} 无 viewspace_points/radii")
            continue
        w, h = int(view.image_width), int(view.image_height)
        tile_gaussians, visible_count, num_tiles = _compute_tile_gaussians(
            viewspace_points, radii, w, h, tile_size
        )
        num_tiles_x = math.ceil(w / tile_size)
        num_tiles_y = math.ceil(h / tile_size)
        wl = _tile_gaussians_to_workload_frame(
            tile_gaussians,
            num_tiles_x,
            num_tiles_y,
            visible_count,
            fid,
            chunk_size,
        )
        workloads.append(wl)
        if verbose:
            print(f"[workload_loader] frame {fid}: visible_gaussians={visible_count}, num_tiles={num_tiles}, max_per_tile={max(tile_gaussians.values()) if tile_gaussians else 0}")

    if not workloads:
        if verbose:
            print("[workload_loader] 未得到任何有效帧 workload，将使用合成 workload")
        return None
    return workloads
