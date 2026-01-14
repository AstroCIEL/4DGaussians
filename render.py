#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
import sys
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, quiet: bool = False, render_num: int = 9999999):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    deform_time_list=[] # deformation time statistics
    rasterization_time_list=[] # rasterization(full forward cuda kernel) time statistic
    total_time_list=[]
    deform_breakdown_lists = {}
    print("point nums:",gaussians._xyz.shape[0])
    # When running under profilers (e.g., Nsight Compute), stdout/stderr may not be a TTY and tqdm
    # can spam one line per update. Use --quiet to disable the progress bar in those cases.
    progress = tqdm(
        views,
        desc="Rendering progress",
        disable=bool(quiet),
        file=sys.stderr,
        dynamic_ncols=True,
        leave=not bool(quiet),
    )
    for idx, view in enumerate(progress):
        if idx >= min(render_num, len(views)):
            break
        rendering_results = render(view, gaussians, pipeline, background,cam_type=cam_type)
        rendering = rendering_results["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if rendering_results["time"] is not None:
            deform_time_list.append(rendering_results["time"]["deformation"])
            rasterization_time_list.append(rendering_results["time"]["rasterization"])
            total_time_list.append(rendering_results["time"]["total"])
            bd = rendering_results["time"].get("deformation_breakdown")
            if isinstance(bd, dict):
                for k, v in bd.items():
                    if not isinstance(v, (int, float)):
                        continue
                    deform_breakdown_lists.setdefault(k, []).append(float(v))
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    print("FPS:",1/np.mean(total_time_list))
    if deform_breakdown_lists:    
        sta_file=os.path.join(model_path, name, "ours_{}".format(iteration), "statistics_deform_detail.txt")
    else:
        sta_file=os.path.join(model_path, name, "ours_{}".format(iteration), "statistics.txt")
    with open(sta_file, "w") as f:
        f.write("gaussian point nums: {}\n".format(gaussians._xyz.shape[0]))
        f.write("Deformation Time ave: {}\n".format(np.mean(deform_time_list)))
        f.write("Rasterization Time ave: {}\n".format(np.mean(rasterization_time_list)))
        f.write("FPS ave: {}\n".format(1/np.mean(total_time_list)))
        if deform_breakdown_lists:
            f.write("\n")
            f.write("Deformation Breakdown (seconds, GPU-event based):\n")
            for k in sorted(deform_breakdown_lists.keys()):
                vals = deform_breakdown_lists[k]
                if len(vals) == 0:
                    continue
                f.write("  {} ave: {}\n".format(k, float(np.mean(vals))))
        if (np.mean(rasterization_time_list)>1.0):
            f.write("WARNING: Detected rasterization Time ave > 1.0s\n")
            f.write("You may running nsight compute. FPS data invalid\n")

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, quiet: bool = False, render_num: int = 9999999):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, quiet=quiet, render_num=render_num)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, quiet=quiet, render_num=render_num)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, cam_type, quiet=quiet, render_num=render_num)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--render_num", type=int, default=9999999)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        quiet=args.quiet,
        render_num=args.render_num,
    )