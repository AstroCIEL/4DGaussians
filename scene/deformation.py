import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane

def _cuda_time_s(fn):
    """
    Time a block on GPU using cuda events and return (out, seconds).
    Falls back to CPU wall time if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        import time as _time
        t0 = _time.perf_counter()
        out = fn()
        return out, _time.perf_counter() - t0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, (start.elapsed_time(end) / 1000.0)


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(
            args.bounds,
            args.kplanes_config,
            args.multires,
            profile=bool(getattr(args, "profile_deformation", False)),
            profile_detail=bool(getattr(args, "profile_deformation_hexplane_detail", False)),
        )
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
        self.last_timing = None
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            # Optional: mark HexPlane interval for GPU-event profiling (no sync here).
            prof = getattr(self, "_lat_profiler", None)
            if prof is not None and getattr(self.args, "profile_latency_hexplane", False):
                prof.mark("hexplane_start")
                grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
                prof.mark("hexplane_end")
            else:
                grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        timing = None
        if bool(getattr(self.args, "profile_deformation", False)):
            timing = {}
            hidden, timing["deform_query_time_s"] = _cuda_time_s(
                lambda: self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
            )
            # merge HexPlane timing if present
            if getattr(self.grid, "last_timing", None):
                for k, v in (self.grid.last_timing or {}).items():
                    timing[k] = v
        else:
            hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)

        if self.args.static_mlp:
            if timing is None:
                mask = self.static_mlp(hidden)
            else:
                mask, timing["deform_mask_s"] = _cuda_time_s(lambda: self.static_mlp(hidden))
        elif self.args.empty_voxel:
            if timing is None:
                mask = self.empty_voxel(rays_pts_emb[:,:3])
            else:
                mask, timing["deform_mask_s"] = _cuda_time_s(lambda: self.empty_voxel(rays_pts_emb[:,:3]))
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            if timing is None:
                dx = self.pos_deform(hidden)
                pts = torch.zeros_like(rays_pts_emb[:,:3])
                pts = rays_pts_emb[:,:3]*mask + dx
            else:
                dx, timing["deform_dx_head_s"] = _cuda_time_s(lambda: self.pos_deform(hidden))
                pts, timing["deform_dx_apply_s"] = _cuda_time_s(lambda: rays_pts_emb[:,:3]*mask + dx)
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            if timing is None:
                ds = self.scales_deform(hidden)
                scales = torch.zeros_like(scales_emb[:,:3])
                scales = scales_emb[:,:3]*mask + ds
            else:
                ds, timing["deform_ds_head_s"] = _cuda_time_s(lambda: self.scales_deform(hidden))
                scales, timing["deform_ds_apply_s"] = _cuda_time_s(lambda: scales_emb[:,:3]*mask + ds)
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            if timing is None:
                dr = self.rotations_deform(hidden)
                rotations = torch.zeros_like(rotations_emb[:,:4])
                if self.args.apply_rotation:
                    rotations = batch_quaternion_multiply(rotations_emb, dr)
                else:
                    rotations = rotations_emb[:,:4] + dr
            else:
                dr, timing["deform_dr_head_s"] = _cuda_time_s(lambda: self.rotations_deform(hidden))
                if self.args.apply_rotation:
                    rotations, timing["deform_dr_apply_s"] = _cuda_time_s(lambda: batch_quaternion_multiply(rotations_emb, dr))
                else:
                    rotations, timing["deform_dr_apply_s"] = _cuda_time_s(lambda: rotations_emb[:,:4] + dr)

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            if timing is None:
                do = self.opacity_deform(hidden) 
                opacity = torch.zeros_like(opacity_emb[:,:1])
                opacity = opacity_emb[:,:1]*mask + do
            else:
                do, timing["deform_do_head_s"] = _cuda_time_s(lambda: self.opacity_deform(hidden))
                opacity, timing["deform_do_apply_s"] = _cuda_time_s(lambda: opacity_emb[:,:1]*mask + do)
        if self.args.no_dshs:
            shs = shs_emb
        else:
            if timing is None:
                dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
                shs = torch.zeros_like(shs_emb)
                # breakpoint()
                shs = shs_emb*mask.unsqueeze(-1) + dshs
            else:
                dshs, timing["deform_dshs_head_s"] = _cuda_time_s(
                    lambda: self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
                )
                shs, timing["deform_dshs_apply_s"] = _cuda_time_s(lambda: shs_emb*mask.unsqueeze(-1) + dshs)

        if timing is not None:
            self.last_timing = timing
        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)
        self.last_timing = None
        
        # Static gaussian optimization: mask for gaussians that don't need deformation
        self._static_mask = None  # (N,) bool tensor, True = static (skip deformation)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    
    def set_static_mask(self, static_mask):
        """
        Set the static mask for skipping deformation on static gaussians.
        
        Args:
            static_mask: (N,) bool tensor, True = static (skip deformation)
        """
        self._static_mask = static_mask
        
    def get_static_mask(self):
        """Get the current static mask"""
        return self._static_mask
    
    def clear_static_mask(self):
        """Clear the static mask (enable deformation for all gaussians)"""
        self._static_mask = None
    
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        timing = None
        
        # Check if we have a static mask to skip deformation for some gaussians
        static_mask = self._static_mask
        use_static_optimization = static_mask is not None and static_mask.any()
        
        if use_static_optimization:
            # Only compute deformation for non-static (dynamic) gaussians
            dynamic_mask = ~static_mask
            num_dynamic = dynamic_mask.sum().item()
            num_total = point.shape[0]
            
            if num_dynamic == 0:
                # All gaussians are static, skip deformation entirely
                return point, scales, rotations, opacity, shs
            
            # Extract dynamic gaussians
            point_dyn = point[dynamic_mask]
            scales_dyn = scales[dynamic_mask]
            rotations_dyn = rotations[dynamic_mask]
            opacity_dyn = opacity[dynamic_mask]
            shs_dyn = shs[dynamic_mask]
            times_sel_dyn = times_sel[dynamic_mask]
            
            # Compute deformation only for dynamic gaussians
            if bool(getattr(self.deformation_net.args, "profile_deformation", False)):
                timing = {}
                timing["num_dynamic_gaussians"] = num_dynamic
                timing["num_total_gaussians"] = num_total
                timing["static_ratio"] = 1.0 - num_dynamic / num_total
                (point_emb, scales_emb, rotations_emb), timing["deform_poc_fre_s"] = _cuda_time_s(
                    lambda: (
                        poc_fre(point_dyn, self.pos_poc),
                        poc_fre(scales_dyn, self.rotation_scaling_poc),
                        poc_fre(rotations_dyn, self.rotation_scaling_poc),
                    )
                )
            else:
                point_emb = poc_fre(point_dyn, self.pos_poc)
                scales_emb = poc_fre(scales_dyn, self.rotation_scaling_poc)
                rotations_emb = poc_fre(rotations_dyn, self.rotation_scaling_poc)
            
            if timing is None:
                means3D_dyn, scales_dyn_out, rotations_dyn_out, opacity_dyn_out, shs_dyn_out = self.deformation_net(
                    point_emb,
                    scales_emb,
                    rotations_emb,
                    opacity_dyn,
                    shs_dyn,
                    None,
                    times_sel_dyn,
                )
            else:
                (means3D_dyn, scales_dyn_out, rotations_dyn_out, opacity_dyn_out, shs_dyn_out), timing["deform_net_forward_s"] = _cuda_time_s(
                    lambda: self.deformation_net(
                        point_emb,
                        scales_emb,
                        rotations_emb,
                        opacity_dyn,
                        shs_dyn,
                        None,
                        times_sel_dyn,
                    )
                )
                if getattr(self.deformation_net, "last_timing", None):
                    for k, v in (self.deformation_net.last_timing or {}).items():
                        timing[k] = v
                self.last_timing = timing
            
            # Merge results: static gaussians keep original values, dynamic gaussians use deformed values
            means3D_out = point.clone()
            scales_out = scales.clone()
            rotations_out = rotations.clone()
            opacity_out = opacity.clone()
            shs_out = shs.clone()
            
            means3D_out[dynamic_mask] = means3D_dyn
            scales_out[dynamic_mask] = scales_dyn_out
            rotations_out[dynamic_mask] = rotations_dyn_out
            opacity_out[dynamic_mask] = opacity_dyn_out
            shs_out[dynamic_mask] = shs_dyn_out
            
            return means3D_out, scales_out, rotations_out, opacity_out, shs_out
        
        # Original path: compute deformation for all gaussians
        if bool(getattr(self.deformation_net.args, "profile_deformation", False)):
            timing = {}
            (point_emb, scales_emb, rotations_emb), timing["deform_poc_fre_s"] = _cuda_time_s(
                lambda: (
                    poc_fre(point, self.pos_poc),
                    poc_fre(scales, self.rotation_scaling_poc),
                    poc_fre(rotations, self.rotation_scaling_poc),
                )
            )
        else:
            # times_emb = poc_fre(times_sel, self.time_poc)
            point_emb = poc_fre(point,self.pos_poc)
            scales_emb = poc_fre(scales,self.rotation_scaling_poc)
            rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        if timing is None:
            means3D, scales, rotations, opacity, shs = self.deformation_net(
                point_emb,
                scales_emb,
                rotations_emb,
                opacity,
                shs,
                None,
                times_sel,
            )
        else:
            (means3D, scales, rotations, opacity, shs), timing["deform_net_forward_s"] = _cuda_time_s(
                lambda: self.deformation_net(
                    point_emb,
                    scales_emb,
                    rotations_emb,
                    opacity,
                    shs,
                    None,
                    times_sel,
                )
            )
            # merge sub timings recorded inside Deformation/HexPlaneField
            if getattr(self.deformation_net, "last_timing", None):
                for k, v in (self.deformation_net.last_timing or {}).items():
                    timing[k] = v
            self.last_timing = timing
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb