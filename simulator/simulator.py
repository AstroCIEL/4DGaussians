import os
import yaml
import simpy
import json
import copy
from datetime import datetime

from simulator.structures import (
    WorkloadFrame,
    SimStats,
    TileTask,
    build_synthetic_workload,
    parse_resolution,
)
from simulator.workload_loader import load_workload_from_scene
from simulator.preprocess_udpe import UDPEConfig, UnifiedDeformPreprocessEngine
from simulator.sort_hse import HSEConfig, HierarchicalSortEngine
from simulator.scheduler_wbs import WBSConfig, WorkloadBalancingScheduler
from simulator.rasterize_fre import FREConfig, FoveatedRasterEngine
from simulator.memory import MemoryConfig, MemorySystem
from simulator.analyzer import Analyzer


class Simulator:
    """simpy 事件驱动的三阶段流水模拟器。"""

    def __init__(self, config_path: str, config_override = None, dump_enabled: bool = True):
        self.config_path = config_path
        self.config = config_override if config_override is not None else self._load_config(config_path)
        self.stats = SimStats()
        self.analyzer = Analyzer(
            self.stats,
            output_path=self.config.get("output", {}).get("stats_file", "results/stats.json"),
            verbose=self.config.get("output", {}).get("verbose", True),
            dump_enabled=dump_enabled,
        )
        self.workloads = self._build_workloads()

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_workloads(self):
        algo = self.config.get("algorithm", {})
        tile_size = algo.get("tile_size", 32)
        chunk_size = self.config.get("workload", {}).get("chunk_size", 256)
        verbose = self.config.get("output", {}).get("verbose", True)
        workloads = load_workload_from_scene(self.config, config_path=self.config_path, tile_size=tile_size, chunk_size=chunk_size, verbose=verbose)
        if workloads:
            return workloads

        # 合成负载回退
        sim_cfg = self.config.get("simulation", {})
        res_str = sim_cfg.get("resolution", "1440x1024")
        width, height = parse_resolution(res_str)
        width = (width // tile_size) * tile_size
        height = (height // tile_size) * tile_size
        frames_cfg = sim_cfg.get("frames", 0)
        frame_ids = [frames_cfg] if isinstance(frames_cfg, int) else [int(x) for x in str(frames_cfg).replace(" ", "").split(",")]
        num_gaussians = self.config.get("workload", {}).get("num_gaussians", 100_000)
        fov_x = self.config.get("algorithm", {}).get("fov_x", 90.0)
        foveated_enabled = self.config.get("algorithm", {}).get("foveated_enabled", True)
        return [
            build_synthetic_workload(
                width,
                height,
                tile_size,
                num_gaussians,
                chunk_size=chunk_size,
                frame_id=fid,
                fov_x=fov_x,
                foveated_enabled=foveated_enabled,
            )
            for fid in frame_ids
        ]

    def _build_components(self, env: simpy.Environment):
        hw = self.config.get("hardware", {})
        algo = self.config.get("algorithm", {})
        clock = hw.get("clock_frequency", 1.0) # GHz
        mem = MemorySystem(
            MemoryConfig(
                memory_bandwidth_gbps=hw.get('memory', {}).get("memory_bandwidth", 51.2),
                cache_size_bytes=hw.get('memory', {}).get("cache_size", 1_048_576),
                clock_frequency_ghz=clock,
                read_latency_hiding_rate=hw.get('memory', {}).get("read_latency_hiding_rate", 0.8),
            )
        )
        udpe = UnifiedDeformPreprocessEngine(
            env,
            UDPEConfig(
                cull_cycles=hw.get("udpe", {}).get("cull_cycles", 1.0),
                deform_cycles=hw.get("udpe", {}).get("deform_cycles", 5.0),
                intersection_cycles=hw.get("udpe", {}).get("intersection_cycles", 2.0),
                fifo_depth=hw.get("udpe", {}).get("fifo_depth", 16),
                static_ratio=self.config.get("workload", {}).get("static_ratio", 0.4),
                quasi_ratio=self.config.get("workload", {}).get("quasi_ratio", 0.4),
                skip_enabled=hw.get("udpe", {}).get("skip_enabled", True),
            ),
            self.analyzer,
        )
        hse = HierarchicalSortEngine(
            env,
            HSEConfig(
                num_cores=hw.get("hse", {}).get("num_cores", 16),
                coarse_cycles_per_chunk=hw.get("hse", {}).get("coarse_sort_cycles", 4.0),
                fine_cycles_per_chunk=hw.get("hse", {}).get("fine_sort_cycles", 4.0),
                early_stop_ratio=algo.get("early_stop_ratio", 0.3),
            ),
            self.analyzer,
        )
        fre = FoveatedRasterEngine(
            env,
            FREConfig(
                num_cores=hw.get("fre", {}).get("num_cores", 16),
                base_cycles_per_gaussian=hw.get("fre", {}).get("base_cycles_per_gaussian", 2.0),
                interpolation_cycles=hw.get("fre", {}).get("interpolation_cycles", 8.0),
                early_stop_ratio=algo.get("early_stop_ratio", 0.3),
            ),
            self.analyzer,
        )
        wbs = WorkloadBalancingScheduler(
            env,
            WBSConfig(window_size=hw.get("wbs", {}).get("window_size_k", 8), fifo_depth=hw.get("wbs", {}).get("fifo_depth", 32)),
            sort_engine=hse,
            raster_engine=fre,
            analyzer=self.analyzer,
        )
        return mem, udpe, hse, wbs, fre

    def _feed_workload(self, env: simpy.Environment, frame: WorkloadFrame, udpe: UnifiedDeformPreprocessEngine):
        """将整个 frame 注入 UDPE，带背压。"""
        try:
            yield udpe.in_queue.put(frame)
        except simpy.resources.store.StoreFull:
            self.analyzer.record_fifo_block("udpe_in_full")
            yield udpe.in_queue.put(frame)
        # 发送结束信号
        yield udpe.in_queue.put(None)

    def _wire_modules(self, env: simpy.Environment, udpe, wbs):
        udpe.start()
        wbs.start()

        # 链接 FIFO：udpe -> wbs（UDPE 输出直接送到 WBS）
        env.process(self._pipe(env, udpe.out_queue, wbs.in_queue, "udpe_to_wbs"))

    def _pipe(self, env: simpy.Environment, src, dst, name: str):
        while True:
            item = yield src.get()
            try:
                yield dst.put(item)
            except simpy.resources.store.StoreFull:
                self.analyzer.record_fifo_block(name)
                yield dst.put(item)
            if item is None:
                break

    def _feed_all_workloads(self, env: simpy.Environment, udpe: UnifiedDeformPreprocessEngine):
        """将所有 frame 依次注入 UDPE，实现流水线处理。"""
        for frame in self.workloads:
            try:
                yield udpe.in_queue.put(frame)
            except simpy.resources.store.StoreFull:
                self.analyzer.record_fifo_block("udpe_in_full")
                yield udpe.in_queue.put(frame)
        # 发送结束信号
        yield udpe.in_queue.put(None)

    def _estimate_memory_cycles(self, mem: MemorySystem, frame: WorkloadFrame) -> float:
        """基于高斯数估算一次帧的内存 stall 周期。"""
        bytes_per_gaussian = self.config.get("hardware", {}).get("memory", {}).get("bytes_per_gaussian", 64)
        total_gaussians = sum(t.num_gaussians for t in frame.tiles.values())
        bytes_accessed = mem.estimate_bytes_for_gaussians(total_gaussians, bytes_per_gaussian)
        return mem.estimate_cycles(bytes_accessed)

    def run(self):
        """
        运行仿真，支持多 frame 流水线处理。
        所有 frame 共享同一个 simpy 环境，UDPE 和 WBS 可以流水线工作。
        """
        # 记录模拟开始时间和配置
        self.analyzer.start_simulation(self.config)
        
        if not self.workloads:
            hw = self.config.get("hardware", {})
            clock_period_ns = hw.get("clock_period_ns", 1.0)
            self.analyzer.finalize(0.0, clock_period_ns)
            return self.stats
        
        # 创建全局环境，所有 frame 共享
        env = simpy.Environment()
        mem, udpe, hse, wbs, fre = self._build_components(env)
        self._wire_modules(env, udpe, wbs)
        
        # 将所有 frame 送入 UDPE（流水线处理）
        env.process(self._feed_all_workloads(env, udpe))
        
        # 运行仿真直到所有任务完成
        env.run()
        
        # HSE/FRE 为多核模块，此处在仿真结束后统计其沿时间轴的忙碌时长（critical path）
        if hasattr(hse, "finalize_busy"):
            hse.finalize_busy()
        if hasattr(fre, "finalize_busy"):
            fre.finalize_busy()
        
        # 计算内存 stall 周期（所有 frame 累加）
        total_mem_cycles = 0.0
        for frame in self.workloads:
            mem_cycles = self._estimate_memory_cycles(mem, frame)
            if mem_cycles > 0:
                self.analyzer.record_busy("memory", mem_cycles)
                total_mem_cycles += mem_cycles
        
        # 记录总周期数
        total_cycles = env.now + total_mem_cycles
        
        # 由于是流水线处理，每个 frame 的完成时间难以精确追踪
        # 这里使用总周期数除以 frame 数作为平均每帧周期
        avg_frame_cycles = total_cycles / len(self.workloads) if self.workloads else 0.0
        for _ in self.workloads:
            self.stats.frame_cycles.append(avg_frame_cycles)
        
        # 获取时钟周期配置（从 GHz 转换为纳秒）
        hw = self.config.get("hardware", {})
        clock_frequency_ghz = hw.get("clock_frequency", 1.0)
        # clock_period_ns = 1 / clock_frequency_ghz (GHz转纳秒: 1GHz = 1ns周期)
        clock_period_ns = 1.0 / clock_frequency_ghz if clock_frequency_ghz > 0 else 1.0
        
        # 完成统计，计算每帧用时
        self.analyzer.finalize(total_cycles, clock_period_ns)
        return self.stats


def run_simulator(config_path: str) -> SimStats:
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    sim_cfg = (base_config or {}).get("simulation", {}) if isinstance(base_config, dict) else {}
    dataset = sim_cfg.get("dataset")
    scene = sim_cfg.get("scene")

    def _scene_is_missing(v) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    if not dataset or not _scene_is_missing(scene):
        sim = Simulator(config_path, config_override=base_config, dump_enabled=True)
        return sim.run()

    # scene 未指定：枚举 dataset 下所有 scene，逐个运行，并将结果汇总到同一输出文件
    base_output = sim_cfg.get("base_output", "output")
    model_root = os.path.join(base_output, dataset)
    data_root = os.path.join("data", dataset)

    scenes_model = set()
    scenes_data = set()
    if os.path.isdir(model_root):
        scenes_model = {d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))}
    if os.path.isdir(data_root):
        scenes_data = {d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))}

    scenes = sorted(list(scenes_model & scenes_data)) if (scenes_model and scenes_data) else sorted(list(scenes_model or scenes_data))
    if not scenes:
        # 没有可枚举场景：回退为单次运行（可能走合成 workload）
        sim = Simulator(config_path, config_override=base_config, dump_enabled=True)
        return sim.run()

    output_path = (base_config.get("output", {}) if isinstance(base_config, dict) else {}).get("stats_file", "results/stats.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results: dict = {
        "mode": "multi_scene",
        "dataset": dataset,
        "scenes": [],
        "generated_at": datetime.now().isoformat(),
        "config_path": os.path.abspath(config_path),
    }

    print(f"[simulator] multi-scene enabled. dataset={dataset}, scenes={len(scenes)}")
    for i, sc in enumerate(scenes):
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("simulation", {})
        cfg["simulation"]["dataset"] = dataset
        cfg["simulation"]["scene"] = sc

        # 多场景模式下：禁止 Analyzer 覆盖写，最终由这里统一写汇总文件
        print(f"[simulator] ({i+1}/{len(scenes)}) running scene={sc}")
        sim = Simulator(config_path, config_override=cfg, dump_enabled=False)
        st = sim.run()
        d = st.to_dict()
        d["simulation"] = {"dataset": dataset, "scene": sc}
        results["scenes"].append(d)

        # 同一运行内每个 scene 都单独打印一段（Analyzer 仍会打印 summary）
        # 这里再加一行收尾，方便 grep/查看
        print(f"[simulator] ({i+1}/{len(scenes)}) done scene={sc}, total_cycles={getattr(st, 'total_cycles', 0.0):.2f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
