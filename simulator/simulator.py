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
from simulator.preprocess_udpe import UDPEConfig, UnifiedDeformPreprocessEngine
from simulator.sort_hse import HSEConfig, HierarchicalSortEngine
from simulator.scheduler_wbs import WBSConfig, WorkloadBalancingScheduler
from simulator.rasterize_fre import FREConfig, FoveatedRasterEngine
from simulator.memory import MemoryConfig, MemorySystem
from simulator.analyzer import Analyzer
from simulator.utils.stats_logger import append_stats_to_csv


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
            relaxation_factor=self.config.get("hardware", {}).get("relaxation_factor", 0.6),
            dump_enabled=dump_enabled,
        )
        self.workloads = self._build_workloads()

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_workloads(self):
        algo = self.config.get("algorithm", {})
        tile_size = algo.get("tile_size", 32)
        verbose = self.config.get("output", {}).get("verbose", True)
        # 真实 workload 加载依赖 numpy/torch；在缺少依赖或加载失败时，必须可靠回退到合成 workload
        try:
            from simulator.workload_loader import load_workload_from_scene  # 延迟导入，避免硬依赖
            workloads = load_workload_from_scene(
                self.config,
                config_path=self.config_path,
                tile_size=tile_size,
                verbose=verbose,
            )
            if workloads:
                return workloads
        except Exception as e:
            if verbose:
                print(f"[simulator] workload_loader unavailable or failed ({type(e).__name__}: {e}). fallback to synthetic workload.")

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
                bandwidth_utilization=hw.get('memory', {}).get("bandwidth_utilization", 0.5),
                clock_frequency_ghz=clock,
                read_latency_hiding_rate=hw.get('memory', {}).get("read_latency_hiding_rate", 0.8),
                bytes_per_gaussian=hw.get('memory', {}).get("bytes_per_gaussian", 120),
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
                udpe_utilization=hw.get("udpe", {}).get("udpe_utilization", 0.9),
            ),
            self.analyzer,
            memory=mem,
        )
        hse = HierarchicalSortEngine(
            env,
            HSEConfig(
                num_cores=hw.get("hse", {}).get("num_cores", 16),
                coarse_cycles=hw.get("hse", {}).get("coarse_sort_cycles", 4.0),
                fine_cycles=hw.get("hse", {}).get("fine_sort_cycles", 4.0),
                early_stop_ratio=algo.get("early_stop_ratio", 0.3),
            ),
            self.analyzer,
            memory=mem,
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
            memory=mem,
        )
        # 获取分辨率信息用于希尔伯特曲线排序
        sim_cfg = self.config.get("simulation", {})
        res_str = sim_cfg.get("resolution", "1440x1024")
        width, height = parse_resolution(res_str)
        tile_size = algo.get("tile_size", 32)
        
        wbs = WorkloadBalancingScheduler(
            env,
            WBSConfig(
                window_size=hw.get("wbs", {}).get("window_size_k", 8),
                fifo_depth=hw.get("wbs", {}).get("fifo_depth", 32),
                scheduling_mode=hw.get("wbs", {}).get("scheduling_mode", "hilbert_window"),
            ),
            sort_engine=hse,
            raster_engine=fre,
            analyzer=self.analyzer,
            width=width,
            height=height,
            tile_size=tile_size,
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

        # 链接 FIFO：udpe -> wbs（UDPE 输出整个 frame 的 TileTask 列表到 WBS）
        env.process(self._pipe(env, udpe.out_queue, wbs.frame_queue, "udpe_to_wbs"))

    def _timed_put(self, env: simpy.Environment, store: simpy.Store, item, name: str):
        """
        对 simpy.Store.put 做计时：如果由于容量限制发生等待，把等待 cycles 计入统计。
        注意：simpy.Store 满时不会抛 StoreFull，而是 put 事件阻塞直到有空间。
        """
        t0 = env.now
        yield store.put(item)
        dt = env.now - t0
        if dt > 0:
            self.analyzer.record_fifo_block_cycles(name, dt)

    def _timed_get(self, env: simpy.Environment, store: simpy.Store, name: str):
        """
        对 simpy.Store.get 做计时：如果由于空队列发生等待，把等待 cycles 计入统计。
        """
        t0 = env.now
        item = yield store.get()
        dt = env.now - t0
        if dt > 0:
            self.analyzer.record_fifo_block_cycles(name, dt)
        return item

    def _pipe(self, env: simpy.Environment, src, dst, name: str):
        while True:
            item = yield from self._timed_get(env, src, f"{name}.src_get")
            yield from self._timed_put(env, dst, item, f"{name}.dst_put")
            if item is None:
                break

    def _feed_all_workloads(self, env: simpy.Environment, udpe: UnifiedDeformPreprocessEngine):
        """将所有 frame 依次注入 UDPE，实现流水线处理。"""
        for frame in self.workloads:
            yield from self._timed_put(env, udpe.in_queue, frame, "udpe.in_put")
        # 发送结束信号
        yield from self._timed_put(env, udpe.in_queue, None, "udpe.in_put")


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

        # 记录平均硬件利用率（需要 total_cycles）
        total_cycles = env.now
        if hasattr(hse, "finalize_utilization"):
            hse.finalize_utilization(total_cycles)
        if hasattr(fre, "finalize_utilization"):
            fre.finalize_utilization(total_cycles)

        def _summarize(values):
            v = [float(x) for x in values if x is not None]
            if not v:
                return {"count": 0}
            v.sort()
            n = len(v)
            s = sum(v)
            def _p(q):
                if n == 1:
                    return v[0]
                # q in [0,1]
                pos = q * (n - 1)
                lo = int(pos)
                hi = min(n - 1, lo + 1)
                w = pos - lo
                return v[lo] * (1 - w) + v[hi] * w
            return {
                "count": n,
                "mean": s / n,
                "max": v[-1],
                "p50": _p(0.50),
                "p90": _p(0.90),
                "p99": _p(0.99),
            }

        # 长尾：任务服务时间分布（HSE/FRE）
        if hasattr(hse, "get_task_service_times"):
            self.analyzer.record_task_time_stats("hse", _summarize(hse.get_task_service_times()))
        if hasattr(fre, "get_task_service_times"):
            self.analyzer.record_task_time_stats("fre", _summarize(fre.get_task_service_times()))

        # 长尾：hilbert_window 的窗口选取 workload 分布
        if hasattr(wbs, "get_window_selected_workloads"):
            wl = wbs.get_window_selected_workloads()
            # workload 是 num_gaussians，整数也用同一 summarize
            self.analyzer.record_scheduling_stats("wbs.window_selected_workload", _summarize(wl))
        
        # 内存延迟已经集成到各个子模块中，不再单独计算
        # 记录总周期数
        
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
    
    # 确定 CSV 路径（与汇总 JSON 在同一目录）
    cfg_output = (base_config or {}).get("output", {}) if isinstance(base_config, dict) else {}
    csv_path = cfg_output.get(
        "csv_file",
        os.path.join(os.path.dirname(output_path), "stats_log.csv"),
    )
    
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

        # 记录到 CSV（每个 scene 一行）
        try:
            append_stats_to_csv(d, csv_path)
        except Exception as e:
            # 不要因为记录 CSV 失败影响主流程
            print(f"[simulator] warn: failed to append stats to csv for scene={sc} ({type(e).__name__}: {e})")

        # 同一运行内每个 scene 都单独打印一段（Analyzer 仍会打印 summary）
        # 这里再加一行收尾，方便 grep/查看
        print(f"[simulator] ({i+1}/{len(scenes)}) done scene={sc}, total_cycles={getattr(st, 'total_cycles', 0.0):.2f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
