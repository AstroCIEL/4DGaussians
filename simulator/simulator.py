import os
import yaml
import simpy

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

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.stats = SimStats()
        self.analyzer = Analyzer(self.stats, output_path=self.config.get("output", {}).get("stats_file", "results/stats.json"), verbose=self.config.get("output", {}).get("verbose", True))
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
        frames_cfg = sim_cfg.get("frames", 0)
        frame_ids = [frames_cfg] if isinstance(frames_cfg, int) else [int(x) for x in str(frames_cfg).replace(" ", "").split(",")]
        num_gaussians = self.config.get("workload", {}).get("num_gaussians", 100_000)
        return [
            build_synthetic_workload(width, height, tile_size, num_gaussians, chunk_size=chunk_size, frame_id=fid)
            for fid in frame_ids
        ]

    def _build_components(self, env: simpy.Environment):
        hw = self.config.get("hardware", {})
        algo = self.config.get("algorithm", {})
        clock = hw.get("clock_frequency", 1.0)
        mem = MemorySystem(
            MemoryConfig(
                memory_bandwidth_gbps=hw.get("memory_bandwidth", 51.2),
                cache_size_bytes=hw.get("cache_size", 1_048_576),
                clock_frequency_ghz=clock,
            )
        )
        udpe = UnifiedDeformPreprocessEngine(
            env,
            UDPEConfig(
                cull_cycles=hw.get("cull_cycles", 2.0),
                deform_cycles=hw.get("deform_cycles", 4.0),
                intersection_cycles=hw.get("intersection_cycles", 1.0),
                fifo_depth=hw.get("fifo_depth", 32),
                static_ratio=self.config.get("workload", {}).get("static_ratio", 0.4),
                quasi_ratio=self.config.get("workload", {}).get("quasi_ratio", 0.4),
            ),
            self.analyzer,
        )
        hse = HierarchicalSortEngine(
            env,
            HSEConfig(
                coarse_cycles_per_chunk=hw.get("coarse_sort_cycles", 4.0),
                fine_cycles_per_gaussian=hw.get("fine_sort_cycles", 0.05),
                fifo_depth=hw.get("fifo_depth", 32),
            ),
            self.analyzer,
        )
        fre = FoveatedRasterEngine(
            env,
            FREConfig(
                num_cores=hw.get("rasterizing_units", 16),
                base_cycles_per_gaussian=hw.get("raster_cycles", 1.0),
                interpolation_cycles=hw.get("interpolation_cycles", 8.0),
            ),
            self.analyzer,
        )
        wbs = WorkloadBalancingScheduler(
            env,
            WBSConfig(window_size=hw.get("window_size_k", 8), fifo_depth=hw.get("fifo_depth", 32)),
            raster_engine=fre,
            analyzer=self.analyzer,
        )
        return mem, udpe, hse, wbs, fre

    def _feed_workload(self, env: simpy.Environment, frame: WorkloadFrame, udpe: UnifiedDeformPreprocessEngine):
        """将 tile/chunk 任务注入 UDPE，带背压。"""
        for tile in frame.tiles.values():
            if tile.num_gaussians <= 0:
                continue
            # 若 chunk_sizes 为空但有高斯列表，退化为一个 chunk
            chunk_sizes = tile.chunk_sizes or ([len(tile.gaussian_ids)] if tile.gaussian_ids else [])
            for idx, csize in enumerate(chunk_sizes):
                c_labels = None
                if tile.chunk_label_counts and idx < len(tile.chunk_label_counts):
                    c_labels = tile.chunk_label_counts[idx]
                task = TileTask(
                    frame_id=frame.frame_id,
                    tile_id=tile.tile_id,
                    num_gaussians=csize,
                    region=tile.region,
                    chunk_index=idx,
                    gaussian_ids=tile.gaussian_ids,
                    label_counts=c_labels,
                )
                try:
                    yield udpe.in_queue.put(task)
                except simpy.resources.store.StoreFull:
                    self.analyzer.record_fifo_block("udpe_in_full")
                    yield udpe.in_queue.put(task)
        # 发送结束信号
        yield udpe.in_queue.put(None)

    def _wire_modules(self, env: simpy.Environment, udpe, hse, wbs):
        udpe.start()
        hse.start()
        wbs.start()

        # 链接 FIFO：udpe -> hse -> wbs
        env.process(self._pipe(env, udpe.out_queue, hse.in_queue, "udpe_to_hse"))
        env.process(self._pipe(env, hse.out_queue, wbs.in_queue, "hse_to_wbs"))

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

    def _run_single_frame(self, frame: WorkloadFrame) -> float:
        env = simpy.Environment()
        mem, udpe, hse, wbs, fre = self._build_components(env)
        self._wire_modules(env, udpe, hse, wbs)
        env.process(self._feed_workload(env, frame, udpe))
        env.run()
        return env.now

    def run(self):
        total = 0.0
        for frame in self.workloads:
            cycles = self._run_single_frame(frame)
            self.stats.frame_cycles.append(cycles)
            total += cycles
        self.analyzer.finalize(total)
        return self.stats


def run_simulator(config_path: str) -> SimStats:
    sim = Simulator(config_path)
    stats = sim.run()
    return stats
