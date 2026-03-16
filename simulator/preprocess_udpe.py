from dataclasses import dataclass
from typing import List, Tuple
import random
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask, WorkloadFrame
from simulator.memory import MemorySystem, MemoryConfig


@dataclass
class UDPEConfig:
    cull_cycles: float = 1.0               # 单高斯 cull 开销
    deform_cycles: float = 5.0             # 单高斯 deform 开销
    intersection_cycles: float = 2.0       # 单高斯 intersection 开销
    fifo_depth: int = 16
    static_ratio: float = 0.4
    quasi_ratio: float = 0.4  # 其余视为 dynamic
    skip_enabled: bool = True


class UnifiedDeformPreprocessEngine:
    """
    UDPE：以 frame 为单位的前端形变/剔除建模。
    - 接收整个 WorkloadFrame，以 frame 级别的高斯集合作为处理单位
    - cull 与 deform 模块并行，intersection 在存活后执行
    - 按 frame 级别计算处理周期，提高硬件利用率
    - 处理完成后将 frame 拆分为 TileTask 输出给下游模块
    """

    def __init__(self, env: simpy.Environment, config: UDPEConfig, analyzer: Analyzer, memory: MemorySystem):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.memory = memory
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.out_queue = simpy.Store(env, capacity=config.fifo_depth)
        random.seed(0)

    def classify_counts(self, frame: WorkloadFrame) -> Tuple[int, int, int]:
        """
        根据 frame 级别的 label_counts 统计不同类型的高斯数量。
        直接使用 WorkloadFrame 中预计算的 label_counts（已在 workload_loader 中去重）。
        返回: (static_n, quasi_n, dynamic_n)
        """
        total_gaussians = frame.num_gaussians
        if total_gaussians <= 0:
            return 0, 0, 0
        
        # 优先使用 frame 中预计算的 label_counts
        if frame.label_counts:
            static_n = frame.label_counts.get(0, 0)
            quasi_n = frame.label_counts.get(1, 0)
            dynamic_n = frame.label_counts.get(2, 0)
            
            # 如果统计到的标签总数小于总高斯数，说明有些高斯没有标签
            # 将剩余的高斯按比例分配
            labeled_count = static_n + quasi_n + dynamic_n
            if labeled_count < total_gaussians:
                unlabeled_count = total_gaussians - labeled_count
                if labeled_count > 0:
                    # 按已有标签的比例分配未标记的高斯
                    ratio_static = static_n / labeled_count
                    ratio_quasi = quasi_n / labeled_count
                    static_n += int(unlabeled_count * ratio_static)
                    quasi_n += int(unlabeled_count * ratio_quasi)
                    dynamic_n = total_gaussians - static_n - quasi_n
                else:
                    # 所有高斯都没有标签，使用默认比例
                    static_n = int(total_gaussians * self.config.static_ratio)
                    quasi_n = int(total_gaussians * self.config.quasi_ratio)
                    dynamic_n = max(0, total_gaussians - static_n - quasi_n)
            return static_n, quasi_n, dynamic_n
        
        # 如果没有预计算的 label_counts，回退到从 gaussian_attrs 统计
        if frame.gaussian_attrs:
            static_n = 0
            quasi_n = 0
            dynamic_n = 0
            
            # 遍历所有高斯球属性，统计标签
            for attr in frame.gaussian_attrs.values():
                if attr.label is not None:
                    if attr.label == 0:
                        static_n += 1
                    elif attr.label == 1:
                        quasi_n += 1
                    elif attr.label == 2:
                        dynamic_n += 1
            
            labeled_count = static_n + quasi_n + dynamic_n
            if labeled_count < total_gaussians:
                unlabeled_count = total_gaussians - labeled_count
                if labeled_count > 0:
                    ratio_static = static_n / labeled_count
                    ratio_quasi = quasi_n / labeled_count
                    static_n += int(unlabeled_count * ratio_static)
                    quasi_n += int(unlabeled_count * ratio_quasi)
                    dynamic_n = total_gaussians - static_n - quasi_n
                else:
                    static_n = int(total_gaussians * self.config.static_ratio)
                    quasi_n = int(total_gaussians * self.config.quasi_ratio)
                    dynamic_n = max(0, total_gaussians - static_n - quasi_n)
            return static_n, quasi_n, dynamic_n
        
        # 无标签或标签不可用时按比例估计
        static_n = int(total_gaussians * self.config.static_ratio)
        quasi_n = int(total_gaussians * self.config.quasi_ratio)
        dynamic_n = max(0, total_gaussians - static_n - quasi_n)
        return static_n, quasi_n, dynamic_n

    def processing_cycles(self, frame: WorkloadFrame) -> float:
        """
        按 frame 级别计算处理周期：
        - cull 工作量：frame 内所有高斯 N_total
        - deform 工作量：quasi+dynamic 高斯
        - 两者并行：max(cull_time, deform_time)
        - intersection：对所有高斯执行
        """
        if frame.num_gaussians <= 0:
            return 0.0

        static_n, quasi_n, dynamic_n = self.classify_counts(frame)
        total_n = static_n + quasi_n + dynamic_n
        c = self.config
        if c.skip_enabled:
            cull_time = total_n * c.cull_cycles
            deform_time = (quasi_n + dynamic_n) * c.deform_cycles
            inter_time = total_n * c.intersection_cycles
            return max(cull_time, deform_time, inter_time)
        else:
            cull_time = total_n * c.cull_cycles
            deform_time = total_n * c.deform_cycles
            inter_time = total_n * c.intersection_cycles
            return deform_time + deform_time + inter_time
    
    def frame_to_tile_tasks(self, frame: WorkloadFrame) -> List[TileTask]:
        """
        将处理后的 frame 拆分为 TileTask 列表，供下游模块使用。
        """
        tasks = []
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
                tasks.append(task)
        return tasks

    def start(self):
        return self.env.process(self._run())

    def _run(self):
        while True:
            frame = yield self.in_queue.get()
            if frame is None:
                # 透传结束信号
                yield self.out_queue.put(None)
                break
            
            # 1. 访存延迟：获取整个frame的负载（与高斯数量成正比）
            mem_cycles = self.memory.estimate_memory_cycles_for_frame(frame.num_gaussians) * 0.5
            if mem_cycles > 0:
                self.analyzer.record_busy("memory", mem_cycles)
                yield self.env.timeout(mem_cycles)
            
            # 2. 按 frame 级别计算处理周期
            cycles = self.processing_cycles(frame)
            self.analyzer.record_busy("udpe", cycles + mem_cycles)
            yield self.env.timeout(cycles)
            
            # 将 frame 拆分为 TileTask 列表
            tasks = self.frame_to_tile_tasks(frame)
            # 模拟写回 DRAM（延迟不建模，但需要一次性输出整个列表）
            # 一次性输出整个 frame 的 TileTask 列表
            try:
                yield self.out_queue.put(tasks)
            except simpy.resources.store.StoreFull:
                self.analyzer.record_fifo_block("udpe_out_full")
                yield self.out_queue.put(tasks)


def main():
    """独立的 UDPE 测试主函数，用于检查逻辑。"""
    import simpy
    from simulator.structures import SimStats, WorkloadFrame, TileWorkload
    
    print("=" * 60)
    print("UDPE (Unified Deform Preprocess Engine) 测试 - Frame 级别处理")
    print("=" * 60)
    
    # 创建环境
    env = simpy.Environment()
    
    # 创建配置
    config = UDPEConfig(
        cull_cycles=1.0,
        deform_cycles=5.0,
        intersection_cycles=2.0,
        fifo_depth=16,
        static_ratio=0.4,
        quasi_ratio=0.4,
    )
    
    # 创建统计和分析器
    stats = SimStats()
    analyzer = Analyzer(stats, verbose=False)
    
    # 创建 UDPE
    udpe = UnifiedDeformPreprocessEngine(env, config, analyzer)
    
    # 创建测试 frame
    # Frame 0: 包含多个 tile，有标签计数
    tiles_frame0 = {
        0: TileWorkload(
            tile_id=0,
            gaussian_ids=list(range(100)),
            chunk_sizes=[100],
            label_counts={0: 40, 1: 40, 2: 20},
            region="fovea",
        ),
        1: TileWorkload(
            tile_id=1,
            gaussian_ids=list(range(100, 300)),
            chunk_sizes=[200],
            label_counts=None,  # 无标签，使用默认比例
            region="transition",
        ),
        2: TileWorkload(
            tile_id=2,
            gaussian_ids=list(range(300, 350)),
            chunk_sizes=[50],
            label_counts={0: 20, 1: 20, 2: 10},
            region="periphery",
        ),
    }
    frame0 = WorkloadFrame(
        frame_id=0,
        width=1440,
        height=1024,
        tile_size=32,
        num_gaussians=350,
        num_tiles=3,
        tiles=tiles_frame0,
    )
    
    # Frame 1: 不同帧
    tiles_frame1 = {
        0: TileWorkload(
            tile_id=0,
            gaussian_ids=list(range(0, 150)),
            chunk_sizes=[150],
            label_counts={0: 60, 1: 60, 2: 30},
            region="fovea",
        ),
        1: TileWorkload(
            tile_id=1,
            gaussian_ids=None,
            chunk_sizes=[80],
            label_counts=None,
            region="transition",
        ),
    }
    frame1 = WorkloadFrame(
        frame_id=1,
        width=1440,
        height=1024,
        tile_size=32,
        num_gaussians=230,
        num_tiles=2,
        tiles=tiles_frame1,
    )
    
    test_frames = [frame0, frame1]
    
    # 输出 frame 信息
    print("\n[输入 Frame]")
    for i, frame in enumerate(test_frames, 1):
        print(f"\nFrame {i} (frame_id={frame.frame_id}):")
        print(f"  总高斯数: {frame.num_gaussians}")
        print(f"  Tile 数量: {frame.num_tiles}")
        for tile_id, tile in frame.tiles.items():
            print(f"    Tile {tile_id}: {tile.num_gaussians} 高斯, region={tile.region}, label_counts={tile.label_counts}")
    
    # 定义发送 frame 的进程
    def send_frames():
        for frame in test_frames:
            print(f"\n[时间 {env.now:.2f}] 发送 Frame {frame.frame_id}: {frame.num_gaussians} 高斯")
            yield udpe.in_queue.put(frame)
        print(f"\n[时间 {env.now:.2f}] 发送结束信号")
        yield udpe.in_queue.put(None)
    
    # 定义接收 TileTask 列表的进程
    def receive_tasks():
        frame_count = 0
        while True:
            tasks = yield udpe.out_queue.get()
            if tasks is None:
                print(f"\n[时间 {env.now:.2f}] 收到结束信号")
                break
            frame_count += 1
            print(f"[时间 {env.now:.2f}] 收到 Frame {frame_count} 的 TileTask 列表: {len(tasks)} 个任务")
            for task in tasks:
                print(f"  - TileTask: frame={task.frame_id}, tile={task.tile_id}, chunk={task.chunk_index}, n={task.num_gaussians}")
    
    # 启动进程
    udpe.start()
    env.process(send_frames())
    env.process(receive_tasks())
    
    # 运行仿真
    print("\n" + "=" * 60)
    print("开始仿真...")
    print("=" * 60)
    env.run()
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("仿真结果统计")
    print("=" * 60)
    print(f"总仿真时间: {env.now:.2f} 周期")
    print(f"UDPE 忙碌时间: {stats.module_busy.get('udpe', 0.0):.2f} 周期")
    print(f"预处理周期: {stats.preprocess_cycles:.2f} 周期")
    if stats.fifo_blocked:
        print(f"FIFO 阻塞次数: {stats.fifo_blocked}")
    
    # 详细分析每个 frame 的处理周期
    print("\n" + "=" * 60)
    print("Frame 处理周期分析")
    print("=" * 60)
    
    for i, frame in enumerate(test_frames, 1):
        cycles = udpe.processing_cycles(frame)
        static_n, quasi_n, dynamic_n = udpe.classify_counts(frame)
        
        print(f"\nFrame {i} (frame_id={frame.frame_id}):")
        print(f"  总高斯数: {frame.num_gaussians}")
        print(f"  分类: static={static_n}, quasi={quasi_n}, dynamic={dynamic_n}")
        print(f"  处理周期: {cycles:.2f}")
        if frame.num_gaussians > 0:
            total_n = static_n + quasi_n + dynamic_n
            print(f"  计算详情:")
            print(f"    cull_time = {total_n} * {config.cull_cycles:.1f} = {total_n * config.cull_cycles:.1f}")
            print(f"    deform_time = {quasi_n + dynamic_n} * {config.deform_cycles:.1f} = {(quasi_n + dynamic_n) * config.deform_cycles:.1f}")
            print(f"    inter_time = {total_n} * {config.intersection_cycles:.1f} = {total_n * config.intersection_cycles:.1f}")
            print(f"    max({total_n * config.cull_cycles:.1f}, {(quasi_n + dynamic_n) * config.deform_cycles:.1f}, {total_n * config.intersection_cycles:.1f}) = {cycles:.2f}")
        
        # 显示拆分后的 TileTask
        tasks = udpe.frame_to_tile_tasks(frame)
        print(f"  拆分后的 TileTask 数量: {len(tasks)}")
        for task in tasks:
            print(f"    - Tile {task.tile_id}, chunk {task.chunk_index}: {task.num_gaussians} 高斯")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
