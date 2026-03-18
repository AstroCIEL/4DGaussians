from dataclasses import dataclass
import simpy
from typing import List, Optional

from simulator.analyzer import Analyzer
from simulator.structures import TileTask, WorkloadFrame
from simulator.rasterize_fre import FoveatedRasterEngine
from simulator.sort_hse import HierarchicalSortEngine


@dataclass
class WBSConfig:
    window_size: int = 8
    fifo_depth: int = 2  # 保留用于兼容性，但不再使用
    scheduling_mode: str = "hilbert_window"  # "hilbert_window" | "hilbert_fifo" | "default_fifo" | "global_greedy"


def hilbert_curve_order(x: int, y: int, n: int) -> int:
    """
    计算点 (x, y) 在 n x n 网格上的希尔伯特曲线顺序。
    使用标准的迭代希尔伯特曲线映射算法（xy2d）。
    
    Args:
        x: tile 的 x 坐标（0 到 n-1）
        y: tile 的 y 坐标（0 到 n-1）
        n: 网格大小（向上取整到最近的 2 的幂）
    
    Returns:
        希尔伯特曲线顺序值（0 到 n*n-1）
    """
    if n <= 0:
        return 0
    
    # 计算网格大小（向上取整到最近的 2 的幂）
    s = 1
    while s < n:
        s <<= 1
    
    # 限制坐标范围
    x = min(x, s - 1)
    y = min(y, s - 1)
    
    # 迭代计算希尔伯特曲线顺序（xy2d 算法）
    d = 0
    rx = 0
    ry = 0
    t = s >> 1
    
    while t > 0:
        rx = 1 if (x & t) else 0
        ry = 1 if (y & t) else 0
        d += t * t * ((3 * rx) ^ ry)
        
        # 旋转和反射
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        
        t >>= 1
    
    return d


def sort_tasks_by_hilbert_curve(tasks: List[TileTask], width: int, height: int, tile_size: int) -> List[TileTask]:
    """
    按照希尔伯特曲线顺序对 TileTask 列表进行排序。
    
    Args:
        tasks: TileTask 列表
        width: 图像宽度
        height: 图像高度
        tile_size: tile 大小
    
    Returns:
        按希尔伯特曲线顺序排序的 TileTask 列表
    """
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size
    
    # 计算每个任务的希尔伯特曲线顺序
    def get_hilbert_order(task: TileTask) -> int:
        # 从 tile_id 计算 (tx, ty)
        tx = task.tile_id % num_tiles_x
        ty = task.tile_id // num_tiles_x
        # 使用较大的维度作为网格大小
        n = max(num_tiles_x, num_tiles_y)
        return hilbert_curve_order(tx, ty, n)
    
    # 按希尔伯特曲线顺序排序
    sorted_tasks = sorted(tasks, key=get_hilbert_order)
    return sorted_tasks


def sort_tasks_by_default_order(tasks: List[TileTask], width: int, height: int, tile_size: int) -> List[TileTask]:
    """
    按照默认顺序（从左到右从上到下，即 tile_id 顺序）对 TileTask 列表进行排序。
    
    Args:
        tasks: TileTask 列表
        width: 图像宽度（未使用，保持接口一致性）
        height: 图像高度（未使用，保持接口一致性）
        tile_size: tile 大小（未使用，保持接口一致性）
    
    Returns:
        按 tile_id 顺序排序的 TileTask 列表
    """
    # 按 tile_id 排序（从左到右从上到下）
    sorted_tasks = sorted(tasks, key=lambda task: task.tile_id)
    return sorted_tasks


class WorkloadBalancingScheduler:
    """
    窗口化空间局部性调度，管理 HSE 和 FRE 核的解耦调度。
    
    支持三种调度模式：
    1. "hilbert_window": 希尔伯特曲线排序 + 窗口内贪心（创新模式）
       - UDPE 处理完一个 frame 后，将所有 TileTask 写回 DRAM（延迟不建模）
       - WBS 接收整个 frame 的所有 TileTask，按希尔伯特曲线排序
       - 使用容量为 K 的滑动窗口作为任务分发池
       - 当 HSE 单元可用时，从窗口中选择最大工作负载的任务
       - 当窗口最前端的图块被分发后，窗口向后滑动
    
    2. "default_fifo": 默认顺序 + FIFO 分配（Baseline 1）
       - Tile 按照从左到右从上到下的默认顺序排序
       - 分配给最先空闲的 HSE 核心（FIFO）
    
    3. "global_greedy": 全局贪心分配（Baseline 2）
       - 整个 frame 的 tiletask 直接按工作负载最重的优先分配
       - 不设任何窗口限制
    4. "hilbert_fifo": 希尔伯特曲线排序 + FIFO 分配（确定性模式）
       - 仍使用希尔伯特顺序以保留空间局部性
       - 但窗口内不做“最大 workload 优先”的在线决策，避免因任务时长变化导致的非单调 makespan
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: WBSConfig,
        sort_engine: HierarchicalSortEngine,
        raster_engine: FoveatedRasterEngine,
        analyzer: Analyzer,
        width: int,
        height: int,
        tile_size: int,
    ):
        self.env = env
        self.config = config
        self.sort_engine = sort_engine
        self.raster_engine = raster_engine
        self.analyzer = analyzer
        self.width = width
        self.height = height
        self.tile_size = tile_size
        
        # 移除 in_queue，改为接收整个 frame 的 TileTask 列表
        self.frame_queue = simpy.Store(env, capacity=config.fifo_depth)
        
        # 全局任务列表（根据模式排序）
        self.global_task_list: List[TileTask] = []
        self.global_list_index = 0  # 当前已提取到全局列表的位置（用于 hilbert_window 和 default_fifo）
        
        # 滑动窗口任务池（容量为 K，仅用于 hilbert_window 模式）
        self.window_pool: List[TileTask] = []
        self.window_start_index = 0  # 窗口在全局列表中的起始位置
        
        # 已分发标记：记录窗口中的任务是否已被分发（仅用于 hilbert_window 模式）
        self.window_dispatched: List[bool] = []
        
        # 全局贪心模式的待分发任务列表（仅用于 global_greedy 模式）
        self.pending_greedy: List[TileTask] = []
        
        self.sorted_queue: List[TileTask] = []  # 排序完成等待光栅化的任务队列
        self.upstream_done = False
        self.hse_in_flight = 0  # HSE 核正在处理的任务数
        self.fre_in_flight = 0  # FRE 核正在处理的任务数
        # 调度统计：记录每次从窗口选中的 task workload（num_gaussians）
        self._window_selected_workloads: List[int] = []

    def start(self):
        return self.env.process(self._run())

    def _fill_window(self):
        """从全局任务列表中提取任务填充窗口，直到窗口容量为 K。"""
        while len(self.window_pool) < self.config.window_size and self.global_list_index < len(self.global_task_list):
            task = self.global_task_list[self.global_list_index]
            self.window_pool.append(task)
            self.window_dispatched.append(False)
            self.global_list_index += 1

    def _slide_window(self):
        """
        当窗口最前端的图块被分发后，窗口向后滑动。
        移除已分发的任务，并补充新的任务。
        """
        if not self.window_pool:
            return

        # 记录原窗口中首个未分发任务之前的已分发数量，用于更新起始位置
        first_undispatched_idx = None
        for i, dispatched in enumerate(self.window_dispatched):
            if not dispatched:
                first_undispatched_idx = i
                break
        if first_undispatched_idx is None:
            self.window_start_index += len(self.window_pool)
        else:
            self.window_start_index += first_undispatched_idx

        # 移除所有已分发任务，保留未分发任务的相对顺序并前移压实
        compacted_tasks = []
        for task, dispatched in zip(self.window_pool, self.window_dispatched):
            if not dispatched:
                compacted_tasks.append(task)
        self.window_pool = compacted_tasks
        self.window_dispatched = [False] * len(self.window_pool)

        # 补充窗口到容量 K
        self._fill_window()

    def _dispatch_hse(self):
        """
        根据调度模式分发任务到 HSE 核。
        """
        if self.config.scheduling_mode == "hilbert_window":
            self._dispatch_hse_hilbert_window()
        elif self.config.scheduling_mode == "hilbert_fifo":
            self._dispatch_hse_hilbert_fifo()
        elif self.config.scheduling_mode == "default_fifo":
            self._dispatch_hse_default_fifo()
        elif self.config.scheduling_mode == "global_greedy":
            self._dispatch_hse_global_greedy()
        else:
            raise ValueError(f"Unknown scheduling mode: {self.config.scheduling_mode}")

    def _dispatch_hse_hilbert_fifo(self):
        """
        确定性调度：在希尔伯特顺序下，窗口仅作为缓冲，始终优先分发窗口最前端任务（FIFO）。
        这样分发顺序不依赖于任务完成时间变化，从而避免“某些参数变好但 makespan 变差”的反直觉现象。
        """
        # 确保窗口已填充
        self._fill_window()
        while self.window_pool and self.sort_engine.has_free_core():
            # 找到窗口最前端第一个未分发任务
            front_idx = -1
            for i, dispatched in enumerate(self.window_dispatched):
                if not dispatched:
                    front_idx = i
                    break
            if front_idx == -1:
                break
            task = self.window_pool[front_idx]
            self.window_dispatched[front_idx] = True
            self.hse_in_flight += 1
            self._window_selected_workloads.append(int(task.num_gaussians))
            self.env.process(self._process_sort(task))
            # 只要最前端被分发，就滑窗
            if front_idx == 0:
                self._slide_window()

    def _dispatch_hse_hilbert_window(self):
        """
        尝试把窗口内 workload 最大的未分发任务分配给空闲的 HSE 核。
        使用最长处理时间（LPT）策略。
        """
        while self.window_pool and self.sort_engine.has_free_core():
            # 找到窗口中未分发且工作负载最大的任务
            max_idx = -1
            max_workload = -1
            for i, task in enumerate(self.window_pool):
                if not self.window_dispatched[i] and task.num_gaussians > max_workload:
                    max_workload = task.num_gaussians
                    max_idx = i
            
            if max_idx == -1:
                # 窗口中没有未分发的任务
                break
            
            # 分发任务
            task = self.window_pool[max_idx]
            self.window_dispatched[max_idx] = True
            self.hse_in_flight += 1
            self._window_selected_workloads.append(int(task.num_gaussians))
            # 启动 HSE 核进行排序
            self.env.process(self._process_sort(task))
            
            # 如果分发的是窗口最前端的任务，滑动窗口
            if max_idx == 0:
                self._slide_window()

    def _dispatch_hse_default_fifo(self):
        """
        Baseline 1: 默认顺序 + FIFO 分配。
        按照全局任务列表的顺序（从左到右从上到下），分配给最先空闲的 HSE 核心。
        """
        while self.global_list_index < len(self.global_task_list) and self.sort_engine.has_free_core():
            # 按顺序取出下一个任务
            task = self.global_task_list[self.global_list_index]
            self.global_list_index += 1
            self.hse_in_flight += 1
            # 启动 HSE 核进行排序
            self.env.process(self._process_sort(task))

    def _dispatch_hse_global_greedy(self):
        """
        Baseline 2: 全局贪心分配。
        从整个 frame 的待分发任务中选择工作负载最重的任务分配给空闲的 HSE 核心。
        """
        while self.pending_greedy and self.sort_engine.has_free_core():
            # 找到工作负载最大的任务
            max_idx = max(range(len(self.pending_greedy)), key=lambda i: self.pending_greedy[i].num_gaussians)
            task = self.pending_greedy.pop(max_idx)
            self.hse_in_flight += 1
            # 启动 HSE 核进行排序
            self.env.process(self._process_sort(task))

    def _dispatch_fre(self):
        """尝试把排序完成的任务分配给空闲的 FRE 核。"""
        while self.sorted_queue and self.raster_engine.has_free_core():
            task = self.sorted_queue.pop(0)  # FIFO：先排序完成的先光栅化
            self.fre_in_flight += 1
            # 启动 FRE 核进行光栅化
            self.env.process(self._process_raster(task))

    def _process_sort(self, task: TileTask):
        """HSE 核处理排序任务。"""
        yield self.sort_engine.process(task)
        # 排序完成，加入等待光栅化的队列
        self.sorted_queue.append(task)
        self.hse_in_flight -= 1
        # 尝试立即分发到 FRE 核
        self._dispatch_fre()
        # 尝试分发新的 HSE 任务
        self._dispatch_hse()

    def _process_raster(self, task: TileTask):
        """FRE 核处理光栅化任务。"""
        yield self.raster_engine.process(task)
        # 光栅化完成
        self.fre_in_flight -= 1

    def _process_frame(self, tasks: List[TileTask]):
        """
        处理一个 frame 的所有 TileTask。
        根据调度模式选择不同的处理方式。
        """
        if not tasks:
            return
        
        if self.config.scheduling_mode in ("hilbert_window", "hilbert_fifo"):
            # 按希尔伯特曲线排序
            sorted_tasks = sort_tasks_by_hilbert_curve(tasks, self.width, self.height, self.tile_size)
            # 添加到全局任务列表
            self.global_task_list.extend(sorted_tasks)
            # 填充窗口
            self._fill_window()
        elif self.config.scheduling_mode == "default_fifo":
            # 按默认顺序排序（从左到右从上到下）
            sorted_tasks = sort_tasks_by_default_order(tasks, self.width, self.height, self.tile_size)
            # 添加到全局任务列表
            self.global_task_list.extend(sorted_tasks)
            # 不需要填充窗口，直接按顺序分发
        elif self.config.scheduling_mode == "global_greedy":
            # 直接添加到待分发列表（不排序，贪心时会选择最大的）
            self.pending_greedy.extend(tasks)
        else:
            raise ValueError(f"Unknown scheduling mode: {self.config.scheduling_mode}")

    def _is_all_done(self) -> bool:
        """检查是否所有任务都完成。"""
        if not self.upstream_done:
            return False
        if self.sorted_queue or self.hse_in_flight > 0 or self.fre_in_flight > 0:
            return False
        
        if self.config.scheduling_mode == "hilbert_window":
            return (len(self.global_task_list) == self.global_list_index and 
                    not self.window_pool)
        elif self.config.scheduling_mode == "default_fifo":
            return len(self.global_task_list) == self.global_list_index
        elif self.config.scheduling_mode == "global_greedy":
            return not self.pending_greedy
        else:
            return False

    def _run(self):
        while True:
            # 检查是否所有任务都完成
            if self._is_all_done():
                break

            # 尝试分发任务到 HSE 和 FRE 核
            self._dispatch_hse()
            self._dispatch_fre()
            
            if self._is_all_done():
                break

            # 等待新 frame 的 TileTask 列表（若队列为空会阻塞）
            t0 = self.env.now
            frame_tasks = yield self.frame_queue.get()
            dt = self.env.now - t0
            if dt > 0:
                self.analyzer.record_fifo_block_cycles("wbs.frame_get", dt)
            
            if frame_tasks is None:
                self.upstream_done = True
            else:
                # 处理整个 frame 的 TileTask 列表
                self._process_frame(frame_tasks)
            
            # 再次尝试分发
            self._dispatch_hse()
            self._dispatch_fre()

    def get_window_selected_workloads(self) -> List[int]:
        return list(self._window_selected_workloads)
