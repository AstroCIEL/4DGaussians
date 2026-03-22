from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class MemoryConfig:
    memory_bandwidth_gbps: float = 51.2
    bandwidth_utilization: float = 0.5
    clock_frequency_ghz: float = 1.0
    read_latency_hiding_rate: float = 0.5
    bytes_per_gaussian: int = 64  # 每个高斯的字节数
    cache_hit_enabled: bool = True


class MemorySystem:
    """
    内存模型：根据访问字节数和cache命中率估算延迟。
    
    支持两种模式：
    1. 全量访存：UDPE获取整个frame时，延迟与高斯数量成正比
    2. 增量访存：HSE/FRE处理tile时，考虑cache命中率（基于高斯重复程度）
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.bytes_per_cycle = (
            config.memory_bandwidth_gbps * config.bandwidth_utilization * 1e9 / (config.clock_frequency_ghz * 1e9)
        )
        self._cache_hit_gaussians = 0
        self._cache_total_gaussians = 0

    def estimate_cycles(self, bytes_accessed: float) -> float:
        """返回因带宽限制产生的额外周期。"""
        if self.bytes_per_cycle <= 0:
            return 0.0
        return max(0.0, bytes_accessed / self.bytes_per_cycle) * (1 - self.config.read_latency_hiding_rate)

    def estimate_bytes_for_gaussians(self, num_gaussians: int, bytes_per_gaussian: Optional[int] = None) -> float:
        """简单估计：每个高斯读写若干字节。"""
        if bytes_per_gaussian is None:
            bytes_per_gaussian = self.config.bytes_per_gaussian
        return float(num_gaussians * max(0, bytes_per_gaussian))

    def estimate_memory_cycles_for_frame(self, num_gaussians: int) -> float:
        """
        估算UDPE获取整个frame的访存延迟。
        延迟与frame中的高斯球数量成正比。
        """
        bytes_accessed = self.estimate_bytes_for_gaussians(num_gaussians)
        return self.estimate_cycles(bytes_accessed)

    def estimate_memory_cycles_for_tile(
        self, 
        current_gaussians: Optional[Set[int]], 
        previous_gaussians: Optional[Set[int]],
        num_gaussians: int
    ) -> float:
        """
        估算HSE/FRE处理tile的访存延迟。
        
        Args:
            current_gaussians: 当前tile的高斯ID集合
            previous_gaussians: 上一次处理的tile的高斯ID集合（None表示首次访问）
            num_gaussians: 当前tile的高斯数量
        
        Returns:
            访存延迟（周期数）
        """
        if num_gaussians == 0:
            return 0.0

        cache_hit_ratio, hit_count, total_access, cache_miss_count = self._compute_tile_cache_stats(
            current_gaussians,
            previous_gaussians,
            num_gaussians,
        )

        if total_access > 0:
            self._cache_total_gaussians += int(total_access)
            if self.config.cache_hit_enabled:
                self._cache_hit_gaussians += int(hit_count)

        # 估算访存字节数（只加载未命中的高斯）
        bytes_accessed = self.estimate_bytes_for_gaussians(int(cache_miss_count))

        # 计算访存延迟
        return self.estimate_cycles(bytes_accessed)

    def get_tile_cache_hit_ratio(
        self,
        current_gaussians: Optional[Set[int]],
        previous_gaussians: Optional[Set[int]],
        num_gaussians: int,
    ) -> float:
        if num_gaussians == 0:
            return 0.0
        cache_hit_ratio, _, _, _ = self._compute_tile_cache_stats(
            current_gaussians,
            previous_gaussians,
            num_gaussians,
        )
        return float(cache_hit_ratio)

    def _compute_tile_cache_stats(
        self,
        current_gaussians: Optional[Set[int]],
        previous_gaussians: Optional[Set[int]],
        num_gaussians: int,
    ) -> tuple[float, int, int, float]:
        if not self.config.cache_hit_enabled:
            total_access = max(0, int(num_gaussians))
            return 0.0, 0, total_access, float(num_gaussians)

        # 如果没有高斯ID信息，使用保守估计（假设无cache命中）
        if current_gaussians is None or previous_gaussians is None:
            total_access = max(0, int(num_gaussians))
            return 0.0, 0, total_access, float(num_gaussians)

        # 计算重复程度（cache命中率）
        if len(previous_gaussians) == 0:
            cache_hit_ratio = 0.0
            hit_count = 0
        else:
            intersection = current_gaussians & previous_gaussians
            cache_hit_ratio = len(intersection) / len(current_gaussians) if len(current_gaussians) > 0 else 0.0
            hit_count = len(intersection)

        total_access = len(current_gaussians) if len(current_gaussians) > 0 else max(0, int(num_gaussians))
        cache_miss_count = num_gaussians * (1.0 - cache_hit_ratio * 0.5)
        return float(cache_hit_ratio), int(hit_count), int(total_access), float(cache_miss_count)

    def get_cache_hit_ratio(self) -> float:
        if not self.config.cache_hit_enabled:
            return 0.0
        if self._cache_total_gaussians <= 0:
            return 0.0
        return float(self._cache_hit_gaussians) / float(self._cache_total_gaussians)
