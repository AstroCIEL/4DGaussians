from dataclasses import dataclass


@dataclass
class MemoryConfig:
    memory_bandwidth_gbps: float = 51.2
    cache_size_bytes: int = 1_048_576
    clock_frequency_ghz: float = 1.0
    read_latency_hiding_rate: float = 0.5


class MemorySystem:
    """极简内存模型：根据访问字节数估算延迟与带宽占用。"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.bytes_per_cycle = (
            config.memory_bandwidth_gbps * 1e9 / (config.clock_frequency_ghz * 1e9)
        )

    def estimate_cycles(self, bytes_accessed: float) -> float:
        """返回因带宽限制产生的额外周期。"""
        if self.bytes_per_cycle <= 0:
            return 0.0
        return max(0.0, bytes_accessed / self.bytes_per_cycle) * (1 - self.config.read_latency_hiding_rate)

    def estimate_bytes_for_gaussians(self, num_gaussians: int, bytes_per_gaussian: int) -> float:
        """简单估计：每个高斯读写若干字节。"""
        return float(num_gaussians * max(0, bytes_per_gaussian))
