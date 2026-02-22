# simulator/memory.py
"""简化的内存系统：带宽与缓存模型，用于估算访存延迟（周期）"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """内存相关配置"""
    memory_bandwidth_gbps: float  # GB/s
    cache_size_bytes: int
    clock_frequency_ghz: float  # GHz，用于将秒转为周期


class MemorySystem:
    """简化内存系统：按带宽计算访存所需周期"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig(
            memory_bandwidth_gbps=51.2,
            cache_size_bytes=1048576,
            clock_frequency_ghz=1.0,
        )

    def configure(self, memory_bandwidth_gbps: float, cache_size_bytes: int, clock_frequency_ghz: float):
        self.config = MemoryConfig(
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            cache_size_bytes=cache_size_bytes,
            clock_frequency_ghz=clock_frequency_ghz,
        )

    def cycles_for_bytes(self, bytes_count: int) -> float:
        """给定字节数，返回在带宽限制下需要的周期数（简化：不考虑 cache 命中）。"""
        if bytes_count <= 0:
            return 0.0
        # time_sec = bytes / (bandwidth_byte_per_sec)
        # cycles = time_sec * frequency_hz = time_sec * (clock_frequency_ghz * 1e9)
        bandwidth_bps = self.config.memory_bandwidth_gbps * (1 << 30)  # GB -> bytes, per second
        time_sec = bytes_count / bandwidth_bps
        cycles = time_sec * (self.config.clock_frequency_ghz * 1e9)
        return cycles

    def stall_cycles(self, read_bytes: int = 0, write_bytes: int = 0) -> float:
        """读写总字节对应的 stall 周期（简化：读写共享带宽）。"""
        return self.cycles_for_bytes(read_bytes + write_bytes)
