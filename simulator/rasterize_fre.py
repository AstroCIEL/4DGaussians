from dataclasses import dataclass
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class FREConfig:
    num_cores: int = 16
    base_cycles_per_gaussian: float = 2.0
    interpolation_cycles: float = 8.0
    early_stop_ratio: float = 0.3 


class FoveatedRasterEngine:
    """多分辨率光栅化引擎，基于 tile 区域调整工作量。"""

    def __init__(self, env: simpy.Environment, config: FREConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.resource = simpy.Resource(env, capacity=config.num_cores)

    def has_free_core(self) -> bool:
        return self.resource.count < self.resource.capacity

    def _region_scale(self, region: str) -> float:
        if region == "transition":
            return 0.5
        if region == "periphery":
            return 0.25
        return 1.0

    def raster_cycles(self, task: TileTask) -> float:
        scale = self._region_scale(task.region)
        core_cycles = task.num_gaussians * self.config.early_stop_ratio * self.config.base_cycles_per_gaussian * scale
        interp = self.config.interpolation_cycles/(1.0 - scale) if scale < 1.0 else 0.0
        return core_cycles + interp

    def process(self, task: TileTask):
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        with self.resource.request() as req:
            yield req
            cycles = self.raster_cycles(task)
            self.analyzer.record_busy("fre", cycles)
            yield self.env.timeout(cycles)
            # 处理完成，由 WBS 通过 _process_pair 管理完成回调
