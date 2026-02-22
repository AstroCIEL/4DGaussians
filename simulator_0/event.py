# simulator/event.py
"""离散事件模拟：事件类型与处理逻辑"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable


class EventType(Enum):
    """事件类型枚举"""
    # 阶段边界
    SIM_START = "sim_start"
    PREPROCESS_START = "preprocess_start"
    PREPROCESS_DONE = "preprocess_done"
    SORT_START = "sort_start"
    TILE_COARSE_SORT_DONE = "tile_coarse_sort_done"
    TILE_CHUNK_FINE_SORT_DONE = "tile_chunk_fine_sort_done"
    TILE_ALL_CHUNKS_READY = "tile_all_chunks_ready"
    SORT_DONE = "sort_done"
    RASTERIZE_CHUNK_START = "rasterize_chunk_start"
    RASTERIZE_CHUNK_DONE = "rasterize_chunk_done"
    RASTERIZE_TILE_DONE = "rasterize_tile_done"
    FRAME_DONE = "frame_done"
    # 内存/调度
    MEMORY_ACCESS = "memory_access"
    PHASE_COMPLETE = "phase_complete"
    TILE_READY = "tile_ready"
    RENDER_START = "render_start"


@dataclass
class Event:
    """通用事件：带时间戳、类型与负载数据"""
    event_type: EventType
    timestamp: float = 0.0
    frame_id: Optional[int] = None
    tile_id: Optional[int] = None
    chunk_id: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def handle(self, simulator):
        """由 Simulator 在对应时间戳调用"""
        if self.callback:
            self.callback(simulator, self)
        else:
            self.default_handler(simulator)

    def default_handler(self, simulator):
        """默认处理：转发给 Simulator 的全局事件分发"""
        simulator.dispatch_event(self)


def make_event(event_type: EventType, frame_id: Optional[int] = None,
               tile_id: Optional[int] = None, chunk_id: Optional[int] = None,
               data: Optional[Dict[str, Any]] = None, callback: Optional[Callable] = None) -> Event:
    return Event(
        event_type=event_type,
        frame_id=frame_id,
        tile_id=tile_id,
        chunk_id=chunk_id,
        data=data or {},
        callback=callback,
    )
