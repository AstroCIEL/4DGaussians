# 4DGS ASIC Cycle-Accurate Simulator (基于事件驱动)

## 基本信息

* **simulator源码位置**：`/DISK1/home/rh_xu30/4DGaussians/simulator`
* **算法源码根目录**：`/DISK1/home/rh_xu30/4DGaussians`
* **完整创新点说明文件**：`/DISK1/home/rh_xu30/4DGaussians/simulator/4DGS行文逻辑.pdf`（重要）

该Simulator是一个基于Python的离散事件驱动（Discrete-Event Driven）的Cycle-Accurate模拟器。它专门用于验证和评估一种**面向边缘端低功耗、高帧率的创新型4DGS（4D Gaussian Splatting）渲染ASIC架构**。

在 `config.yaml` 文件中指定了数据集（如DyNeRF）、场景、帧、真实的Workload路径以及具体的硬件微架构参数（如FIFO深度、各模块周期数、SRAM大小、调度窗口K等）后，Simulator将模拟硬件管线中的并发、流水线停顿（Stall）、FIFO反压（Backpressure），并最终输出渲染该帧所需的精确周期数（Latency）、模块利用率和DRAM带宽等细节报告。

> **参考代码1**: 纯软件视角的3DGS数据流模拟器 `gscore_sw_simulator`（绝对路径：`/DISK1/home/rh_xu30/4DGaussians/gscore_sw_simulator`）。本Simulator在其基础上增加了严格的时序和硬件结构约束，并将预处理扩展到了4DGS。
> **参考代码2**: C++编写的3DGS ASIC模拟器 `Neo`（绝对路径：`/DISK1/home/rh_xu30/Neo/src/simulator/neo/src`）。本Simulator采用Python（推荐基于 `simpy` 库）重写，使用了轻量级的DRAM模型，并且在排序和调度上采用了我们论文中独创的WSLAS方案和多分辨率光栅化方案。

## 核心设计原则与技术栈

* **框架建议**：强烈建议使用 `simpy` 库，利用 `simpy.Resource` 模拟计算核心，利用 `simpy.Store` 模拟带容量限制的 FIFO 和缓存。
* **真实工作负载驱动**：Simulator 不使用随机数据，而是直接读取算法离线训练后导出的高斯模型参数（如 `.ply` 或 `.json`）、预计算的 2-bit 动静标签，以及 Foveated Rendering 的掩码分区，以保证评估的准确性。
* **模块解耦与并发**：各个硬件Engine作为独立的进程/协程运行，通过 FIFO 相互握手。

## 文件结构

* `config/`: 包含 `.yaml` 配置文件（需包含：各算子Latency设定、`NUM_RASTER_CORES`、FIFO_DEPTH、WBS `WINDOW_SIZE_K` 等）。
* `results/`: 每次仿真的性能报告、图表和日志保存目录。
* `simulator.py`: 主Simulator环境配置与顶层连接。
* `preprocess_udpe.py`: **【重点模块】** 包含 Unified Deformation-Preprocess Engine 的实现，处理动静高斯的动态路由。
* `scheduler_wbs.py`: **【重点模块】** 包含 Workload Balancing Scheduler (WSLAS) 的实现。
* `sort_hse.py`: 包含 Hierarchical Sort Engine (粗排与双调排序) 的实现。
* `rasterize_fre.py`: **【重点模块】** 包含 Foveated Rasterizing Engine 的实现，支持多分辨率降采样与插值。
* `memory.py`: 简化的DRAM带宽与延迟模型。
* `analyzer.py`: 统计总周期数、FIFO满/空时间占比、各Raster Core利用率，并生成图表。
* `main.py`: 仿真入口，负责加载 Workload 数据并启动仿真。
* `generate_labels.py`: 离线动静标签生成，自动与 simulator 集成（运行时若无标签会触发生成）。

## ASIC 数据流与微架构要求 (Cursor 核心实现参考)

Cursor在生成代码时，必须严格遵循以下三个阶段的创新硬件行为：

### 1. 预处理阶段 (UDPE: Unified Deformation-Preprocess Engine)

**非传统的固定流水线，而是基于 2-bit 标签的动态路由结构。**
给定时间步t，UDPE 的 Dispatcher 首先读取高斯的 2-bit 离线标签，并将其路由到不同的并发数据通路：

* **静止高斯 (Static, 约40%)**：直接进入 Cull Module (视锥剔除)，完成后**完全跳过** Deform Module，通过零开销旁路 (Bypass) 直接输出。
* **微动高斯 (Quasi-static, 约40%)**：先进入 Cull Module 做粗剔除 -> 进入 `FIFO_cull_to_deform` -> 存活者进入 Deform Module 计算形变。
* **巨变高斯 (Dynamic, 约20%)**：先进入 Deform Module 计算形变 -> 进入 `FIFO_deform_to_cull` -> 进入 Cull Module 做精确剔除。
* **相交测试 (Intersection Test)**：存活的高斯球在此步骤计算其覆盖的 Subtile，并将结果打包输出。
*(代码要求：必须实现 Inter-stage FIFOs。当处理速度不匹配时，FIFO 满将引发正确的反压 Stall 行为。)*

### 2. 排序与调度阶段 (HSE & WBS)

**引入了结合空间局部性与贪心负载均衡的 WSLAS (Windowed Spatial-Locality Aware Scheduling)。**

* **数据流**：UDPE 处理完成后，将 frame 拆分为 TileTask 列表，直接送入 WBS 的等待队列。
* **HSE+FRE 配对核架构**：
  - HSE 和 FRE 具有相同的核数，并一一对应形成配对核。
  - 每对核（一个 HSE 核 + 一个 FRE 核）作为一个整体，负责一个 TileTask 的完整处理流程。
  - 处理流程：HSE 核进行排序 -> FRE 核进行光栅化（同一对核内顺序执行）。
  - 不同配对核之间没有数据依赖，可以完全并行处理不同的 TileTask。
* **WBS 调度器 (关键)**：
1. **空间重排**：TileTask 列表最初按照 Hilbert 曲线或 Z-order 进行空间连续性排序进入等待队列。
2. **滑动窗口 (Sliding Window)**：维护一个大小为 K 的指令窗口 (Instruction Window)。
3. **配对核资源管理**：WBS 监视所有 HSE+FRE 配对核的工作状态。
4. **局部贪心分发 (Greedy Dispatch)**：当存在空闲的配对核时，WBS 检查窗口内 K 个 TileTask，选出 **Workload（高斯球数量）最大**的 TileTask 分配给空闲的配对核，然后窗口向前滑动补充新的 TileTask。

### 3. 多分辨率光栅化阶段 (FRE: Foveated Rasterizing Engine)

**基于 Tile 的中央凹渲染 (Foveated Rendering)，改变传统 Workload 计算。**

* 接收到 Tile 任务的 Raster Core，会首先检查该 Tile 所在的视野区域（根据配置读取）。真实画幅若无法被 `tile_size` 整除，将直接舍弃右侧/底部不足一整 tile 的区域，仅对可整除区域生成/排序/光栅化任务：
* **Fovea (中心区)**：1x 原始分辨率计算（Workload 不变）。
* **Transition (过渡区)**：2x 降采样（每个 Subtile/Tile 实际需要并行处理的像素减少一半，计算 Latency 相应缩短）。
* **Periphery (外围区)**：4x 降采样（处理像素减少至四分之一）。
* **插值重建 (Interpolation)**：对于经历过降采样的 Tile，在流水线末端需加上轻量级的插值重建周期（主要为移位和加法延迟），然后再写回 Frame Buffer。

## 使用说明（含动静标签生成）

1) 配置：编辑 `simulator/configs/default.yaml`，至少设置 `simulation.dataset` / `simulation.scene`，可选 `simulation.model_path` / `simulation.source_path`；`workload.static_ratio` / `quasi_ratio` 指定动静比例；`labeling.output_npy` / `output_json` 控制标签文件名。内存模型估计使用 `hardware.bytes_per_gaussian`（单高斯读写字节数）。

2) 运行仿真：  
```bash
python -m simulator.main --config simulator/configs/default.yaml
```
运行时会尝试读取模型目录下的标签文件（默认 `motion_labels.npy`）；若缺失则自动调用 `simulator.generate_labels` 离线生成后再继续仿真。

3) 单独生成标签（可选，手动先行）：  
```bash
python -m simulator.generate_labels --config simulator/configs/default.yaml
```
生成结果会写入模型目录并在后续仿真中复用。
