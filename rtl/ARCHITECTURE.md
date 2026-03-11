# 4DGS 加速器架构文档

## 1. 项目概述

本项目实现了一个专用的 4D 高斯点渲染（4D Gaussian Splatting）硬件加速器，采用 ASIC 设计。该加速器通过以下三种关键技术优化内存带宽和计算延迟：

- **MTDS (Motion-Aware Tagging and Deformation Skipping)**: 运动感知标记和变形跳过
- **WSLAS (Windowed Spatial-Locality Aware Scheduling)**: 窗口化空间局部性感知调度
- **MAFR (Motion-aware Foveated Rendering)**: 运动感知焦点渲染

### 1.1 系统特性

- **目标时钟频率**: 1GHz
- **并行核心数**: 16 个 HSE 核心 + 16 个 FRE 核心
- **瓦片尺寸**: 32×32 像素
- **子瓦片尺寸**: 4×4 像素
- **指令队列窗口大小**: K=32
- **混合单元阵列**: 8×8，每个单元 4 个并行通道

### 1.2 技术指标

- **总线宽度**: 512 位 AXI4 数据总线
- **地址宽度**: 64 位
- **高斯数据宽度**: 256 位
- **运动标签宽度**: 2 位
- **深度宽度**: 24 位
- **颜色宽度**: 24 位（RGB 各 8 位）

## 2. 系统架构

### 2.1 顶层架构

系统采用四阶段流水线架构：

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  UDPE   │ --> │   WBS   │ --> │   HSE   │ --> │   FRE   │
│ 引擎    │     │ 调度器  │     │ 排序引擎│     │ 渲染引擎│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
    ↓               ↓               ↓               ↓
  DRAM          瓦片负载        深度排序        帧缓冲输出
```

### 2.2 数据流

1. **输入阶段**: 从 LPDDR4 DRAM 读取高斯原始数据（带 2 位运动标签）
2. **预处理阶段 (UDPE)**: 
   - 根据运动标签路由高斯数据
   - 执行变形计算（动态高斯）
   - 执行视锥剔除
   - 计算屏幕空间特征和瓦片交集
3. **调度阶段 (WBS)**:
   - 估计瓦片工作负载
   - 应用 MAFR 下采样率（1x/2x/4x）
   - 使用希尔伯特曲线序列生成任务
   - 仲裁并分发任务到空闲核心
4. **排序阶段 (HSE)**:
   - 16 个并行核心执行深度排序
   - 快速排序（粗排序）+ 双调排序（细排序）
5. **渲染阶段 (FRE)**:
   - 16 个并行混合核心执行 α 混合
   - 双线性插值重构下采样像素
   - 输出最终像素到帧缓冲

## 3. 模块详细说明

### 3.1 UDPE (Unified Deformation-Preprocess Engine)

统一变形-预处理引擎，集成变形计算和预处理功能以最大化硬件并行度。

#### 3.1.1 udpe_dispatcher.sv

**功能**: 解析调度器，根据 2 位运动标签路由高斯数据。

**运动标签编码**:
- `00` (Static): 完全跳过变形，直接路由到剔除单元
- `01` (Quasi-static): 先路由到剔除单元，如果可见则再变形
- `10` (Dynamic): 先路由到变形单元，然后到剔除单元

**接口**:
- 输入: `input_fifo_data_i`, `input_fifo_valid_i`
- 输出到变形单元: `deform_data_o`, `deform_valid_o`
- 输出到剔除单元: `cull_data_o`, `cull_valid_o`

#### 3.1.2 udpe_deform.sv

**功能**: 变形单元，执行轻量级神经网络更新高斯参数。

**组件**:
- 权重缓存接口
- 脉动阵列（MAC 操作）
- 特征缓冲区

**处理流程**:
1. 提取特征（位置 + 时间步）
2. 从权重缓存获取权重
3. 执行 MAC 运算计算增量
4. 应用增量到规范高斯参数

#### 3.1.3 udpe_culling.sv

**功能**: 剔除单元，计算相机空间坐标并评估视锥可见性。

**处理流程**:
1. 矩阵-向量乘法：`camera_pos = view_matrix * world_pos`
2. 视锥剔除检查（6 个平面）
3. 深度计算
4. 可见性标志设置

**输出**:
- 可见的高斯数据到交集单元
- 准静态路径的可见高斯到变形单元

#### 3.1.4 udpe_intersect.sv

**功能**: 2D 特征和交集单元，计算屏幕空间特征和瓦片交集。

**处理流程**:
1. 投影变换：`clip_pos = proj_matrix * camera_pos`
2. 透视除法
3. 视口变换到屏幕坐标
4. 瓦片交集计算
5. 工作负载估计

**输出**: 更新的高斯属性和相交瓦片 ID 到 WBS

### 3.2 WBS (Workload Balancing Scheduler)

工作负载平衡调度器，物理实现 WSLAS 算法。

#### 3.2.1 wbs_load_gen.sv

**功能**: 负载生成器，估计计算工作负载并应用 MAFR 下采样率。

**特性**:
- 基于高斯数量估计工作负载
- 根据瓦片到注视点的偏心距应用下采样率（1x/2x/4x）
- 按希尔伯特曲线序列输出任务

**下采样率决策**:
- 偏心距 < 注视半径: 1x（无下采样）
- 偏心距 < 2×注视半径: 2x 下采样
- 其他: 4x 下采样

#### 3.2.2 wbs_iq.sv

**功能**: 指令队列，存储大小为 K=32 的活动候选窗口。

**实现**: 循环缓冲区（FIFO）

**接口**:
- 输入: 来自负载生成器的任务
- 输出: 任务窗口数组（32 个任务）到仲裁器

#### 3.2.3 wbs_arbiter.sv

**功能**: 瓦片仲裁器，组合比较器树快速识别最大工作负载的瓦片。

**算法**: 递归比较器树查找最大值

**输出**: 选中的任务（最大工作负载）和索引

#### 3.2.4 wbs_dispatcher.sv

**功能**: 任务调度器，监控 HSE/FRE 核心的就绪/空闲状态。

**调度策略**:
- 监控 16 个 HSE 核心和 16 个 FRE 核心的状态
- 优先级编码器查找第一个可用核心对
- 将最重的瓦片从仲裁器分发到可用核心

### 3.3 HSE (Hierarchical Sorting Engine)

分层排序引擎，对分发瓦片内的高斯进行深度排序。

#### 3.3.1 hse_core.sv

**功能**: HSE 核心，执行两阶段排序过程。

**排序算法**:
1. **快速排序**: 粗排序阶段
   - 分区逻辑
   - 枢轴选择
2. **双调排序**: 细排序阶段
   - 双调合并网络
   - 多阶段排序

**本地存储**:
- 高斯 ID 数组（最多 1024 个）
- 深度值数组

**输出**: 排序后的索引块到对应的 FRE 核心

#### 3.3.2 hse_top.sv

**功能**: HSE 顶层模块，实例化 16 个并行排序核心。

**特性**:
- 16 个并行核心
- 每个核心独立处理一个瓦片
- 时钟门控支持（功耗优化）

### 3.4 FRE (Foveated Rasterizing Engine)

焦点渲染引擎，执行焦点 α 混合并输出最终像素值。

#### 3.4.1 fre_bc.sv

**功能**: 混合核心，8×8 混合单元阵列。

**微架构**:
- 每个单元包含 4 个并行通道
- α 计算
- 透射率更新（带早期终止逻辑）
- 颜色混合（4 个独立像素）

**本地 SRAM**: 每个核心专用的特征缓冲区，存储当前分配瓦片的高斯属性

**处理流程**:
1. 从外部内存加载高斯特征
2. 对每个高斯，计算对 8×8 瓦片内每个像素的贡献
3. 执行 α 混合
4. 早期终止（透射率 < 阈值）

#### 3.4.2 fre_interp.sv

**功能**: 插值重构器，执行双线性插值重构省略的像素。

**功能**:
- 基于焦点下采样率（1x/2x/4x）的双线性插值
- 缓存瓦片边界像素以消除跨瓦片接缝伪影
- 重构后写入输出缓冲区

**插值算法**:
- 1x: 无插值
- 2x: 2×2 → 1 像素插值
- 4x: 4×4 → 1 像素插值

#### 3.4.3 fre_top.sv

**功能**: FRE 顶层模块，实例化 16 个并行混合核心和共享插值单元。

**特性**:
- 16 个并行混合核心
- 共享插值重构器
- 时钟门控支持

### 3.5 gs_accel_top.sv

**功能**: 顶层系统模块，实例化所有四个主要引擎并处理 AXI4 事务。

#### 3.5.1 AXI4 Master 接口

**读通道**:
- 从 DRAM 读取高斯参数
- 地址: `gaussian_data_base_addr`
- 突发长度: 可配置

**写通道**:
- 写入最终帧缓冲到 DRAM
- 地址: `frame_buffer_base_addr`
- 数据格式: RGB (24 位)

#### 3.5.2 AXI4-Lite Slave 接口 (CSR)

**寄存器映射**:

| 地址偏移 | 寄存器 | 说明 |
|---------|--------|------|
| 0x00-0x3F | view_matrix[0:15] | 相机视图矩阵（16×32 位） |
| 0x40-0x4F | proj_matrix[0:15] | 相机投影矩阵（16×32 位） |
| 0x50 | time_step | 当前时间步 t（32 位） |
| 0x54 | gaze_x | 注视点 X 坐标（16 位） |
| 0x58 | gaze_y | 注视点 Y 坐标（16 位） |
| 0x5C | fovea_radius | 注视半径（16 位） |
| 0x60 | screen_width | 屏幕宽度（16 位） |
| 0x64 | screen_height | 屏幕高度（16 位） |
| 0x68-0x6B | gaussian_data_base_addr | 高斯数据基地址（64 位） |
| 0x6C-0x6F | frame_buffer_base_addr | 帧缓冲基地址（64 位） |
| 0x70 | control | 控制寄存器（start_render 位） |
| 0x74 | status | 状态寄存器（render_done 位） |

## 4. 数据类型定义

### 4.1 高斯原始结构 (gaussian_primitive_t)

```systemverilog
typedef struct packed {
    logic [31:0] mu_x, mu_y, mu_z;        // 均值位置（3D）
    logic [31:0] sigma_xx, sigma_xy, sigma_xz;  // 协方差矩阵元素
    logic [31:0] sigma_yy, sigma_yz, sigma_zz;
    logic [7:0]  opacity;                  // 基础不透明度
    logic [95:0] sh_coeffs;                // 球谐系数（12 个系数 × 8 位）
    logic [1:0]  motion_tag;               // 运动标签
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
} gaussian_primitive_t;
```

### 4.2 变形后高斯结构 (deformed_gaussian_t)

```systemverilog
typedef struct packed {
    logic [31:0] mu_x, mu_y, mu_z;        // 变形后的均值位置
    logic [31:0] sigma_xx, sigma_xy, sigma_xz;
    logic [31:0] sigma_yy, sigma_yz, sigma_zz;
    logic [7:0]  opacity;
    logic [95:0] sh_coeffs;
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
    logic [DEPTH_WIDTH-1:0] depth;         // 相机空间深度
    logic        visible;                  // 视锥可见性标志
} deformed_gaussian_t;
```

### 4.3 瓦片工作负载结构 (tile_workload_t)

```systemverilog
typedef struct packed {
    deformed_gaussian_t gaussian;
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [15:0] workload_estimate;       // 此高斯的估计工作负载
} tile_workload_t;
```

### 4.4 WBS 任务结构 (wbs_task_t)

```systemverilog
typedef struct packed {
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [15:0] workload;
    logic [1:0]  downsample_rate;
    logic [GAUSSIAN_ID_WIDTH-1:0] start_gaussian_id;
    logic [GAUSSIAN_ID_WIDTH-1:0] num_gaussians;
} wbs_task_t;
```

## 5. 接口协议

### 5.1 Ready/Valid 握手协议

所有模块间使用严格的 Ready/Valid 握手协议：

- `valid`: 发送方指示数据有效
- `ready`: 接收方指示可以接受数据
- 数据传输发生在 `valid && ready` 的时钟上升沿

### 5.2 流水线握手

```
UDPE → WBS → HSE → FRE
```

每个阶段之间都有握手信号，确保数据不会丢失。

## 6. 时钟和复位

### 6.1 时钟域

- **主时钟**: `clk` (目标 1GHz)
- **复位**: `rst_n` (低电平有效)

### 6.2 时钟门控

为功耗优化，在以下位置插入时钟门控：

- HSE 核心：等待 WBS 分发任务时
- FRE 核心：等待 WBS 分发任务时
- UDPE：无数据时

**注意**: 实际时钟门控需要使用库单元实现。

## 7. 参数化配置

所有关键参数在 `gs_types.sv` 中定义，可配置：

- `TILE_SIZE`: 瓦片尺寸（默认 32）
- `SUBTILE_SIZE`: 子瓦片尺寸（默认 4）
- `NUM_CORES`: 并行核心数（默认 16）
- `IQ_WINDOW_SIZE`: 指令队列窗口大小（默认 32）
- `BLEND_UNIT_SIZE`: 混合单元阵列尺寸（默认 8）
- `DATA_WIDTH`: AXI 数据宽度（默认 512）
- `GAUSSIAN_DATA_WIDTH`: 高斯数据宽度（默认 256）

## 8. 使用说明

### 8.1 初始化流程

1. **配置 CSR 寄存器**:
   - 设置相机视图矩阵和投影矩阵
   - 设置时间步 t
   - 设置注视参数（gaze_x, gaze_y, fovea_radius）
   - 设置屏幕分辨率
   - 设置高斯数据基地址和帧缓冲基地址

2. **启动渲染**:
   - 写入控制寄存器，设置 `start_render` 位

3. **等待完成**:
   - 轮询状态寄存器，检查 `render_done` 位

### 8.2 数据格式

**输入数据格式** (从 DRAM):
- 每个高斯原始数据: 256 位
- 包含: 位置、协方差、不透明度、球谐系数、运动标签、高斯 ID

**输出数据格式** (到帧缓冲):
- 每个像素: 32 位 (RGB + Alpha)
- 格式: `{8'h0, B, G, R}`

### 8.3 性能优化建议

1. **内存访问优化**:
   - 使用 AXI 突发传输最大化带宽
   - 预取高斯数据到本地缓存

2. **工作负载平衡**:
   - WBS 自动平衡瓦片工作负载
   - 确保所有核心充分利用

3. **功耗优化**:
   - 启用时钟门控
   - 使用早期终止减少不必要的计算

## 9. 文件结构

```
rtl/
├── gs_types.sv              # 类型定义和参数
├── gs_accel_top.sv          # 顶层模块
├── udpe_top.sv              # UDPE 顶层
├── udpe_dispatcher.sv       # UDPE 调度器
├── udpe_deform.sv           # UDPE 变形单元
├── udpe_culling.sv          # UDPE 剔除单元
├── udpe_intersect.sv        # UDPE 交集单元
├── wbs_top.sv               # WBS 顶层
├── wbs_load_gen.sv         # WBS 负载生成器
├── wbs_iq.sv                # WBS 指令队列
├── wbs_arbiter.sv           # WBS 仲裁器
├── wbs_dispatcher.sv        # WBS 调度器
├── hse_top.sv               # HSE 顶层
├── hse_core.sv              # HSE 核心
├── fre_top.sv               # FRE 顶层
├── fre_bc.sv                # FRE 混合核心
├── fre_interp.sv            # FRE 插值单元
├── README.md                # 架构规范
└── ARCHITECTURE.md          # 本文档
```

## 10. 验证和测试建议

### 10.1 单元测试

- 每个子模块独立测试
- 测试各种运动标签路径
- 测试边界情况（空瓦片、单高斯等）

### 10.2 集成测试

- 端到端数据流测试
- 多核心并发测试
- 性能基准测试

### 10.3 性能分析

- 内存带宽利用率
- 核心利用率
- 流水线停顿分析
- 功耗分析

## 11. 未来改进方向

1. **更高效的排序算法**: 优化快速排序和双调排序实现
2. **更智能的调度**: 改进 WBS 调度策略
3. **更好的缓存**: 增加多级缓存层次
4. **动态电压频率调节**: 根据工作负载调整频率和电压
5. **错误检测和纠正**: 添加 ECC 支持

## 12. 参考文献

- 4DGS 论文: `xrh_iccad2026_0311.pdf`
- AXI4 协议规范
- SystemVerilog IEEE 1800 标准

---

**文档版本**: 1.0  
**最后更新**: 2024  
**作者**: 4DGS Accelerator Design Team
