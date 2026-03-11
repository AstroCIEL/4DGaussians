# 4DGS 加速器 API 参考文档

## 1. 接口概述

4DGS 加速器提供两个主要接口：

1. **AXI4 Master 接口**: 用于从 DRAM 读取高斯数据和写入帧缓冲
2. **AXI4-Lite Slave 接口**: 用于配置和控制加速器

## 2. AXI4 Master 接口

### 2.1 读通道 (读取高斯数据)

**信号定义**:

| 信号名 | 方向 | 宽度 | 说明 |
|--------|------|------|------|
| m_axi_araddr | 输出 | 64 | 读地址 |
| m_axi_arlen | 输出 | 8 | 突发长度 (0-255) |
| m_axi_arsize | 输出 | 3 | 传输大小 (6 = 64 字节) |
| m_axi_arburst | 输出 | 2 | 突发类型 (01 = INCR) |
| m_axi_arvalid | 输出 | 1 | 地址有效 |
| m_axi_arready | 输入 | 1 | 地址就绪 |
| m_axi_rdata | 输入 | 512 | 读数据 |
| m_axi_rresp | 输入 | 2 | 响应 |
| m_axi_rlast | 输入 | 1 | 最后一个数据 |
| m_axi_rvalid | 输入 | 1 | 数据有效 |
| m_axi_rready | 输出 | 1 | 数据就绪 |

**使用说明**:
- 地址从 `gaussian_data_base_addr` 开始
- 每个高斯数据包为 256 位 (32 字节)
- 支持突发传输以提高效率

### 2.2 写通道 (写入帧缓冲)

**信号定义**:

| 信号名 | 方向 | 宽度 | 说明 |
|--------|------|------|------|
| m_axi_awaddr | 输出 | 64 | 写地址 |
| m_axi_awlen | 输出 | 8 | 突发长度 |
| m_axi_awsize | 输出 | 3 | 传输大小 (2 = 4 字节) |
| m_axi_awburst | 输出 | 2 | 突发类型 (01 = INCR) |
| m_axi_awvalid | 输出 | 1 | 地址有效 |
| m_axi_awready | 输入 | 1 | 地址就绪 |
| m_axi_wdata | 输出 | 512 | 写数据 |
| m_axi_wstrb | 输出 | 64 | 字节使能 |
| m_axi_wlast | 输出 | 1 | 最后一个数据 |
| m_axi_wvalid | 输出 | 1 | 数据有效 |
| m_axi_wready | 输入 | 1 | 数据就绪 |
| m_axi_bresp | 输入 | 2 | 写响应 |
| m_axi_bvalid | 输入 | 1 | 响应有效 |
| m_axi_bready | 输出 | 1 | 响应就绪 |

**使用说明**:
- 地址从 `frame_buffer_base_addr` 开始
- 每个像素为 32 位 (4 字节)
- 格式: `{8'h0, B, G, R}`

## 3. AXI4-Lite Slave 接口 (CSR)

### 3.1 接口信号

**写地址通道**:
- `s_axi_awaddr[31:0]`: 写地址
- `s_axi_awvalid`: 地址有效
- `s_axi_awready`: 地址就绪

**写数据通道**:
- `s_axi_wdata[31:0]`: 写数据
- `s_axi_wstrb[3:0]`: 字节使能
- `s_axi_wvalid`: 数据有效
- `s_axi_wready`: 数据就绪

**写响应通道**:
- `s_axi_bresp[1:0]`: 响应 (00 = OKAY)
- `s_axi_bvalid`: 响应有效
- `s_axi_bready`: 响应就绪

**读地址通道**:
- `s_axi_araddr[31:0]`: 读地址
- `s_axi_arvalid`: 地址有效
- `s_axi_arready`: 地址就绪

**读数据通道**:
- `s_axi_rdata[31:0]`: 读数据
- `s_axi_rresp[1:0]`: 响应 (00 = OKAY)
- `s_axi_rvalid`: 数据有效
- `s_axi_rready`: 数据就绪

### 3.2 寄存器映射表

| 地址偏移 | 寄存器名 | 宽度 | 访问 | 说明 |
|---------|---------|------|------|------|
| 0x00 | view_matrix[0] | 32 | R/W | 视图矩阵元素 [0,0] |
| 0x04 | view_matrix[1] | 32 | R/W | 视图矩阵元素 [0,1] |
| 0x08 | view_matrix[2] | 32 | R/W | 视图矩阵元素 [0,2] |
| 0x0C | view_matrix[3] | 32 | R/W | 视图矩阵元素 [0,3] |
| 0x10 | view_matrix[4] | 32 | R/W | 视图矩阵元素 [1,0] |
| 0x14 | view_matrix[5] | 32 | R/W | 视图矩阵元素 [1,1] |
| 0x18 | view_matrix[6] | 32 | R/W | 视图矩阵元素 [1,2] |
| 0x1C | view_matrix[7] | 32 | R/W | 视图矩阵元素 [1,3] |
| 0x20 | view_matrix[8] | 32 | R/W | 视图矩阵元素 [2,0] |
| 0x24 | view_matrix[9] | 32 | R/W | 视图矩阵元素 [2,1] |
| 0x28 | view_matrix[10] | 32 | R/W | 视图矩阵元素 [2,2] |
| 0x2C | view_matrix[11] | 32 | R/W | 视图矩阵元素 [2,3] |
| 0x30 | view_matrix[12] | 32 | R/W | 视图矩阵元素 [3,0] |
| 0x34 | view_matrix[13] | 32 | R/W | 视图矩阵元素 [3,1] |
| 0x38 | view_matrix[14] | 32 | R/W | 视图矩阵元素 [3,2] |
| 0x3C | view_matrix[15] | 32 | R/W | 视图矩阵元素 [3,3] |
| 0x40 | proj_matrix[0] | 32 | R/W | 投影矩阵元素 [0,0] |
| 0x44 | proj_matrix[1] | 32 | R/W | 投影矩阵元素 [0,1] |
| ... | ... | ... | ... | ... (共 16 个元素) |
| 0x7C | proj_matrix[15] | 32 | R/W | 投影矩阵元素 [3,3] |
| 0x50 | time_step | 32 | R/W | 当前时间步 t (IEEE 754 浮点) |
| 0x54 | gaze_x | 16 | R/W | 注视点 X 坐标 (屏幕坐标) |
| 0x58 | gaze_y | 16 | R/W | 注视点 Y 坐标 (屏幕坐标) |
| 0x5C | fovea_radius | 16 | R/W | 注视半径 (像素) |
| 0x60 | screen_width | 16 | R/W | 屏幕宽度 (像素) |
| 0x64 | screen_height | 16 | R/W | 屏幕高度 (像素) |
| 0x68 | gaussian_base_addr_low | 32 | R/W | 高斯数据基地址 [31:0] |
| 0x6C | gaussian_base_addr_high | 32 | R/W | 高斯数据基地址 [63:32] |
| 0x70 | frame_buffer_base_addr_low | 32 | R/W | 帧缓冲基地址 [31:0] |
| 0x74 | frame_buffer_base_addr_high | 32 | R/W | 帧缓冲基地址 [63:32] |
| 0x78 | control | 32 | R/W | 控制寄存器 |
| 0x7C | status | 32 | R | 状态寄存器 |

### 3.3 控制寄存器 (0x78)

| 位 | 名称 | 说明 |
|----|------|------|
| [0] | start_render | 启动渲染 (写 1 启动, 自动清零) |
| [1] | reset_stats | 重置统计计数器 |
| [2] | enable_debug | 启用调试模式 |
| [31:3] | reserved | 保留 |

### 3.4 状态寄存器 (0x7C)

| 位 | 名称 | 说明 |
|----|------|------|
| [0] | render_done | 渲染完成标志 |
| [1] | udpe_busy | UDPE 引擎忙碌 |
| [2] | wbs_busy | WBS 调度器忙碌 |
| [3] | hse_busy | HSE 排序引擎忙碌 |
| [4] | fre_busy | FRE 渲染引擎忙碌 |
| [5] | error | 错误标志 |
| [15:6] | core_status | 核心状态位图 (16 位) |
| [31:16] | reserved | 保留 |

## 4. 内部模块接口

### 4.1 UDPE 接口

**输入 FIFO**:
```systemverilog
input  logic [GAUSSIAN_DATA_WIDTH-1:0] input_fifo_data_i;
input  logic                            input_fifo_valid_i;
output logic                            input_fifo_ready_o;
```

**输出到 WBS**:
```systemverilog
output tile_workload_t                  tile_workload_o;
output logic                            tile_workload_valid_o;
input  logic                            tile_workload_ready_i;
```

### 4.2 WBS 接口

**输入从 UDPE**:
```systemverilog
input  tile_workload_t                  tile_workload_i;
input  logic                            tile_workload_valid_i;
output logic                            tile_workload_ready_o;
```

**输出到 HSE/FRE**:
```systemverilog
output wbs_task_t                       hse_task_o [0:NUM_CORES-1];
output logic [NUM_CORES-1:0]            hse_task_valid_o;
input  logic [NUM_CORES-1:0]            hse_task_ready_i;

output wbs_task_t                       fre_task_o [0:NUM_CORES-1];
output logic [NUM_CORES-1:0]            fre_task_valid_o;
input  logic [NUM_CORES-1:0]            fre_task_ready_i;
```

### 4.3 HSE 接口

**任务输入**:
```systemverilog
input  wbs_task_t                       task_i [0:NUM_CORES-1];
input  logic [NUM_CORES-1:0]            task_valid_i;
output logic [NUM_CORES-1:0]            task_ready_o;
```

**排序块输出**:
```systemverilog
output sorted_chunk_t                   sorted_chunk_o [0:NUM_CORES-1];
output logic [NUM_CORES-1:0]            sorted_chunk_valid_o;
input  logic [NUM_CORES-1:0]            sorted_chunk_ready_i;
```

### 4.4 FRE 接口

**任务输入**:
```systemverilog
input  wbs_task_t                       task_i [0:NUM_CORES-1];
input  logic [NUM_CORES-1:0]            task_valid_i;
output logic [NUM_CORES-1:0]            task_ready_o;
```

**像素输出**:
```systemverilog
output pixel_data_t                     pixel_data_o;
output logic [15:0]                     pixel_x_o;
output logic [15:0]                     pixel_y_o;
output logic                            pixel_valid_o;
input  logic                            pixel_ready_i;
```

## 5. 数据类型定义

### 5.1 gaussian_primitive_t

```systemverilog
typedef struct packed {
    logic [31:0] mu_x, mu_y, mu_z;        // 均值位置
    logic [31:0] sigma_xx, sigma_xy, sigma_xz;
    logic [31:0] sigma_yy, sigma_yz, sigma_zz;
    logic [7:0]  opacity;
    logic [95:0] sh_coeffs;
    logic [1:0]  motion_tag;
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
} gaussian_primitive_t;
```

### 5.2 deformed_gaussian_t

```systemverilog
typedef struct packed {
    logic [31:0] mu_x, mu_y, mu_z;
    logic [31:0] sigma_xx, sigma_xy, sigma_xz;
    logic [31:0] sigma_yy, sigma_yz, sigma_zz;
    logic [7:0]  opacity;
    logic [95:0] sh_coeffs;
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
    logic [DEPTH_WIDTH-1:0] depth;
    logic        visible;
} deformed_gaussian_t;
```

### 5.3 tile_workload_t

```systemverilog
typedef struct packed {
    deformed_gaussian_t gaussian;
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [15:0] workload_estimate;
} tile_workload_t;
```

### 5.4 wbs_task_t

```systemverilog
typedef struct packed {
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [15:0] workload;
    logic [1:0]  downsample_rate;
    logic [GAUSSIAN_ID_WIDTH-1:0] start_gaussian_id;
    logic [GAUSSIAN_ID_WIDTH-1:0] num_gaussians;
} wbs_task_t;
```

### 5.5 sorted_chunk_t

```systemverilog
typedef struct {
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_ids [0:15];
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [4:0] num_valid;
} sorted_chunk_t;
```

### 5.6 pixel_data_t

```systemverilog
typedef struct packed {
    logic [7:0] r, g, b;
    logic [7:0] alpha;
    logic [DEPTH_WIDTH-1:0] depth;
} pixel_data_t;
```

## 6. 握手协议

### 6.1 Ready/Valid 协议

所有数据接口使用 Ready/Valid 握手协议：

- **发送方**: 驱动 `valid` 信号表示数据有效
- **接收方**: 驱动 `ready` 信号表示可以接受数据
- **传输**: 发生在 `valid && ready` 的时钟上升沿

**规则**:
1. `valid` 一旦置高，必须保持直到 `ready` 置高
2. `ready` 可以独立于 `valid` 变化
3. 不允许 `valid` 和 `ready` 同时置低后 `valid` 先置高

### 6.2 流水线握手示例

```
时钟周期:  0    1    2    3    4    5
valid:    __/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
ready:    ‾‾‾‾‾‾‾‾‾‾‾‾\__/‾‾‾‾‾‾‾‾‾‾
传输:           ✓              ✓
```

## 7. 错误处理

### 7.1 AXI 错误响应

- `rresp/bresp = 2'b00`: OKAY - 正常传输
- `rresp/bresp = 2'b01`: EXOKAY - 独占访问成功
- `rresp/bresp = 2'b10`: SLVERR - 从设备错误
- `rresp/bresp = 2'b11`: DECERR - 解码错误

### 7.2 错误标志

状态寄存器的 `error` 位在以下情况置位：

- AXI 传输错误
- 无效的配置参数
- 内存地址越界
- 数据格式错误

## 8. 性能调优

### 8.1 内存访问优化

- 使用 AXI 突发传输最大化带宽
- 对齐内存地址到 64 字节边界
- 预取高斯数据到本地缓存

### 8.2 工作负载平衡

- WBS 自动平衡瓦片工作负载
- 确保所有核心充分利用
- 监控核心利用率

### 8.3 功耗优化

- 启用时钟门控
- 使用早期终止减少计算
- 动态调整下采样率

## 9. 调试接口

### 9.1 调试模式

设置控制寄存器的 `enable_debug` 位启用调试模式：

- 输出详细的内部状态
- 记录性能计数器
- 输出调试信息到日志

### 9.2 性能计数器

（如果实现）可以通过 CSR 读取：

- 处理的瓦片数
- 处理的高斯数
- 内存访问次数
- 核心利用率

---

**版本**: 1.0  
**最后更新**: 2024
