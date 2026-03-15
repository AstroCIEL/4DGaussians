# SRAM 替换需求分析报告

## 概述
本报告分析了设计中需要替换为 SRAM 的大寄存器数组。通常，超过 1-2 KB 的数组应该使用 SRAM compiler 生成的 SRAM 来替代寄存器实现。

## 类型大小计算

### deformed_gaussian_t
- mu_x, mu_y, mu_z: 3 × 32 = 96 bits
- sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz: 6 × 32 = 192 bits
- opacity: 8 bits
- sh_coeffs: 96 bits
- gaussian_id: 20 bits
- depth: 24 bits
- visible: 1 bit
- **总计: 437 bits ≈ 55 bytes**

### pixel_data_t
- r, g, b: 3 × 8 = 24 bits
- alpha: 8 bits
- depth: 24 bits
- **总计: 56 bits = 7 bytes**

## 需要替换为 SRAM 的大数组

### 🔴 高优先级（必须替换）

#### 1. fre_bc.sv - feature_buffer
```systemverilog
deformed_gaussian_t feature_buffer [0:1023];  // Max Gaussians per tile
```
- **大小**: 1024 × 437 bits = 447,488 bits ≈ **55.9 KB**
- **位置**: `fre_bc.sv:36`
- **用途**: 存储每个瓦片的高斯特征数据
- **访问模式**: 顺序读写，有 write_ptr 和 read_ptr
- **建议**: **必须使用 SRAM**，这是最大的单个数组

#### 2. fre_interp.sv - boundary_cache_h/v
```systemverilog
pixel_data_t boundary_cache_h [0:1023];  // Horizontal boundaries
pixel_data_t boundary_cache_v [0:1023];  // Vertical boundaries
```
- **大小**: 2 × (1024 × 56 bits) = 114,688 bits ≈ **14 KB** (总共)
- **位置**: `fre_interp.sv:31-32`
- **用途**: 存储跨瓦片边界的像素缓存
- **访问模式**: 基于 tile_id 的随机访问
- **建议**: **建议使用 SRAM**，两个缓存可以合并为一个双端口 SRAM

### 🟡 中优先级（建议替换）

#### 3. hse_core.sv - gaussian_ids 和 depths
```systemverilog
logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_ids [0:1023];  // Max Gaussians per tile
logic [DEPTH_WIDTH-1:0] depths [0:1023];
```
- **大小**: 
  - gaussian_ids: 1024 × 20 bits = 20,480 bits ≈ **2.5 KB**
  - depths: 1024 × 24 bits = 24,576 bits ≈ **3 KB**
  - **总计: 5.5 KB**
- **位置**: `hse_core.sv:31-32`
- **用途**: 存储高斯 ID 和深度值用于排序
- **访问模式**: 排序算法中的随机访问
- **建议**: **建议使用 SRAM**，特别是如果有多个核心实例

#### 4. wbs_load_gen.sv - tile_workloads 等数组
```systemverilog
logic [15:0] tile_workloads [0:1023];  // Assuming max 1024 tiles
logic [GAUSSIAN_ID_WIDTH-1:0] tile_gaussian_counts [0:1023];
logic [GAUSSIAN_ID_WIDTH-1:0] tile_start_ids [0:1023];
logic tile_valid [0:1023];
```
- **大小**: 
  - tile_workloads: 1024 × 16 bits = 16,384 bits ≈ **2 KB**
  - tile_gaussian_counts: 1024 × 20 bits = 20,480 bits ≈ **2.5 KB**
  - tile_start_ids: 1024 × 20 bits = 20,480 bits ≈ **2.5 KB**
  - tile_valid: 1024 × 1 bit = 1,024 bits ≈ **0.125 KB**
  - **总计: 7.125 KB**
- **位置**: `wbs_load_gen.sv:30-33`
- **用途**: 存储每个瓦片的工作负载统计信息
- **访问模式**: 基于 tile_id 的随机访问
- **建议**: **建议使用 SRAM**，可以合并为一个多字段 SRAM

### 🟢 低优先级（可以保留为寄存器）

#### 5. 小数组（< 2 KB）
以下数组较小，可以保留为寄存器实现：
- `wbs_iq.sv` - iq_buffer [0:31]: 32 × wbs_task_t ≈ **1.5 KB**
- `wbs_arbiter.sv` - workloads [0:31]: 32 × 16 bits ≈ **64 bytes**
- `udpe_deform.sv` - weight_buffer [0:63]: 64 × 16 bits ≈ **128 bytes**
- `udpe_deform.sv` - feature_buffer [0:15]: 16 × 32 bits ≈ **64 bytes**
- `fre_bc.sv` - pixel_buffer [8×8]: 64 × pixel_data_t ≈ **448 bytes**
- `fre_bc.sv` - transmittance/alpha/color arrays [8×8]: 多个小数组，总计 < 1 KB

## 替换建议

### 优先级 1: fre_bc.sv - feature_buffer
**最紧急**，55.9 KB 的数组会导致综合失败或面积爆炸。

**建议实现**:
- 使用单端口或双端口 SRAM
- 深度: 1024
- 宽度: 437 bits (或对齐到 512 bits)
- 访问: 支持 read_ptr 和 write_ptr 的独立访问

### 优先级 2: fre_interp.sv - boundary_cache
**高优先级**，14 KB 的缓存。

**建议实现**:
- 使用双端口 SRAM
- 深度: 1024
- 宽度: 56 bits (或对齐到 64 bits)
- 可以合并 h 和 v 缓存为一个 SRAM，使用地址高位区分

### 优先级 3: hse_core.sv - 排序数组
**中优先级**，5.5 KB，但如果有 16 个核心实例，总大小会很大。

**建议实现**:
- 每个核心使用独立的 SRAM
- 深度: 1024
- 宽度: 44 bits (20 + 24，可以合并为一个 SRAM)

### 优先级 4: wbs_load_gen.sv - tile 统计数组
**中优先级**，7.125 KB。

**建议实现**:
- 使用单端口 SRAM
- 深度: 1024
- 宽度: 57 bits (16 + 20 + 20 + 1，对齐到 64 bits)

## 实施步骤

1. **识别 SRAM 接口需求**
   - 确定每个数组的读写端口需求
   - 确定访问模式（顺序/随机）
   - 确定时钟域

2. **生成 SRAM 宏**
   - 使用 SRAM compiler (如 TSMC SRAM Compiler)
   - 生成符合工艺的 SRAM 宏
   - 确保时序和功耗符合要求

3. **替换 RTL 代码**
   - 将数组声明替换为 SRAM 实例化
   - 添加 SRAM 控制逻辑（地址、使能、写使能等）
   - 处理 SRAM 的延迟（通常 1-2 个时钟周期）

4. **验证**
   - 功能验证
   - 时序验证
   - 面积和功耗分析

## 注意事项

1. **SRAM 延迟**: SRAM 通常有 1-2 个时钟周期的读取延迟，需要调整控制逻辑
2. **写使能**: SRAM 需要明确的写使能信号
3. **初始化**: SRAM 通常不能像寄存器那样在复位时初始化，需要额外的初始化逻辑
4. **多端口**: 如果需要同时读写，需要使用双端口 SRAM
5. **面积优化**: 合并相关数组可以减少 SRAM 实例数量

## 总结

**必须立即替换**:
- `fre_bc.sv::feature_buffer` (55.9 KB) ⚠️

**强烈建议替换**:
- `fre_interp.sv::boundary_cache_h/v` (14 KB)
- `hse_core.sv::gaussian_ids/depths` (5.5 KB × 核心数)
- `wbs_load_gen.sv::tile_*` 数组 (7.125 KB)

**总计需要替换**: 约 **82+ KB** (不考虑核心实例化)
