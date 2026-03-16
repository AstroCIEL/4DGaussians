# 4DGS 加速器快速开始指南

## 1. 系统概述

4DGS 加速器是一个专用的硬件加速器，用于实时渲染 4D 高斯点。系统采用四阶段流水线架构：

```
UDPE (预处理) → WBS (调度) → HSE (排序) → FRE (渲染)
```

## 2. 快速配置

### 2.1 基本配置步骤

1. **设置相机参数** (通过 AXI4-Lite CSR 接口):
   ```c
   // 视图矩阵 (16 个 32 位寄存器, 地址 0x00-0x3C)
   write_reg(0x00, view_matrix[0]);
   write_reg(0x04, view_matrix[1]);
   // ... 写入所有 16 个矩阵元素
   
   // 投影矩阵 (16 个 32 位寄存器, 地址 0x40-0x7C)
   write_reg(0x40, proj_matrix[0]);
   // ... 写入所有 16 个矩阵元素
   
   // 时间步 (地址 0x50)
   write_reg(0x50, time_step);
   ```

2. **设置注视参数**:
   ```c
   // 注视点 X (地址 0x54)
   write_reg(0x54, gaze_x);
   
   // 注视点 Y (地址 0x58)
   write_reg(0x58, gaze_y);
   
   // 注视半径 (地址 0x5C)
   write_reg(0x5C, fovea_radius);
   ```

3. **设置屏幕分辨率**:
   ```c
   // 屏幕宽度 (地址 0x60)
   write_reg(0x60, screen_width);
   
   // 屏幕高度 (地址 0x64)
   write_reg(0x64, screen_height);
   ```

4. **设置内存地址**:
   ```c
   // 高斯数据基地址低 32 位 (地址 0x68)
   write_reg(0x68, gaussian_base_addr_low);
   
   // 高斯数据基地址高 32 位 (地址 0x6C)
   write_reg(0x6C, gaussian_base_addr_high);
   
   // 帧缓冲基地址低 32 位 (地址 0x70)
   write_reg(0x74, frame_buffer_base_addr_low);
   
   // 帧缓冲基地址高 32 位 (地址 0x78)
   write_reg(0x78, frame_buffer_base_addr_high);
   ```

5. **启动渲染**:
   ```c
   // 控制寄存器 (地址 0x7C), bit[0] = start_render
   write_reg(0x7C, 0x1);
   ```

6. **等待完成**:
   ```c
   // 状态寄存器 (地址 0x80), bit[0] = render_done
   while (!(read_reg(0x80) & 0x1)) {
       // 等待渲染完成
   }
   ```

## 3. 数据格式

### 3.1 输入数据格式 (高斯原始数据)

每个高斯原始数据包: **256 位**

```
[255:248] - 保留
[247:240] - 高斯 ID (高 8 位)
[239:232] - 高斯 ID (中 8 位)
[231:224] - 高斯 ID (低 8 位)
[223:192] - mu_z (32 位浮点)
[191:160] - mu_y (32 位浮点)
[159:128] - mu_x (32 位浮点)
[127:96]  - sigma_zz (32 位浮点)
[95:64]   - sigma_yz (32 位浮点)
[63:32]   - sigma_yy (32 位浮点)
[31:0]    - sigma_xy, sigma_xz, sigma_xx, opacity, sh_coeffs[11:0], motion_tag[1:0]
```

**运动标签编码**:
- `00`: 静态 (Static) - 跳过变形
- `01`: 准静态 (Quasi-static) - 先剔除后变形
- `10`: 动态 (Dynamic) - 先变形后剔除

### 3.2 输出数据格式 (帧缓冲)

每个像素: **32 位**

```
[31:24] - 保留 (0x00)
[23:16] - B (蓝色通道, 8 位)
[15:8]  - G (绿色通道, 8 位)
[7:0]   - R (红色通道, 8 位)
```

## 4. 关键参数

### 4.1 系统参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_SIZE | 32 | 瓦片尺寸 (32×32 像素) |
| NUM_CORES | 16 | HSE/FRE 并行核心数 |
| IQ_WINDOW_SIZE | 32 | 指令队列窗口大小 |
| BLEND_UNIT_SIZE | 8 | 混合单元阵列尺寸 (8×8) |

### 4.2 下采样率

根据瓦片到注视点的距离自动选择：

- **1x**: 无下采样 (注视区域)
- **2x**: 2×2 → 1 像素 (中等距离)
- **4x**: 4×4 → 1 像素 (远距离)

## 5. 性能指标

### 5.1 理论性能

- **时钟频率**: 1 GHz (目标)
- **内存带宽**: 512 位 AXI4 总线
- **并行度**: 16 个排序核心 + 16 个渲染核心

### 5.2 典型工作负载

- **每瓦片高斯数**: 10-1000 个
- **屏幕分辨率**: 1920×1080 (典型)
- **瓦片数**: ~2000 个 (1920×1080 / 32×32)

## 6. 调试和监控

### 6.1 状态寄存器

读取状态寄存器 (地址 0x80) 获取系统状态：

- `bit[0]`: render_done - 渲染完成标志
- `bit[1]`: udpe_busy - UDPE 忙碌
- `bit[2]`: wbs_busy - WBS 忙碌
- `bit[3]`: hse_busy - HSE 忙碌
- `bit[4]`: fre_busy - FRE 忙碌

### 6.2 常见问题

**问题**: 渲染不启动
- **检查**: 确认所有 CSR 寄存器已正确配置
- **检查**: 确认 start_render 位已设置
- **检查**: 确认复位信号已释放

**问题**: 渲染卡住
- **检查**: 检查状态寄存器，确认哪个阶段卡住
- **检查**: 确认内存地址有效
- **检查**: 确认输入数据格式正确

**问题**: 输出图像不正确
- **检查**: 确认相机矩阵正确
- **检查**: 确认注视参数合理
- **检查**: 确认输入高斯数据有效

## 7. 代码示例

### 7.1 C 语言驱动示例

```c
#include <stdint.h>

// 假设的寄存器访问函数
void write_reg(uint32_t addr, uint32_t data);
uint32_t read_reg(uint32_t addr);

void configure_4dgs_accelerator() {
    // 1. 配置相机矩阵
    float view_matrix[16] = { /* ... */ };
    for (int i = 0; i < 16; i++) {
        write_reg(0x00 + i * 4, *(uint32_t*)&view_matrix[i]);
    }
    
    float proj_matrix[16] = { /* ... */ };
    for (int i = 0; i < 16; i++) {
        write_reg(0x40 + i * 4, *(uint32_t*)&proj_matrix[i]);
    }
    
    // 2. 配置时间步
    float time_step = 0.5f;
    write_reg(0x50, *(uint32_t*)&time_step);
    
    // 3. 配置注视参数
    uint16_t gaze_x = 960;  // 屏幕中心
    uint16_t gaze_y = 540;
    uint16_t fovea_radius = 200;
    write_reg(0x54, gaze_x);
    write_reg(0x58, gaze_y);
    write_reg(0x5C, fovea_radius);
    
    // 4. 配置屏幕分辨率
    write_reg(0x60, 1920);
    write_reg(0x64, 1080);
    
    // 5. 配置内存地址
    uint64_t gaussian_base = 0x80000000;
    write_reg(0x68, (uint32_t)(gaussian_base & 0xFFFFFFFF));
    write_reg(0x6C, (uint32_t)(gaussian_base >> 32));
    
    uint64_t frame_buffer_base = 0x90000000;
    write_reg(0x74, (uint32_t)(frame_buffer_base & 0xFFFFFFFF));
    write_reg(0x78, (uint32_t)(frame_buffer_base >> 32));
    
    // 6. 启动渲染
    write_reg(0x7C, 0x1);
    
    // 7. 等待完成
    while (!(read_reg(0x80) & 0x1)) {
        // 轮询或使用中断
    }
    
    // 8. 清除完成标志
    write_reg(0x7C, 0x0);
}
```

## 8. 文件结构

```
rtl/
├── gs_types.sv              # 类型定义
├── gs_accel_top.sv          # 顶层模块
├── udpe_*.sv                # UDPE 相关模块
├── wbs_*.sv                 # WBS 相关模块
├── hse_*.sv                 # HSE 相关模块
├── fre_*.sv                 # FRE 相关模块
├── README.md                # 架构规范
├── ARCHITECTURE.md          # 详细架构文档
└── QUICK_START.md          # 本文档
```

## 9. 下一步

1. 阅读 `ARCHITECTURE.md` 了解详细架构
2. 查看 `README.md` 了解模块规范
3. 参考代码中的注释了解实现细节
4. 运行仿真验证功能
5. 进行综合和时序分析

---

**版本**: 1.0  
**最后更新**: 2024
