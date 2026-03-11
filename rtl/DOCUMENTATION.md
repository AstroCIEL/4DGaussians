# 4DGS 加速器项目文档索引

欢迎使用 4DGS 加速器项目文档。本文档提供了所有可用文档的索引和导航。

## 📚 文档列表

### 1. [README.md](README.md)
**架构规范文档**  
原始的架构设计规范，定义了模块边界、功能和接口。这是实现的基础文档。

**适合**: 架构师、硬件工程师  
**内容**: 
- 系统概述
- 模块规范
- 接口定义
- 实现指导原则

---

### 2. [ARCHITECTURE.md](ARCHITECTURE.md)
**详细架构文档**  
完整的系统架构说明，包括所有模块的详细功能、数据流、接口协议等。

**适合**: 硬件工程师、软件工程师、验证工程师  
**内容**:
- 系统架构概述
- 四阶段流水线详细说明
- 每个模块的功能和实现
- 数据类型定义
- 接口协议
- 参数化配置
- 使用说明
- 文件结构

**推荐阅读顺序**: 在阅读 README.md 后阅读本文档

---

### 3. [QUICK_START.md](QUICK_START.md)
**快速开始指南**  
快速上手指南，帮助您快速配置和使用加速器。

**适合**: 软件工程师、系统集成工程师  
**内容**:
- 基本配置步骤
- 数据格式说明
- 关键参数
- 性能指标
- 调试和监控
- C 语言代码示例

**推荐阅读顺序**: 如果您想快速开始使用，先阅读本文档

---

### 4. [API_REFERENCE.md](API_REFERENCE.md)
**API 参考文档**  
完整的接口和寄存器参考文档。

**适合**: 软件工程师、驱动开发者  
**内容**:
- AXI4 Master 接口详细说明
- AXI4-Lite Slave 接口 (CSR)
- 完整的寄存器映射表
- 数据类型定义
- 握手协议
- 错误处理
- 性能调优

**推荐阅读顺序**: 开发驱动或软件时参考本文档

---

## 🗂️ 文件结构

```
rtl/
├── README.md                # 架构规范
├── ARCHITECTURE.md          # 详细架构文档
├── QUICK_START.md           # 快速开始指南
├── API_REFERENCE.md         # API 参考文档
├── DOCUMENTATION.md          # 本文档（文档索引）
│
├── gs_types.sv              # 类型定义和参数
├── gs_accel_top.sv          # 顶层模块
│
├── udpe_top.sv              # UDPE 顶层
├── udpe_dispatcher.sv       # UDPE 调度器
├── udpe_deform.sv           # UDPE 变形单元
├── udpe_culling.sv          # UDPE 剔除单元
├── udpe_intersect.sv        # UDPE 交集单元
│
├── wbs_top.sv               # WBS 顶层
├── wbs_load_gen.sv          # WBS 负载生成器
├── wbs_iq.sv                # WBS 指令队列
├── wbs_arbiter.sv           # WBS 仲裁器
├── wbs_dispatcher.sv        # WBS 调度器
│
├── hse_top.sv               # HSE 顶层
├── hse_core.sv              # HSE 核心
│
├── fre_top.sv               # FRE 顶层
├── fre_bc.sv                # FRE 混合核心
└── fre_interp.sv            # FRE 插值单元
```

## 🎯 根据角色选择文档

### 硬件架构师
1. 阅读 [README.md](README.md) 了解整体架构
2. 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解详细设计
3. 查看 RTL 代码了解实现细节

### 硬件工程师
1. 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解模块设计
2. 查看 RTL 代码进行实现和优化
3. 参考 [API_REFERENCE.md](API_REFERENCE.md) 了解接口规范

### 软件工程师 / 驱动开发者
1. 阅读 [QUICK_START.md](QUICK_START.md) 快速上手
2. 参考 [API_REFERENCE.md](API_REFERENCE.md) 开发驱动
3. 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解数据流

### 验证工程师
1. 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解功能
2. 参考 [API_REFERENCE.md](API_REFERENCE.md) 了解接口协议
3. 查看 RTL 代码编写测试用例

### 系统集成工程师
1. 阅读 [QUICK_START.md](QUICK_START.md) 了解配置
2. 参考 [API_REFERENCE.md](API_REFERENCE.md) 了解接口
3. 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解系统行为

## 📖 推荐阅读路径

### 路径 1: 快速开始（适合软件开发者）
```
QUICK_START.md → API_REFERENCE.md → 开始开发
```

### 路径 2: 深入理解（适合硬件开发者）
```
README.md → ARCHITECTURE.md → RTL 代码 → API_REFERENCE.md
```

### 路径 3: 完整学习（适合新团队成员）
```
README.md → ARCHITECTURE.md → QUICK_START.md → API_REFERENCE.md → RTL 代码
```

## 🔍 文档内容概览

### README.md
- ✅ 系统概述
- ✅ 模块规范
- ✅ 接口定义
- ✅ 实现指导

### ARCHITECTURE.md
- ✅ 详细架构说明
- ✅ 四阶段流水线
- ✅ 模块功能详解
- ✅ 数据类型定义
- ✅ 接口协议
- ✅ 参数化配置
- ✅ 使用说明
- ✅ 文件结构
- ✅ 验证建议
- ✅ 未来改进方向

### QUICK_START.md
- ✅ 快速配置步骤
- ✅ 数据格式说明
- ✅ 关键参数
- ✅ 性能指标
- ✅ 调试和监控
- ✅ C 语言代码示例

### API_REFERENCE.md
- ✅ AXI4 接口详细说明
- ✅ CSR 寄存器映射
- ✅ 数据类型定义
- ✅ 握手协议
- ✅ 错误处理
- ✅ 性能调优

## 📝 文档更新记录

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.0 | 2024 | 初始版本，包含所有核心文档 |

## 🤝 贡献指南

如果您发现文档中的错误或需要补充的内容，请：

1. 检查现有文档是否已涵盖相关内容
2. 确定需要更新的文档
3. 提供清晰的修改建议

## 📞 获取帮助

- 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解系统工作原理
- 查看 [QUICK_START.md](QUICK_START.md) 解决常见问题
- 查看 [API_REFERENCE.md](API_REFERENCE.md) 了解接口细节

## 🔗 相关资源

- **论文**: `xrh_iccad2026_0311.pdf`
- **代码**: `rtl/*.sv` 文件
- **脚本**: `scripts/` 目录（如果存在）

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: 4DGS Accelerator Design Team
