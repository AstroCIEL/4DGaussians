# simluator 说明

## 基本信息

- simulator源码位置：/DISK1/home/rh_xu30/4DGaussians/simulator
- 算法源码根目录：/DISK1/home/rh_xu30/4DGaussians

该simulator模拟了asic硬件执行4DGS渲染的行为，在config文件中指定了数据集、场景、帧以及其他架构参数后，将会输出asic渲染该帧或指定多帧所需要的周期数（即延时）及其他细节信息。

> 参考代码1: 一个只模拟了软件行为（非硬件视角，仅模拟软件端数据流）的3DGS渲染asic的simulator即gscore可供参考，绝对路径为：/DISK1/home/rh_xu30/4DGaussians/gscore_sw_simulator。事实上，本simulator其实就是gscore的硬件模拟版本，且预处理阶段从3DGS的预处理变为4DGS的预处理，而排序和光栅化的处理方法与gscore一样。

> 参考代码2: 一个cpp编写的3DGS asic simulator即Neo，绝对路径是/DISK1/home/rh_xu30/Neo/src/simulator/neo/src。他的排序算法比较复杂，并且还使用ramulator来仿真与dram的交互。本simulator使用python编写，且使用简化的dram模型，并且排序也使用了与neo不同的方案。但是光栅化部分基本是一致的。

该simulator使用离散事件模拟，基于事件触发状态更新。设计原则：

- 事件驱动：使用事件队列模拟并发
- 模块化：每个阶段独立可配置
- 可扩展：易于添加新算法和硬件模型
- 可视化：内置性能分析和可视化

## 文件结构

- config文件夹：其中yaml文件是每一次仿真所需要的全部配置信息
- results文件夹：每次仿真的结果保存目录
- simulator.py: 主simulator类定义
- event.py: 事件类定义
- preprocess.py: preprocess模块类定义
- sort.py: sort模块类定义
- rasterize.py: rasterize模块定义
- memory.py: memory系统模块定义
- analyzer.py: 性能分析、报告与可视化类
- main.py: 主程序

## asic数据流

该asic用于渲染4DGS，整个渲染过程分为三个阶段：preprocess，sort，rasterize。给定一个scene的已训练的高斯球模型，以及一个要渲染的view和时间步t，第一步先对所有高斯球送入preprocess engine进行预处理。然后在sort engine中像gscore那样进行二阶段排序，然后在rasterizing engine按照chunk去进行光栅化。每个阶段具体的操作如下：

- preprocess：取出每个高斯的xyz，结合t，去过一个hexplane，然后过一个mlp，得到形变后的xyz，s，q。然后根据形变后的xyz先做view变换，然后做frustum culling，得到该帧将会处理的高斯球列表。然后做intersection test，算出每个高斯球影响到的subtile（subtile大小在config中指定）。
- sort：接下来开始tile之间互相独立处理，可并行。对于每一个tile内部：第一阶段是粗排，指定一些pivot，将高斯根据深度分成不同的chunk。第二阶段是精细排序，依次对这些chunk内部的高斯球根据深度排序。
- rasterize：依然tile互相并行。对于每一个tile内部，一旦第一个chunk的精细排序结束，则可以立马开始光栅化。rasterizing engine包含若干个unit（在config中指定数量），每个unit负责一个tile，一个chunk的完整光栅化过程。每个unit中还有若干个subunit，每个subunit将负责一个subtile一个chunk的完整渲染过程。

由于simulator是算法与硬件的桥梁，为了更加准确的模拟实际负载，应使用算法中的训练好的高斯模型，直接获取真实的workload进行模拟，而不是随机平均分配。

