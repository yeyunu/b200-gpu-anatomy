# AI Systems Hardware 可视化笔记

面向初学者的交互式硬件结构图，帮助理解 Blackwell B200，以及 CPU/GPU 的内存与缓存关系。

## 1. NVIDIA Blackwell B200 分层结构图

从四个层级解释 Blackwell B200：

1. 整块 B200：两块 GPU die、HBM3e 与 NV-HBI
2. 单块 GPU die：SM、L2 缓存、显存控制器和控制电路
3. 单个 SM：Warp 调度器、CUDA Core、Tensor Core、寄存器与 Shared Memory/L1
4. 单个 HBM3e 堆栈：8-Hi DRAM die、TSV 与容量计算

### 在线交互版

[打开交互式结构图](https://yeyunu.github.io/b200-gpu-anatomy/)

### 四层结构预览

#### 1. 整块 B200

![B200 GPU 封装结构](b200-overview.png)

#### 2. 放大 GPU die

![GPU die 内部结构](b200-die.png)

#### 3. 放大一个 SM

![SM 内部结构](b200-sm.png)

#### 4. 放大一个 8-Hi HBM3e 堆栈

![8-Hi HBM3e 堆栈结构](b200-hbm.png)

### 关键概念

- B200 物理上包含两块 GPU die，但软件通常将其识别为一个 CUDA GPU 设备。
- 两块 die 通过 NV-HBI 高速互连协同工作。
- HBM 容量表示能存放多少数据，HBM 带宽表示每秒能搬运多少数据。
- SM 是 GPU 执行线程和 CUDA kernel 的主要计算单元。
- CUDA Core 负责一般数值运算，Tensor Core 专门加速矩阵乘法和 AI 计算。
- 晶体管不是与 SM、缓存并列的功能模块；这些电路本身都由晶体管构成。
- 一个 8-Hi HBM3e 堆栈由八层 3 GB DRAM die 垂直组成，容量为 24 GB；B200 的八个堆栈合计 192 GB。

## 2. CPU/GPU 内存与缓存架构对比

[打开交互式内存架构图](https://yeyunu.github.io/b200-gpu-anatomy/cpu-gpu-memory-map.html)

这张图分三步演示变量 `x` 从初始值，到 CPU 写入，再到 GPU 读取时，不同架构如何保持数据一致：

- 传统 CPU + PCIe 独立显卡：CPU 内存和 GPU 显存各自保存数据副本，需要显式复制或同步。
- 典型统一内存 SoC：CPU 和 GPU 共享同一份物理内存，但各自仍有缓存。
- Grace Blackwell GB200：CPU 的 LPDDR5X 与 GPU 的 HBM 物理分离，通过统一地址空间和缓存一致性互连协作。

### 状态 1：初始值 `x=10`

CPU 和 GPU 都读取 `x`，因此各自附近的缓存中都有一份值为 10 的数据。

![CPU/GPU 内存架构初始状态](cpu-gpu-memory-map.png)

### 状态 2：CPU 将 `x` 写成 20

传统独立显卡仍保留旧副本；统一内存 SoC 和 GB200 会通过缓存一致性机制使 GPU 的旧缓存失效。

![CPU 写入后的内存与缓存状态](cpu-gpu-memory-map-step-2.png)

### 状态 3：GPU 再次读取 `x`

传统独立显卡需要显式复制、迁移或同步才能读到 20；统一内存 SoC 和 GB200 可以重新取得最新值。

![GPU 重新读取后的内存与缓存状态](cpu-gpu-memory-map-step-3.png)

## 3. SM、Warp、线程与内存层级

[打开六步交互图解](https://yeyunu.github.io/b200-gpu-anatomy/gpu-sm-warp-memory-explainer.html)

这组图按顺序解释 GPU 的线程执行和数据供给机制：

1. GPU、SM、Warp 与 Thread 的硬件调度层级
2. SM 内部的计算单元、调度器和片上存储
3. SIMT 执行模式与 Warp divergence
4. 多个 Warp 如何隐藏 HBM 访问延迟
5. 寄存器、Shared Memory、L1、L2 与 HBM 的层级
6. 内存受限、计算受限和数据重用

### 1. GPU → SM → Warp → Thread

![GPU、SM、Warp 与线程层级](gpu-explainer-01-hierarchy.png)

### 2. SM 内部结构

![SM 内部计算和存储单元](gpu-explainer-02-sm.png)

### 3. Warp 与 SIMT

![SIMT 与 Warp 分支发散](gpu-explainer-03-simt.png)

### 4. Warp 延迟隐藏

![多个 Warp 隐藏 HBM 访问延迟](gpu-explainer-04-latency.png)

### 5. GPU 内存层级

![寄存器到 HBM 的 GPU 内存层级](gpu-explainer-05-memory.png)

### 6. 计算受限与内存受限

![计算受限、内存受限与数据重用](gpu-explainer-06-bounds.png)

## 说明

这些图用于帮助理解硬件组成关系，并非 NVIDIA 官方芯片版图，图中部件的位置和面积不代表真实物理布局。规格数字参考《AI Systems Performance Engineering》第二章，实际产品配置和可用容量可能有所不同。
