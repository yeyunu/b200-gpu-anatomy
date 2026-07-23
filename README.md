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

## 3. SM、Warp、线程与内存层级：64 个数的具体并行计算

[打开八步交互式并行计算图](https://yeyunu.github.io/b200-gpu-anatomy/gpu-sm-warp-memory-explainer.html)

这组图把计算过程和数据位置同时展开。教学例子使用一个包含 64 个线程的 Block，对 HBM 中的 `1…64` 求和，最终得到 `2080`。64 个线程组成两个 Warp；示例线程数用于讲解，并非 Blackwell 的固定配置。

八个步骤顶部使用同一张固定位置图：HBM 始终画在 GPU die 外，L2 画在 GPU die 内但 SM 外，L1、寄存器、Shared Memory 和加法单元始终画在 SM 内。每一步只改变高亮位置和数据值，便于连续追踪数据在哪里。

1. `Thread tid` 与 HBM 地址 `x[tid]` 一一对应
2. 每个线程并行读取一个数，保存到自己的寄存器
3. 64 个线程显式写入 Shared Memory 的 64 个槽位
4. Warp 0 的 32 条 Lane 同时完成第一轮 32 次加法
5. `stride` 从 16 降到 1，逐轮得到全部中间值
6. 对照表展示每个阶段的数据实际存放位置
7. Warp 时间线展示怎样用其他工作覆盖 HBM 等待
8. 汇总 64 次读取、63 次加法和 1 次写回

### 1. Thread ID 与输入数据分工

![64 个线程与 HBM 中 64 个输入元素的对应关系](gpu-parallel-01-mapping.png)

### 2. 数据并行读入线程私有寄存器

![两个 Warp 的 64 个线程分别保存自己的寄存器值](gpu-parallel-02-registers.png)

### 3. 数据显式写入 Shared Memory

![寄存器值与 Shared Memory 地址和值的对应关系](gpu-parallel-03-shared-storage.png)

### 4. 第一轮 32 路并行加法

![一个 Warp 的 32 条 Lane 同时执行 32 个具体加法](gpu-parallel-04-round32.png)

### 5. 后续各轮的具体计算结果

![stride 从 16 到 1 的每轮活跃线程、算式和中间值](gpu-parallel-05-rounds.png)

### 6. 每个阶段的数据存放位置

![HBM、寄存器、Shared Memory 和输出在各阶段保存的数据](gpu-parallel-06-storage-map.png)

### 7. Warp 调度与延迟隐藏时间线

![一个 Warp 等待 HBM 时另一个 Warp 继续工作的时间线](gpu-parallel-07-warp-timeline.png)

### 8. 完整并行计算过程

![从 HBM 读取到片上并行归约再写回 HBM 的完整过程](gpu-parallel-08-overview.png)


## 4. 第二章后半章：网络、机架运营与路线图

[打开 19 节交互式图册](https://yeyunu.github.io/b200-gpu-anatomy/chapter2-visual-atlas.html)

图册按书中的小节顺序拆成三组。图中的事实、数字和比较均从对应小节正文提取；视觉布局与“记忆主线”是为了帮助理解而进行的重组，不额外补充正文没有给出的产品规格。

### A. 超大规模网络

#### 01. Ultrascale Networking Treating Many GPUs as One

![把 72 个 GPU 组织成一台机架级超级计算机](ch2-01-ultrascale-networking.png)

#### 02. NVLink and NVSwitch

![NVLink 与 NVSwitch 的连接关系](ch2-02-nvlink-nvswitch.png)

#### 03. Multi-GPU Programming

![多 GPU 编程中的地址、同步与硬件路径](ch2-03-multi-gpu-programming.png)

#### 04. In-Network Aggregations with NVIDIA SHARP

![SHARP 在交换网络中完成聚合](ch2-04-nvidia-sharp.png)

#### 05. Multirack and Storage Communication

![多机架、DPU、NIC 与共享存储通信](ch2-05-multirack-storage.png)

#### 06. Preintegrated Rack Appliance

![NVL72 预集成机架的组成与现场接入](ch2-06-preintegrated-rack.png)

#### 07. Co-Packaged Optics: Future of Networking Hardware

![共封装光学缩短交换芯片到光引擎的电路径](ch2-07-co-packaged-optics.png)

### B. 机架运营

#### 08. Compute Density and Power Requirements

![NVL72 的约 130 kW 功耗构成与冗余供电](ch2-08-power-density.png)

#### 09. Liquid Cooling Versus Air Cooling

![冷板、机架水路、CDU 与设施水路组成的液冷闭环](ch2-09-liquid-vs-air-cooling.png)

#### 10. Performance Monitoring and Utilization in Practice

![沿等待链排查 GPU 利用率、网络、温度与功耗](ch2-10-monitoring-utilization.png)

#### 11. Sharing and Scheduling

![SLURM 或 Kubernetes 的 GPU 分配与单卡 MIG 切分](ch2-11-sharing-scheduling.png)

#### 12. ROI of Upgrading Your Hardware

![用总拥有成本而非单卡价格判断硬件升级回报](ch2-12-upgrade-roi.png)

### C. NVIDIA 路线图与本章总结

#### 13. A Glimpse into the Future: NVIDIA's Roadmap

![从 Blackwell 到 Feynman 的书中路线图](ch2-13-nvidia-roadmap.png)

#### 14. Blackwell Ultra and Grace Blackwell Ultra

![B200 与 B300 的书中规格对比](ch2-14-blackwell-ultra.png)

#### 15. Vera Rubin Superchip (2026)

![Vera CPU 与两块 Rubin GPU 的超级芯片构想](ch2-15-vera-rubin.png)

#### 16. Rubin Ultra and Vera Rubin Ultra (2027)

![书中对四 die Rubin Ultra 与更大 NVL 系统的展望](ch2-16-rubin-ultra.png)

#### 17. Feynman GPU (2028) and Doubling Something Every Year

![把 Feynman 的远期趋势与未确认规格分开阅读](ch2-17-feynman.png)

#### 18. Key Takeaways

![第二章的六层系统检查表](ch2-18-key-takeaways.png)

#### 19. Conclusion

![从芯片、超级芯片、机架、基础设施到有效工作的协同设计](ch2-19-conclusion.png)

## 说明

这些图用于帮助理解硬件组成关系，并非 NVIDIA 官方芯片版图，图中部件的位置和面积不代表真实物理布局。规格数字参考《AI Systems Performance Engineering》第二章，实际产品配置和可用容量可能有所不同。Vera Rubin、Rubin Ultra 与 Feynman 图中的远期数字按书中措辞标为路线图、预期或推测，不代表已经发布的最终产品规格。
