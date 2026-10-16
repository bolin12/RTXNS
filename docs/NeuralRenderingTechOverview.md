# 神经渲染与神经外观技术体系全景指南

## 一、底层工程基建：RTX Neural Shading (RTXNS) 架构

RTX Neural Shading 是 NVIDIA 开源的底层基础 SDK，旨在打通现代 GPU 硬件张量单元与实时图形着色器之间的调用屏障，实现在 Shader 内部直接执行微型神经网络前向推理与在线反向传播训练。

* 硬件接口与加速单元：依赖现代 GPU 的专用张量硬件核心（Tensor Core / MMA），通过 DirectX 12 Agility SDK 预览版的 Cooperative Vector 扩展（Shader Model 6.9/6.10）及 Vulkan 的 VK_NV_cooperative_vector 扩展，直接在着色器内执行微型矩阵乘加运算。
* 跨语言与编译架构：基于 Slang 着色语言与 SlangPy 工具链，实现从 Python 端算法原型快速迭代、着色器自动微分（AutoDiff）、反向传播优化器（Adam/SGD 等），到最终统一生成 DXIL/SPIR-V 硬件机器码。
* 构建体系与工程结构：由 CMake 进行全平台管理，支持 Ninja 及 Visual Studio 2022 解决方案生成，跨平台配置由 CMakePresets.json 驱动。
* 核心示例与定位：
  * SimpleInferencing：预加载训练收敛的 Disney BRDF 材质权重，演示运行时纯前向推理与物理渲染 Ground Truth 的实时比对。
  * SimpleTraining：在 Compute Shader 中利用反向传播实时在线拟合 2D 纹理，直观展示网络从纯噪波到收敛的全过程。
  * ShaderTraining：利用 Slang 的自动微分特性，在 GPU 上对高维 BRDF 着色器进行在线反向传播训练。
  * SlangpyTraining / SlangpyInferencing：展示在 Python 环境中并行训练多种候选网络拓扑，并将收敛权重导出为二进制文件供 C++ 运行时直接调用的跨语言工程闭环。

## 二、神经着色的数学本质与工程权衡

### 1. 拟合对象与数学本质
物理材质模型（BRDF/BSDF）本质上是一套确定性的高维非线性映射函数：
输入包含微表面几何法线与光线夹角（N·L, N·V, N·H, L·H 等）及材质属性（粗糙度、金属度、各向异性参数），输出为光线能量散射与菲涅尔反射系数。
神经材质利用通用近似定理，将传统着色器中复杂的微表面遮蔽（Smith 项）、法线分布函数（GGX/NDF）、多层介质内折射积分等高开销数值解算，转化为由微型多层感知机（MLP）拟合的高维连续查找表（Continuous LUT）。

### 2. 训练机制的工程选型
* 运行时实时训练：仅适用于场景间接光、动态辐射场等无法预先穷举的不可预测信号（如 NRC 神经辐射缓存）。材质和 Shader 拟合因其物理规律在局部完全闭合，不需要也不应该在游戏运行时实时训练。
* 离线训练与静态推理：工业界落地标准。在资产导入期或自动化 CI/CD 管线中离线训练网络，收敛后将几 KB 的参数固化，运行时在像素着色器中以 O(1) 的固定开销执行纯前向推理。

### 3. 存储与算力收益边界
* 简单 Shader 无需使用：简单的 Lambertian 漫反射或单层 GGX 仅消耗极少 ALU 周期，引入神经网络会徒增寄存器压力和寻址开销。
* 降维打击高维 LUT：对于 4 维至 5 维以上的复杂物理模型或实测材质（Measured Materials），传统多维查找表贴图往往占用数十至上百 MB 显存，且存在走样问题；微型网络仅需几 KB 参数即可高保真复现，显存压缩比可达数千倍。
* 消除 Shader 变种膨胀：固定矩阵运算结构的微型网络作为通用解算器，仅通过切换不同权重即可表现截然不同的材质，避免传统引擎中多层材质排列组合引发的变种爆炸。
* 突破传统实时光学计算瓶颈：使得传统实时管线无法承受的物理模型（如多波长连续光谱色散渲染、薄膜干涉与波动光学、复杂多层车漆清漆与毛发微纤维散射）能够在恒定算力预算内实时呈现。

## 三、神经渲染三支柱与全管线协同

在实时图形学向全神经渲染演进的过程中，形成了相互协同的三大支柱技术：

1. 神经纹理压缩 (Neural Texture Compression, NTC)
* 作用域：空间域 (u, v)。
* 解决瓶颈：解决 4K/8K 超高分辨率纹理对显存容量与显存读取带宽的吞吐瓶颈。
* 机制：在片元着色器采样时直接调用硬件张量算子就地解码，以远超传统 BC7 块压缩的压缩比保留极高的微观纹理细节。

2. 神经材质 (Neural Materials / Neural BSDF)
* 作用域：方向角向域 (ωi, ωo)。
* 解决瓶颈：解决多层物理材质在着色器层面的 ALU 计算复杂度和分支开销。
* 机制：使用微型 MLP 替代多层 Substrate / MaterialX 材质图的复杂光照评估与出射光线重要性采样。

3. 神经辐射缓存 (Neural Radiance Cache, NRC)
* 作用域：三维空间与光线路径域 (x, ω)。
* 解决瓶颈：解决实时光线追踪全局光照中多跳漫反射间接光积分的采样数爆炸问题。
* 机制：在运行时每帧自适应微调网络权重，在空间域直接预测未采样光线路径的间接光能量，消除传统屏幕空间时域滤波（Temporal Filter）产生的拖影与滞后。

三者的全量协同演示：NVIDIA 官方基于虚幻引擎 5 打造的完整路径追踪演示项目《Zorah》，首次在同一实时场景中全部开启了 NTC、Neural Materials、NRC、Mega Geometry 与 DLSS，验证了全管线神经化协同渲染的可行性。

## 四、学术前沿演进与代表性论文

近两三年顶级图形学会议（SIGGRAPH / SIGGRAPH Asia）聚焦于用数据驱动和神经方法解决极限物理材质渲染问题：

1. 工业级多层材质实时烘焙
* 代表成果：Real-Time Neural Appearance Models (SIGGRAPH 2024, NVIDIA 团队)
* 核心突破：将影视级复杂多层材质（MaterialX 与 MDL）烘焙为分层潜空间纹理与神经解码器，在实时路径追踪中实现 12 到 24 倍的材质评估性能加速，并开源了完整研究代码库 NVlabs/neuralappearance。

2. 几何视差与边缘轮廓保持
* 代表成果：Towards Comprehensive Neural Materials: Dynamic Structure-Preserving Synthesis with Accurate Silhouette at Instant Inference Speed (SIGGRAPH 2025, UCSB 闫令琪/王璐/王贝贝团队)
* 核心突破：针对以往神经材质只能表达扁平光照的局限，提出 INT8 量化微型网络与动态两步位移追踪（Dynamic Displacement Tracing），在极高推理帧率下实现了真实的微观几何凹凸自遮挡与边缘轮廓剪影（Silhouette）。

3. 任意微观几何免数学推导拟合
* 代表成果：PureSample: Neural Materials Learned by Sampling Microgeometry (SIGGRAPH 2026, 南京大学 / Adobe 团队)
* 核心突破：终结了繁琐的解析公式推导。直接在微观几何表面执行前向光线随机游走采样，采用流匹配（Flow Matching）生成式模型拟合出射概率分布，配合微型 MLP 预测视线相关反照率，实现了对任意复杂微观几何的高保真神经拟合。

4. 神经资产生成与 AIGC 对接
* 代表成果：Toward Richer Material Generation via Procedural Data Enhancement (SIGGRAPH 2026, NVIDIA 团队)
* 定位辨析：将多层材质紧凑编码进共享潜空间，并对接视频扩散模型（Video Diffusion Model）进行潜纹理直接生成。该方法定位为制作期 AIGC 资产生成工具，生成后的资产仍导入传统引擎进行物理交互与实时光线追踪，并非取代游戏引擎。

## 五、商业引擎落地现实：对标 Nanite/Lumen 与跨平台局限

1. Nanite 与 Lumen 的工业现状与痛点
* 性能下限与多端割裂：主机（PS5/XSX 性能约等同 RTX 2070 Super）与主流 PC 算力吃紧。Nanite 和 Lumen 的基础固定开销极高，难以在多端稳定保住 60 帧。
* 局限场景明显：Nanite 早期对骨骼动画、植被 WPO 和半透明材质支持成本高，海量几何体流送极度依赖高速 NVMe SSD；Lumen 严重依赖跨帧时域累积，镜头快速运动或动态打光突变时伴随明显的噪波与鬼影。
* 工业分化：追求极致影视沉浸感的 3A 单机选择开启，而竞技类、动作类、MMO 及全平台跨端项目多数坚守传统 LOD、CSM 阴影与离线烘焙/探针方案。

2. 神经方案的优势与落地的致命伤
* 理论超越点：NRC 依靠空间参数化推理避免了 Lumen 的时域累积鬼影；神经位移与外观模型规避了 Nanite 处理超高频微观几何时的显存带宽膨胀和摩尔纹走样。
* 致命的硬件断代：神经渲染高度绑定 GPU 专有张量硬件（Tensor Core / Wave Matrix）。现役主机（PS5 / Xbox Series X 均采用 AMD RDNA 2 架构）在底层硬件上完全缺失着色器张量单元，移动端更无原生支持。开启神经方案意味着商业项目必须维护双套资产与编译管线，这是大多数商业化跨平台项目无法承受的成本。

3. Nanite 的神经几何对应流派
* 神经几何场（Neural Geometry Fields, SIGGRAPH 2024）：以粗糙网格为基底，通过坐标网络隐式编码微观几何形变，网格压缩率可达 50 倍以上。
* 融合趋势（宏观显式 + 微观神经）：未来演进更倾向于 Nanite 负责大尺度宏观拓扑与中远景流送，配合底层 RTX Mega Geometry 加速硬件光追 BVH 构建；而微观亚毫米级几何、微纤维凹凸与复杂多层光照，交由神经材质与微观位移追踪接管。

## 六、神经外观 (Neural Appearance) 全景架构

1. 概念层级关系
神经外观不是与神经纹理、神经材质并列的概念，而是“顶层大一统体系与底层组成模块”的层级关系：
* 神经纹理（NTC）：负责空间域 (u, v)，以隐式潜向量压缩存储空间位置属性与特征。
* 神经材质（Neural BSDF）：负责方向角向域 (ωi, ωo)，以微型 MLP 解算微表面物理散射与出射重要性采样。
* 神经外观（Neural Appearance）：顶层完整外观系统。将神经纹理的潜变量提取、神经材质的角向物理求解，与微观几何法线框架和阴影先验联合打包，光线击中表面时单次求值输出最终高保真视觉响应。

2. 工业研发进阶路径建议
* 第一阶段（单点切入）：优先在引擎中跑通独立的神经材质求值（如基于 RTXNS 验证 Cooperative Vector 底层算子与 BRDF 拟合）或独立的神经纹理压缩（NTC）。模块职责单一，能平稳嵌入现有引擎管线并快速验证收益。
* 第二阶段（体系演进）：在底层张量硬件接口、着色器编译桥接和离线训练导出流程成熟后，再推进到空间纹理与方向角向联合解算的完整神经外观系统，实现对高复杂度物理外观资产的工业级实时闭环。
