# TalkingGaussian Baseline Enhacement

## 11. 当前项目相对于 TalkingGaussian 的两个主要创新点

当前项目保留 TalkingGaussian 原有的 **Face–Mouth Decomposition**：Face branch 负责头部、
上半脸和整体面部形变，Mouth branch 独立负责口腔内部 Gaussian 的位置形变，最后通过同一个
3D Gaussian Rasterizer 合成。本文的修改包含一个由两个子模块组成的在线 Face branch 方法，
以及一个离线数据处理方法：

1. **Query-Guided Scale-Aware Face Modulation（QGSFM）**

   该创新点由两个前后串联的子模块组成。**Spatial Query-Guided Audio–AU Fusion（SQAF）**
   使用三组可学习 plane query 建立 XY、YZ、XZ 三平面的空间模板，对音频条件和上半脸 AU
   条件进行逐位置门控融合；**Gaussian-Scale-Aware Modulation（GSAM）** 再将融合特征扩展为
   `8×8`、`16×16`、`32×32` 三尺度调制图，并根据每个 Gaussian 的静态空间特征和
   canonical scale 自适应混合调制尺度。二者共同构成一个完整的 Face branch 条件调制方法。

2. **Canonical Feature Head Stabilizer（CFHS）**

   在 3DMM 粗跟踪之后、生成 `transforms_train.json` 和 `transforms_val.json` 之前，从可靠训练帧建立人物专属 canonical 特征模板，让每帧直接对齐同一模板，并通过连续置信度与全序列 SE(3) 时间约束修正相机姿态。

后文使用 `QGSFM` 表示完整的“查询引导尺度感知人脸调制”，使用 `SQAF` 和 `GSAM` 指代其中
的条件融合与尺度路由子模块；`CFHS` 表示“Canonical 特征头部稳定器”。

第一个创新点 QGSFM 参与 Face motion network 的训练和推理；第二个创新点 CFHS 属于数据
预处理，只离线生成更稳定的相机外参，不作为训练时的额外神经网络。

### 整体流程图

```text
                         Offline data preprocessing
┌───────────────────────────────────────────────────────────────────────────┐
│ Reference video → 3DMM coarse tracking → CFHS canonical alignment       │
│                                                        │                  │
│                                                        ▼                  │
│                                      refined camera poses R*, T*          │
│                                                        │                  │
│                                                        ▼                  │
│                              transforms_train.json / transforms_val.json  │
└───────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                               Shared camera and projection
                                                         │
                  ┌──────────────────────────────────────┴──────────────┐
                  │                                                     │
                  ▼                                                     ▼
        Face Gaussian field                                  Mouth Gaussian field
                  │                                                     │
                  ▼                                                     ▼
   learned plane queries                               Original mouth motion field
   → SQAF audio/AU fusion
   → GSAM scale-aware modulation
                  │                                                     │
                  ▼                                                     ▼
       Δxyz, Δrotation, Δscale                                  Δxyz (mouth)
                  │                                                     │
                  └──────────────────────┬──────────────────────────────┘
                                         ▼
                              3DGS rasterization + fusion
                                         ▼
                                  synthesized frame
```

---

## 12. 创新点一：Query-Guided Scale-Aware Face Modulation（QGSFM）

QGSFM 将 SQAF 和 GSAM 视为同一调制链路的两个组成部分：SQAF 解决异构驱动条件如何映射到
三平面空间，GSAM 解决融合后的空间条件如何适配不同 Gaussian 的表征尺度。两者共享中间特征
`F16` 并联合端到端训练，因此论文中应作为一个完整创新点提出，而不是两套独立方法。

### 12.1 组成部分一：Spatial Query-Guided Audio–AU Fusion（SQAF）

#### 12.1.1 要解决的问题

Face branch 同时接收语音和上半脸 AU。两类条件的作用区域和物理含义不同：语音主要驱动与发音
相关的下半脸运动，AU 则提供眨眼、眉眼和上半脸表情。如果只把两个全局向量直接拼接到每个
Gaussian 的特征上，网络需要同时学习“条件是什么”和“条件应当作用在哪里”，容易产生空间上
过于一致的响应，也难以抑制音频与 AU 在无关区域的干扰。

当前方法使用可学习的三平面 query 作为空间载体。音频和 AU 分别形成候选特征，门控网络在
每个 plane 的每个空间位置选择两类条件的相对贡献，随后生成具有明确空间布局的调制图。

#### 12.1.2 可学习 Plane Query

模块为三个 plane 建立共享于所有帧的可学习参数：

```text
plane_query: [1,3,32,16,16]
```

其中 `3` 对应 XY、YZ、XZ，`32` 是 query channel，`16×16` 是基础空间分辨率。它可以理解为
三张可训练的 canonical plane template：参数本身学习稳定的空间组织方式，当前帧的音频和 AU
则在该模板上产生条件偏移。

音频和 AU 分别投影成每个 plane 的通道偏置：

```text
enc_a        [1,32] → projector → audio_bias [1,3,32,1,1]
enc_e_global [1, 6] → projector → upper_bias [1,3,32,1,1]
```

通过广播加法得到两类条件化 query：

```text
audio_query = plane_query + audio_bias
upper_query = plane_query + upper_bias
```

这种设计没有直接让全局条件预测完整调制图，而是让条件在共享空间模板上改变响应，从而保留
跨帧一致的三平面坐标语义。

#### 12.1.3 空间门控融合

两类条件化 query 分别经过卷积得到候选特征：

```text
audio_candidate: [3,32,16,16]
upper_candidate: [3,32,16,16]
```

门控网络将基础 query 和两类候选特征拼接：

```text
concat(query, audio_candidate, upper_candidate): [3,96,16,16]
                              │
                              ▼
                     gate_conv + softmax
                              │
                              ▼
                     gate: [3,2,16,16]
```

这里的两路 softmax 在每个 plane、每个像素位置归一化：

```text
gate_audio(p) + gate_upper(p) = 1

F16(p) = gate_audio(p) × audio_candidate(p)
       + gate_upper(p) × upper_candidate(p)
```

因此 gate 不是一组全局音频/AU 权重，而是三平面上的空间变化权重。融合结果
`F16: [3,32,16,16]` 随当前帧条件变化，并作为后续固定尺度或多尺度 gamma/beta 调制图的共享
生成特征。

#### 12.1.4 SQAF 子模块流程图

```text
DeepSpeech context                         Upper-face AUs
       │                                         │
       ▼                                         ▼
 AudioNet + temporal attention               AU encoder
       │ enc_a [1,32]                           │ enc_e [1,6]
       ▼                                         ▼
 audio projector                            upper projector
       │ audio_bias                             │ upper_bias
       └──────────────┐          ┌──────────────┘
                      ▼          ▼
             learned plane_query [1,3,32,16,16]
                      │          │
          query + audio_bias    query + upper_bias
                      │          │
                      ▼          ▼
               audio Conv     upper Conv
                      │          │
                      └────┬─────┘
                           ▼
          concat(base query, audio candidate, AU candidate)
                           │
                    spatial gate + softmax
                           │
                           ▼
             gate_audio(p), gate_upper(p)
                           │
                           ▼
             gated spatial feature F16
                           │
                           ▼
                  gamma/beta modulation maps
```

#### 12.1.5 QGSFM 内部串联关系

SQAF 回答“音频与 AU 如何在三平面空间中分配”，输出统一的 `F16`；GSAM 回答“每个
Gaussian 应从哪种空间分辨率读取调制”，从 `F16` 构建多尺度金字塔并进行 Gaussian 级路由。
两者前后相接、共享中间表示并接受同一重建目标监督，共同构成 QGSFM。

---

### 12.2 组成部分二：Gaussian-Scale-Aware Modulation（GSAM）

#### 12.2.1 要解决的问题

原来的空间调制只生成固定分辨率的 `16×16` 调制图。对所有 Gaussian 使用同一空间分辨率，
无法显式区分以下需求：

- 大 Gaussian 或平滑区域更适合低分辨率调制，以表达稳定的低频运动；
- densification 后的小 Gaussian、眼睑和嘴角等区域需要更高分辨率调制，以保留局部变化；
- 不同 Gaussian 的空间尺寸和各向异性不同，对调制尺度的需求也不应完全相同。

因此，当前 Face branch 在保持原有 HashGrid、注意力和 deformation MLP 不变的前提下，新增
多尺度调制金字塔和 Gaussian 级尺度路由器。

#### 12.2.2 多尺度调制金字塔

音频与 AU 首先按照第 5 节所述方式生成共享特征：

```text
DeepSpeech → AudioNet + AudioAttNet ─┐
                                     ├─→ audio/AU gated fusion → F16
Upper-face AUs → AU encoder ─────────┘
```

其中：

```text
F16: [3, 32, 16, 16]
```

这里的 `3` 对应 XY、YZ、XZ 三个 plane。当前实现不训练三套完全独立的生成器，而是从同一个
`F16` 构建三尺度特征：

```text
                         stride-2 Conv
                       ┌────────────────→ F8  [3,32, 8, 8]
                       │
F16 [3,32,16,16] ──────┼────────────────→ F16 [3,32,16,16]
                       │
                       └─ bilinear upsample + Conv
                                         → F32 [3,32,32,32]
```

每个尺度分别经过 `1×1 Conv: 32 → 24`，输出 12 个 hash channel 的 gamma 和 beta：

```text
M8:  [1,3,2,12, 8, 8]
M16: [1,3,2,12,16,16]
M32: [1,3,2,12,32,32]
```

三种调制图共享上游 audio/AU 条件和 `F16` 特征，因此实验变量主要是空间分辨率，而不是三套
互不相关的调制网络。

#### 12.2.3 Gaussian 尺度描述符

尺度路由使用 Gaussian 模型中的 canonical/base scale，而不是 deformation MLP 当前帧预测的
`d_scale`。渲染器将下面的数据传入 MotionNetwork：

```text
pc.get_scaling.detach(): [N,3]
```

对每个 Gaussian 的三个正尺度 `s_x, s_y, s_z` 取对数：

```text
log_s = log(clamp(s, 1e-8))                         [N,3]
mean_s = mean(log_s, dim=-1)                        [N,1]
q = [mean_s, log_s_x-mean_s, log_s_y-mean_s,
     log_s_z-mean_s]                                [N,4]
```

`mean_s` 表达 Gaussian 的总体大小，后三维表达三个轴相对于均值的各向异性。`detach()` 保证
路由损失不会通过该输入直接改变 Gaussian 的基础尺度。

#### 12.2.4 Gaussian-Level Scale Router

路由器输入由静态三平面 HashGrid 特征和尺度描述符组成：

```text
static enc_x [N,36] ─┐
                     ├─ concat [N,40] → Linear 40→32 → ReLU
scale q      [N, 4] ─┘                         → Linear 32→3 → Softmax
                                                               │
                                                               ▼
                                                    W [N,3]
                                                  = [w8,w16,w32]
```

每个 Gaussian 获得一组满足 `w8 + w16 + w32 = 1` 的路由权重。当前版本是
**Gaussian-level routing**：同一个 Gaussian 的三个 plane 和每个 plane 的 12 个 hash channel
共享这组尺度权重，暂时不做 channel-level 或 hash-level routing。

路由器最后一层采用以下初始化：

```text
weight = 0
bias   = [-1, 2, -1]       # 依次对应 8、16、32
```

因此训练初期优先使用已经验证过的 `16×16` 路径，再逐步学习是否将不同 Gaussian 路由到
`8×8` 或 `32×32`。

#### 12.2.5 多尺度采样、融合和 FiLM

对一个 plane，以 XY 为例，Gaussian 的 XY 坐标会分别在三个尺度的调制图中进行双线性采样：

```text
XY coordinates [N,2]
        ├─ grid_sample(M8)  → gamma8,  beta8  [N,12]
        ├─ grid_sample(M16) → gamma16, beta16 [N,12]
        └─ grid_sample(M32) → gamma32, beta32 [N,12]
```

然后由该 Gaussian 的路由权重进行尺度混合：

```text
gamma_hat = w8*gamma8 + w16*gamma16 + w32*gamma32
beta_hat  = w8*beta8  + w16*beta16  + w32*beta32
```

最终保持原来的 FiLM 公式：

```text
feat_dyn = (1 + strength*tanh(gamma_hat))*feat + strength*beta_hat
```

XY、YZ、XZ 三个 plane 分别处理后仍得到：

```text
3 × [N,12] → enc_x_dyn [N,36]
```

因此后续 audio attention、eye attention、`[N,74]` 特征拼接和 deformation MLP 与原流程兼容。

#### 12.2.6 QGSFM 完整流程图

```text
DeepSpeech [8,29,16]                          Upper-face AUs [6]
          │                                             │
          ▼                                             ▼
AudioNet + AudioAttNet                              AU encoder
          │ [1,32]                                    │ [1,6]
          └──────────────────┬──────────────────────────┘
                             ▼
                  audio/AU spatial gated fusion
                             │
                    F16 [3,32,16,16]
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
               M8 γ/β     M16 γ/β    M32 γ/β
                  │          │          │
Gaussian XYZ ─────┼──────────┼──────────┼──→ tri-plane coordinate sampling
    │             │          │          │
    ▼             └──────────┴──────────┘
Tri-plane HashGrid             │
    │ static [N,36]            ▼
    ├──────────────────→ Gaussian-scale weighted mixture ──→ gamma_hat, beta_hat
    │                                  ▲
    │                                  │ [w8,w16,w32]
    │                    Gaussian Scale Router
    │                      ▲              ▲
    │                      │              │
    │             static enc_x       detached base scale [N,3]
    │
    └──────────────────────────────┐
                                   ▼
                         FiLM-modulated enc_x [N,36]
                                   │
                    audio attention + eye attention
                                   │
                              concat [N,74]
                                   │
                            deformation MLP
                                   │
                         Δxyz, Δrotation, Δscale
                                   │
                            Face Gaussian rendering
```

#### 12.2.7 训练约束和日志

多尺度版本沿用原有重建、形变、mask、attention 和 LPIPS 损失。与固定 `16×16` 版本相比，
TV loss 改为三个尺度 TV loss 的平均值：

```text
L_TV_multi = (TV(M8) + TV(M16) + TV(M32)) / 3
```

当前第一版没有加入 routing entropy、load balancing 或 Gaussian-scale prior loss，避免将多个
实验因素混在一起。训练时 TensorBoard 每 100 iteration 记录：

```text
scale_router/weight_8
scale_router/weight_16
scale_router/weight_32
```

它们是当前帧所有 Gaussian 的平均路由权重，可用于观察路由是否始终退化到某一个尺度。

默认参数：

```text
--geometry_mod_multiscale 1
--geometry_mod_map_res 16
```

可以通过训练脚本的第四个参数控制多尺度开关：

```bash
# 开启 8/16/32 多尺度调制（默认）
bash scripts/train_xx.sh data/<ID> output/<run_name> <GPU_ID> 1

# 固定 16×16 调制，用作消融实验
bash scripts/train_xx.sh data/<ID> output/<run_name> <GPU_ID> 0
```

多尺度与固定尺度模型的 checkpoint 结构不同，加载时会进行一致性检查，不能直接混用。

---

## 13. 创新点二：Canonical Feature Head Stabilizer（CFHS）

### 13.1 它要解决什么问题

TalkingGaussian 先通过 3DMM tracking 得到每帧头部姿态，再把姿态写入
`transforms_train.json` 和 `transforms_val.json`。训练时，Face 和 Mouth 两组 Gaussian 都使用这套相机变换。如果某一帧的旋转或平移有一点误差，渲染结果就会表现为整个头部突然移动；网络
还可能把相机误差错误地学习成面部形变。

这个问题在大角度转头时尤其明显：一侧脸被遮挡、轮廓点的语义发生变化、嘴眼区域又受到表情
影响，逐帧 3DMM tracking 很容易产生小幅跳变。

CFHS 的核心思路可以概括为：

> 先从同一个人的可靠帧中建立一个统一头部特征模板，再让每一帧直接寻找自己在该模板中的
> 位置，最后对整段姿态修正施加时间约束。

它位于粗 3DMM tracking 与 transforms 生成之间：

```text
ori_imgs + parsing + landmark confidence + track_params.pt
                              │
                              ▼
            DenseMarks canonical feature extraction
                              │
                              ▼
          identity-specific canonical feature template
                              │
                              ▼
       per-frame SE(3) direct alignment + confidence fusion
                              │
                              ▼
              full-sequence temporal refinement
                              │
                              ▼
                  track_params_canonical.pt
                              │
                              ▼
          transforms_train.json / transforms_val.json
```

CFHS 固定 3DMM identity 和 expression geometry，只修正每帧粗姿态上的 6 维 SE(3) 增量。它
不改变音频、AU、Gaussian 初始化或 Face/Mouth motion network，也不是训练期间额外运行的网络。

### 13.2 第一步：提取逐帧 Canonical 特征

对每一帧原始图像，DenseMarks 输出：

```text
uvw feature: [3,Hf,Wf]
head mask:   [Hf,Wf]
```

这里的 `uvw` 不是 RGB 颜色，而是与统一头部表面对应的 canonical embedding。直观理解是：
同一人的鼻梁位置即使在不同帧、不同角度下出现在不同像素处，理想情况下仍应具有相近的
canonical 特征。

提取结果保存在 `canonical_features/` 中。manifest 会记录图像哈希、特征尺寸、模型权重哈希和
帧顺序；图像被修改、帧数不一致或权重不匹配时会明确报错，防止错误复用旧缓存。

### 13.3 第二步：选择可靠帧并建立人物模板

模板不能由任意单帧直接复制，因为单帧可能包含闭眼、张嘴、自遮挡或 tracking 误差。系统先用
粗 3DMM 姿态和几何计算每个表面 anchor 的朝向与 z-buffer 可见性，再综合以下指标选择模板帧：

- landmark detector confidence；
- 3DMM landmark 重投影误差；
- 刚性头部区域的可见比例；
- yaw 分箱，保证模板包含不同观察角度；
- 每个 yaw 区间的候选帧数量上限，避免正脸帧淹没侧脸帧。

默认只从训练区间，即完整序列前 `10/11` 中选择模板帧，避免读取验证帧建立模板。最多使用
128 个可靠帧。

对同一 3DMM 刚性表面点，在所有可靠帧的投影位置采样 DenseMarks 特征，然后聚合为人物专属
模板：

```text
3D surface point X_i
       │ project with coarse R_t,T_t
       ▼
image position p_(i,t)
       │ sample DenseMarks
       ▼
canonical observation f_(i,t)
       │ robust multi-frame aggregation
       ▼
template feature f_bar_i
```

默认的 `expression_invariant` 模式还会对 3DMM expression 参数做 PCA，并用岭回归分离与表情
相关的特征变化。回归截距作为中性模板特征，因此张嘴或眨眼不会被简单写进身份模板。每个
anchor 还会根据跨帧重复性、观测覆盖率和 yaw 覆盖率得到模板可靠性权重。

### 13.4 第三步：每帧直接对齐模板

对第 `t` 帧，以粗姿态 `(R_t^0,T_t^0)` 为起点，只优化一个 6 维修正：

```text
delta_xi_t = [rotation correction, translation correction]

(R_t*,T_t*) = compose_increment(delta_xi_t, R_t^0,T_t^0)
```

修正后的姿态把刚性 3D anchor 投影到当前帧，在投影点采样当前 DenseMarks 特征，并与模板中同一
表面位置的特征比较：

```text
L_align(t) = sum_i w_(i,t) ||F_t(project(T_t* X_i)) - f_bar_i||
             / sum_i w_(i,t)
```

`w_(i,t)` 同时包含模板可靠性、当前姿态可见性、head mask 和图像边界检查。这样，大 yaw 下被
遮挡的一侧不会参与对齐。优化还包含 pose prior，并限制最大旋转和平移修正，避免特征歧义把
姿态拉到不合理位置。

这一过程与相邻光流跟踪的关键区别是：每一帧都直接对齐同一个模板。某段侧脸发生遮挡后，人物
重新转回正脸时仍然对齐原模板，不依赖遮挡期间累计下来的轨迹。

### 13.5 连续置信度：避免逐帧硬切换

早期实现采用“通过检查就全部使用修正，否则完全退回粗姿态”的硬切换。若相邻两帧刚好落在
阈值两侧，姿态会突然跳变。当前实现为每帧计算 `0～1` 的连续置信度：

```text
c_t = geometric_mean(
    anchor confidence,
    improvement confidence,
    coverage confidence,
    spatial confidence,
    bound confidence
)

delta_xi_t_applied = c_t × delta_xi_t
```

五个组成部分分别检查：

1. **Anchor 数量**：当前帧是否有足够多可靠表面点；
2. **损失改善**：优化后 canonical 特征误差是否真的下降；
3. **覆盖保留率**：优化后仍在 mask 和图像范围内的 anchor 是否大量丢失；
4. **空间分布**：anchor 是否同时覆盖二维头部区域，而不是集中在一个很小的局部；
5. **边界余量**：修正是否贴近最大旋转或最大平移限制。

置信度高时保留大部分直接对齐结果；证据较弱时只应用较小修正；检查完全失败时自然回到粗
3DMM 姿态。这里通过缩放 SE(3) twist 实现连续插值，而不是在两个姿态矩阵之间逐元素平均。

### 13.6 全序列 SE(3) 时间约束

连续置信度消除了硬切换，但逐帧特征本身仍可能存在轻微噪声。因此系统在所有批次完成后，对
整段修正序列统一执行时间优化，而不是只在每个 32 帧 batch 内平滑：

```text
L_temporal = L_data
           + lambda_c × L_coarse
           + lambda_v × L_velocity
           + lambda_a × L_acceleration
```

其中：

```text
L_data         = ||delta'_t - delta_t_applied||²
L_coarse       = (1-c_t) ||delta'_t||²
L_velocity     = ||delta'_t - delta'_(t-1)||²
L_acceleration = ||delta'_(t+1)-2delta'_t+delta'_(t-1)||²
```

- `L_data` 保留每帧 canonical 对齐提供的观测；
- `L_coarse` 让低置信度帧更靠近原始粗姿态；
- `L_velocity` 抑制相邻修正突然变化；
- `L_acceleration` 抑制单帧尖峰和高频摆动。

旋转和平移先分别用允许的最大修正幅度归一化，避免单位不同导致某一项主导优化。当前默认权重
为 `lambda_v=0.15`、`lambda_a=0.05`、`lambda_c=0.10`，优化 100 步。时间项作用于“附加的
姿态修正”，因此原始 3DMM 轨迹中的真实快速转头仍被保留。

### 13.7 两轮模板更新

完整对齐执行两轮：

```text
coarse pose → initial template → first alignment
                              │
                              ▼
               select high-confidence template frames
                              │
                              ▼
           rebuild visibility and canonical template
                              │
                              ▼
                 second alignment + temporal refinement
```

第一轮改善模板帧的姿态；第二轮使用高置信度结果重新计算可见性和模板，再从原始粗姿态出发得到
最终修正。若高置信度模板帧不足，则安全回退到第一轮选择的可靠帧集合。

### 13.8 完整流程图

```text
Reference frames                            Coarse 3DMM
      │                                geometry, expression, R0,T0
      ▼                                           │
DenseMarks uvw + head mask                        ▼
      │                                surface anchors + visibility
      │                                           │
      └───────────────┬───────────────────────────┘
                      ▼
       reliable training-frame selection by
   confidence + reprojection + visibility + yaw bins
                      │
                      ▼
      multi-view expression-invariant identity template
                      │
                      ▼
       each frame directly aligns to the same template
                      │
                      ▼
 anchor count + loss improvement + retained coverage
       + spatial extent + correction-bound margin
                      │
                      ▼
          continuous confidence c_t in [0,1]
                      │
                      ▼
             c_t × per-frame SE(3) correction
                      │
                      ▼
 full-sequence velocity + acceleration refinement
                      │
                      ▼
                refined R*_t,T*_t
                      │
                      ▼
             track_params_canonical.pt
                      │
                      ▼
        transforms_train.json / transforms_val.json
                      │
                      ▼
 shared camera input for Face and Mouth Gaussian rendering
```

### 13.9 输出与诊断文件

CFHS 生成：

```text
canonical_features/
canonical_template.npz
canonical_diagnostics.json
track_params_canonical.pt
transforms_train.json
transforms_val.json
```

- `canonical_features/`：逐帧 DenseMarks 特征、head mask 和校验 manifest；
- `canonical_template.npz`：模板特征、anchor 可靠性、模板帧和 yaw 分箱；
- `track_params_canonical.pt`：保留原 3DMM 参数，并保存最终 `rot`、`trans` 和 `euler`；
- `canonical_diagnostics.json`：记录每帧置信度及其五个分量、优化前后损失、原始与应用后的修正、
  时间调整量、重投影误差和旋转速度/加速度；
- transforms 文件：TalkingGaussian 训练和渲染实际读取的相机外参。

诊断中的 `accepted` 仅表示 `confidence >= 0.5` 的高置信度统计，不再控制全量采用或完全回退。
判断方法是否有效时，应同时查看渲染视频、旋转加速度、LVD、tLPIPS、tOF 和 Flow Warp Error；
不能只凭 canonical 特征损失下降就断言最终视频一定更稳定。

### 13.10 新视频的数据处理方式

完整预处理并启用 CFHS：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --head_stabilizer canonical \
    --canonical_python /path/to/python>=3.10 \
    --densemarks_repo /path/to/densemarks \
    --densemarks_weights /path/to/model.safetensors \
    --canonical_keep_cache
```

基础预处理已经完成时，只执行 canonical 对齐：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --task 11 \
    --canonical_python /path/to/python>=3.10 \
    --densemarks_repo /path/to/densemarks \
    --densemarks_weights /path/to/model.safetensors \
    --canonical_keep_cache \
    --canonical_overwrite
```

然后同步刷新 transforms：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --task 9 \
    --track_params track_params_canonical.pt
```

`--canonical_keep_cache` 会保留特征，后续调整置信度或时间约束时无需再次运行 DenseMarks。
DenseMarks 提取环境要求 Python 3.10 或更高；几何可见性和对齐阶段使用 TalkingGaussian 环境并
依赖 CUDA。

---

## 14. 两个创新点之间的关系

两个创新点解决的是在线局部形变建模与离线全局坐标稳定两个层级的问题：

| 模块 | 所处阶段 | 主要输入 | 主要输出 | 解决的问题 |
| --- | --- | --- | --- | --- |
| QGSFM（由 SQAF 与 GSAM 组成） | Face branch 训练与推理 | audio、AU、learned plane query、Gaussian XYZ 和 base scale | 空间门控融合、多尺度调制与 Gaussian deformation | 异构条件缺少空间分工，且固定调制分辨率与 Gaussian 尺度不匹配 |
| CFHS | 离线数据预处理 | 视频帧、DenseMarks、粗 3DMM 姿态和几何 | 稳定的相机 `R*, T*` 与 transforms 文件 | 大姿态、自遮挡和逐帧对齐噪声造成的全头抖动 |

它们最终在渲染阶段汇合：QGSFM 内部先由 SQAF 将音频与 AU 映射为空间调制特征，再由 GSAM
依据 Gaussian 的空间表征和 canonical scale 选择调制分辨率；CFHS 提供稳定的共享 camera
transform。原始 Mouth branch 继续预测口腔内部位置形变。这样形成“稳定全局坐标 + 尺度适配
局部形变”的两级方法。

---

## 15. 论文问题定义与叙事建议

### 15.1 建议的核心问题

论文不宜把问题简单写成“TalkingGaussian 的调制图分辨率不够”或“我们增加若干模块”。这种表述
容易被审稿人理解为局部工程改动，而且 CFHS 与 Face branch 看起来缺少联系。

更合适的核心问题是：

> Talking head Gaussian avatars require local audio-driven deformation to be learned in a stable spatial
> coordinate system. Existing motion fields do not explicitly align heterogeneous driving signals with
> facial regions and Gaussian spatial scales, while unreliable large-pose tracking further corrupts the
> coordinate system in which these local motions are learned.

对应的中文问题定义是：

> 音频驱动的 Gaussian 数字人需要在稳定的空间坐标中学习局部面部形变。然而，现有 motion
> field 缺少对异构驱动条件、局部空间区域和 Gaussian 表征尺度的显式对齐；大姿态下不可靠的
> 头姿跟踪又会破坏这种空间对应关系，最终造成局部运动表达不足和全局头部抖动。

这个问题可以概括为：

```text
Reliable and Scale-Aligned Gaussian Motion Learning

               What drives the motion?
                         │
                  SQAF
       Audio–AU spatial alignment
                         │
                         ▼
             Where and at what scale?
                         │
                  GSAM
       Gaussian-scale-aware routing
                         │
                         ▼
           In which coordinate system?
                         │
                CFHS canonical stabilization
```

两个创新点覆盖三个相互关联的层次：QGSFM 联合处理 **condition alignment** 和
**scale alignment**，CFHS 处理 **coordinate reliability**。

### 15.2 论文故事的逻辑顺序

建议 Introduction 按以下顺序展开。

#### 第一段：TalkingGaussian 的优势与基本前提

TalkingGaussian 通过 persistent Gaussian fields 和 Face–Mouth decomposition 获得高效、结构
稳定的说话人渲染。其关键过程是：在 canonical Gaussian field 上查询空间特征，根据音频和表情
预测形变，再通过跟踪得到的相机姿态进行渲染。

这里自然引出一个隐藏前提：准确的动画要求 **驱动条件、Gaussian 空间表示和相机坐标三者保持
一致**。

#### 第二段：条件与空间区域没有显式对齐

语音和 AU 是两种异构驱动信号。语音主要描述发音状态，AU 描述眼部和上半脸表情。将它们作为
全局向量送入 deformation MLP，虽然可以产生运动，但没有明确建模两类条件在不同面部区域的
相对作用。

由此提出第一个观察：

> A global condition vector tells the network what happens, but does not explicitly tell it where each
> driving source should dominate.

QGSFM 的第一个组成部分因此引入 learned plane queries。query 提供 canonical 空间模板，空间 gate 再决定
每个位置更依赖 audio candidate 还是 AU candidate，生成空间条件化的 modulation feature。

#### 第三段：固定调制分辨率与 Gaussian 尺度不匹配

即使获得了空间调制图，单一 `16×16` 分辨率仍对所有 Gaussian 一视同仁。Gaussian field 本身是
多尺度且非均匀的：不同 Gaussian 的尺寸和各向异性不同，densification 后的局部结构也具有更细
的空间支持范围。

由此提出第二个观察：

> A single modulation resolution cannot simultaneously preserve smooth regional motion and resolve local
> high-frequency deformation for heterogeneous Gaussians.

QGSFM 的第二个组成部分从共享 `F16` 构建 `8/16/32` modulation pyramid，并让每个 Gaussian 根据静态
HashGrid 特征和 detached canonical scale 学习一组尺度混合权重。这里应使用“learns to select or
mix suitable modulation resolutions”，不要在没有可视化证据时直接声称“小 Gaussian 一定选择
32×32”。

#### 第四段：不稳定相机坐标会污染局部形变学习

前两个模块都依赖 canonical Gaussian 坐标和每帧相机姿态。当大 yaw 导致自遮挡时，可用
landmark 变少，轮廓点的像素位置与真实表面位置也更难稳定对应，容易污染粗 3DMM pose。相机
误差会被训练过程吸收到 Gaussian deformation 中，使网络用局部非刚性形变补偿全局刚性误差，
并在渲染中表现为整体头部抖动。

由此提出第三个观察：

> Scale-aware local motion remains ill-posed when the global camera coordinate is temporally unreliable.

第二个创新点 CFHS 在训练前建立人物专属 canonical 特征模板，使每帧直接对齐同一模板，
再用连续可靠性融合和全序列 SE(3) 时间约束得到稳定相机外参。

#### 第五段：方法闭环

最终方法形成以下闭环：

```text
CFHS stabilizes the coordinate system
                │
                ▼
SQAF aligns audio/AU with spatial regions
                │
                ▼
GSAM aligns modulation resolution with local representation
                │
                ▼
deformation MLP predicts local face motion in the refined coordinate system
```

论文的主张应落在“更可靠、更细致的 motion learning”上，而不是仅强调网络更复杂。

### 15.3 可以直接用于论文的贡献表述

英文 contribution list 可以写成：

1. **Query-Guided Scale-Aware Face Modulation (QGSFM).** We introduce a unified face modulation method
   that first employs learnable canonical plane queries to spatially gate audio and upper-face AU
   conditions, and then constructs a shared multi-resolution modulation pyramid with a Gaussian-level
   router conditioned on static spatial features and canonical Gaussian scales. This enables each facial
   Gaussian to receive source-aware and scale-adaptive modulation before FiLM-based deformation.
2. **Canonical Feature Head Stabilizer (CFHS).** We build an expression-insensitive identity template from
   reliable multi-view canonical features, directly align every frame to this shared template, and combine
   continuous alignment confidence with full-sequence SE(3) temporal refinement for stable camera transforms.

中文表述可以写成：

1. 提出查询引导尺度感知人脸调制方法（QGSFM）：首先基于可学习三平面 query 对音频和 AU
   条件进行逐位置门控融合，再通过共享多分辨率调制金字塔和 Gaussian 级路由器，为不同空间
   位置与尺度的 Gaussian 自适应混合调制分辨率；
2. 提出 Canonical 特征头部稳定器（CFHS）：从可靠多视角观测构建表情不敏感的人物模板，
   将每帧直接对齐共享模板，并结合连续置信度与全序列 SE(3) 时间约束生成稳定相机轨迹。

### 15.4 摘要中的一句话方法概括

可以使用下面这句作为摘要方法部分的基础：

> We present a reliable and scale-aligned Gaussian motion framework that first stabilizes large-pose
> camera trajectories with CFHS, then spatially fuses audio and AU conditions through SQAF, and finally
> uses GSAM to route each facial Gaussian to a learned mixture of multi-resolution modulation fields.

对应中文：

> 我们提出一种可靠且尺度对齐的 Gaussian 运动建模框架：首先稳定大姿态视频中的相机轨迹，
> 随后通过 SQAF 对音频和 AU 条件进行空间门控融合，最后由 GSAM 为每个面部 Gaussian
> 自适应选择多分辨率调制场的组合。

### 15.5 标题方向

如果实验结果同时证明局部质量和大姿态稳定性，可以考虑：

- **Reliable and Scale-Aligned Gaussian Motion Fields for Talking Head Synthesis**
- **ScaleTalkGaussian: Scale-Aligned Motion Fields with Canonical Pose Stabilization**
- **Stable Gaussian Talking Heads via Query-Guided Modulation and Scale-Aware Routing**

第一种最稳妥，能够覆盖两个创新点且不把论文标题绑定在某一个实现细节上。第二种更强调方法名称，
但需要全文统一使用该名称。第三种更具体，适合实验主要围绕 Face branch 和稳定性展开的情况。

### 15.6 论文中需要避免的表述

- 不要把 learned plane query 描述成显式人脸语义分割。当前 gate 是从重建监督中学习的空间权重，
  没有区域语义标签；
- 不要声称 router 必然建立“大 Gaussian→低分辨率、小 Gaussian→高分辨率”的单调映射。当前
  router 同时读取空间特征和尺度描述符，输出是学习到的混合权重；
- 不要把 CFHS 描述成 jointly optimizing 3D geometry and pose。当前实现固定 3DMM geometry，
  优化每帧姿态的 SE(3) 增量，并在整段修正序列上施加时间约束；
- 不要把 DenseMarks 本身描述为本文创新。当前创新点是人物模板构建、直接姿态对齐、连续
  可靠性融合和全序列时间优化组成的稳定流程；
- 不要声称 Face branch 动态修改 opacity 或 SH color。网络虽然输出 `d_opa`，当前 renderer 没有
  应用它，实际动态更新的是 position、rotation 和 scale；
- 不要仅凭总体 PSNR/LPIPS 证明尺度路由或头部稳定。每个主张都需要对应证据。

### 15.7 建议的实验叙事与消融

为了验证两个创新点及 QGSFM 内部两个子模块的作用，建议至少设置以下递进消融：

| 实验 | SQAF | Modulation pyramid | Scale router | CFHS | 回答的问题 |
| --- | --- | --- | --- | --- | --- |
| A0 TalkingGaussian baseline |  |  |  |  | 原始性能 |
| A1 global/ungated modulation |  |  |  |  | 仅增加调制是否有效 |
| A2 SQAF | ✓ |  |  |  | 空间条件融合是否有效 |
| A3 multi-scale uniform average | ✓ | ✓ |  |  | 多尺度本身是否有效 |
| A4 GSAM | ✓ | ✓ | ✓ |  | 自适应路由是否优于平均融合 |
| A5 complete model | ✓ | ✓ | ✓ | ✓ | 稳定坐标是否进一步改善结果 |

对应证据建议如下：

- QGSFM/SQAF 子模块：全脸与上半脸/下半脸区域的 PSNR、LPIPS、landmark error；audio/AU gate
  heatmap；去掉 query、去掉 gate、固定平均 gate 的消融；
- QGSFM/GSAM 子模块：固定 `8×8`、固定 `16×16`、固定 `32×32`、uniform multi-scale 和 learned
  router 对比；按 Gaussian scale 分组统计 routing weights；嘴角、眼睑等局部 crop 的质量；
- CFHS：按 yaw 区间报告 LVD、tOF、Flow Warp Error 和 pose acceleration；与粗 3DMM、
  普通轨迹平滑、无连续置信度、无时间约束以及通用/人物/表情不敏感模板对比；重点展示
  ‘转头—遮挡—返回’序列；
- 完整模型：PSNR、SSIM、LPIPS、FID、CSIM 等总体指标，以及 lip synchronization 指标。总体
  图像质量与时间稳定性应分别报告，避免单一指标承担全部结论。

只有当这些消融支持对应假设后，摘要和结论中才使用“improves spatial specialization”、
“adapts modulation scale”或“reduces large-pose jitter”等结果性表述。
