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

2. **Large-Pose-Aware Head Stabilizer（LPHS）**

   在 3DMM 粗跟踪之后、生成 `transforms_train.json` 和 `transforms_val.json` 之前，利用
   姿态相关可见性、双向光流可靠性、关键帧重锚定和自适应 SE(3) 时序优化修正相机姿态。

后文使用 `QGSFM` 表示完整的“查询引导尺度感知人脸调制”，使用 `SQAF` 和 `GSAM` 指代其中
的条件融合与尺度路由子模块；`LPHS` 表示“大姿态感知头部稳定器”。

第一个创新点 QGSFM 参与 Face motion network 的训练和推理；第二个创新点 LPHS 属于数据
预处理，只离线生成更稳定的相机外参，不作为训练时的额外神经网络。

### 整体流程图

```text
                         Offline data preprocessing
┌───────────────────────────────────────────────────────────────────────────┐
│ Reference video → landmarks + 3DMM coarse tracking → LPHS                │
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

## 13. 创新点二：Large-Pose-Aware Head Stabilizer（LPHS）

### 13.1 模块位置和目标

TalkingGaussian 需要先用 3DMM 将视频帧转换为 canonical Gaussian field 对应的相机姿态。
当 yaw 较大时，一侧面部会发生自遮挡，普通 landmark 或逐帧 optical flow 可能漂移，导致错误的
`R_t, T_t` 被写入 transforms 文件。由于所有 Gaussian 共享该相机变换，姿态误差表现为整个头部
抖动，而不只是局部嘴部误差。

LPHS 插入在粗 3DMM tracking 和 transforms 生成之间：

```text
ori_imgs + landmarks + landmark confidence + track_params.pt
                               │
                               ▼
                              LPHS
                               │
                               ▼
                    track_params_lphs.pt
                               │
                               ▼
              transforms_train.json / transforms_val.json
```

LPHS 固定每帧的 3DMM identity/expression geometry，只优化粗姿态上的 SE(3) 增量，不联合修改
3DMM 几何，也不改变音频特征或 Face/Mouth Gaussian 参数。

### 13.2 Pose-Aware Visibility

LPHS 使用 68 个 3D landmark anchors 和 464 个 rigid mesh anchors，共 532 个 anchor。粗 3DMM
几何经过当前姿态变换后，使用表面朝向分数和 z-buffer 可见性得到：

```text
visibility_weight = front_facing_score × z_buffer_visibility
```

大 yaw 下背向相机或被面部表面遮挡的 anchor 权重接近 0，避免不可见一侧的错误观测拉动全局姿态。
为降低显存和计算开销，可见性渲染按照 `visibility_batch_size` 分批，并可通过
`visibility_max_side` 使用缩放后的图像尺寸。

### 13.3 双向光流与关键帧重锚定

相邻帧计算正向和反向光流，并执行 forward-backward consistency：

```text
p_t ── forward flow ──→ p_(t+1)
 ▲                         │
 └──── backward flow ──────┘
```

往返误差越大，`flow_confidence` 越低。为避免长视频中的顺序累计漂移，系统根据粗 yaw 进行分箱，
从不同姿态区间选择高置信度关键帧。当满足以下条件之一时执行 keyframe re-anchoring：

- 到达固定 `reanchor_interval`；
- 当前帧进入新的 yaw bin；
- 相邻帧跟踪的中位置信度低于阈值。

顺序跟踪结果和关键帧匹配结果根据各自置信度加权融合。默认优先使用 RAFT；RAFT 不可用时可
回退到 DIS optical flow。

### 13.4 动态可靠性权重

LPHS 对每个 anchor、每一帧构造动态权重：

```text
w(i,t) = w_sem(i)
       × w_vis(i,t)
       × w_flow(i,t)
       × w_conf(i,t)
```

其中：

- `w_sem`：语义刚性权重。nose bridge 等刚性区域权重高；face contour 中等；眼睛、眉毛、
  嘴部等表情区域权重低；
- `w_vis`：当前姿态下的朝向与 z-buffer 可见性；
- `w_flow`：双向光流一致性和重锚定可靠性；
- `w_conf`：landmark detector confidence 或传播后的 source confidence。

若某帧有效 anchor 数量低于 `min_effective_anchors`，该帧的观测权重会被置零，主要依靠相邻帧
的时序项和 pose prior 约束，避免少数异常点控制姿态。

### 13.5 Robust SE(3) Temporal Optimization

对粗姿态 `R_t, T_t` 引入 6 维 Lie algebra 增量 `delta_t`：

```text
(R*_t, T*_t) = compose_increment(delta_t, R_t, T_t)
```

优化目标由三部分组成：

```text
L_LPHS = L_detector_reprojection
       + lambda_track × L_track_reprojection
       + lambda_acc × L_adaptive_acceleration
       + lambda_prior × L_pose_prior
```

重投影项使用 Huber loss，并由上一节的动态可靠性权重控制。时序项作用于相邻相对姿态 twist
的二阶差分，即姿态加速度，而不是直接对绝对 yaw/pitch/roll 做平滑：

```text
relative motion: xi_t = log(T_t * inverse(T_(t-1)))
acceleration:    a_t  = xi_t - xi_(t-1)
```

加速度权重根据粗姿态运动量自适应：静止或缓慢运动时加强抖动抑制，快速真实转头时减弱时序
约束。`pose prior` 则限制优化结果不要无依据地远离粗 3DMM 姿态。代码还限制单帧增量的旋转
和位移幅度，以降低异常优化风险。

### 13.6 LPHS 完整流程图

```text
Reference frames                         Coarse 3DMM parameters
      │                                  geometry, R_t, T_t
      ├───────────────┐                         │
      │               │                         ▼
      │               │               3D anchor construction
      │               │                         │
      ▼               ▼                         ▼
68 landmarks     Bidirectional flow    normals + z-buffer visibility
      │               │                         │
      │         keyframe re-anchor              │
      │               │                         │
      ▼               ▼                         ▼
detector conf.   flow confidence          visibility weight
      │               │                         │
      └───────────────┼─────────────────────────┘
                      │
                      ▼
            semantic × visibility × flow × confidence
                      │
                      ▼
              dynamic anchor reliability
                      │
                      ▼
       weighted Huber reprojection + pose prior
                      +
          adaptive SE(3) acceleration constraint
                      │
                      ▼
                 optimized R*_t, T*_t
                      │
                      ▼
              track_params_lphs.pt
                      │
                      ▼
         transforms_train.json / transforms_val.json
                      │
                      ▼
        shared camera input for Face and Mouth rendering
```

### 13.7 输出文件

LPHS 会在数据目录生成：

```text
track_params_lphs.pt
lphs_observations.npz
lphs_diagnostics.json
```

- `track_params_lphs.pt`：保留原始 3DMM 参数，并使用优化后的 `rot`、`trans` 和 `euler`；
- `lphs_observations.npz`：保存光流轨迹、landmark、语义/可见性/光流/置信度权重和关键帧，便于
  可视化或排查；
- `lphs_diagnostics.json`：保存优化配置、使用的 flow backend、关键帧、弱观测帧以及优化前后
  的重投影和姿态时序诊断值。

诊断值用于判断优化过程是否按预期工作，不应在没有实验统计的情况下直接宣称 LPHS 一定提升
所有视频的稳定性。

### 13.8 新视频的数据处理方式

对新视频执行完整预处理并启用 LPHS：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --head_stabilizer lphs \
    --lphs_preset balanced \
    --lphs_flow_backend raft
```

如果基础预处理已经完成，只执行 LPHS：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --task 10 \
    --lphs_preset balanced \
    --lphs_flow_backend raft
```

然后使用 LPHS 参数重新生成 transforms：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4 \
    --task 9 \
    --track_params track_params_lphs.pt
```

若目标文件已经存在并且确定需要覆盖，显式增加 `--lphs_overwrite`。LPHS 的几何可见性步骤与
项目原始 3DMM tracker 一样依赖 CUDA。

---

## 14. 两个创新点之间的关系

两个创新点解决的是在线局部形变建模与离线全局坐标稳定两个层级的问题：

| 模块 | 所处阶段 | 主要输入 | 主要输出 | 解决的问题 |
| --- | --- | --- | --- | --- |
| QGSFM（由 SQAF 与 GSAM 组成） | Face branch 训练与推理 | audio、AU、learned plane query、Gaussian XYZ 和 base scale | 空间门控融合、多尺度调制与 Gaussian deformation | 异构条件缺少空间分工，且固定调制分辨率与 Gaussian 尺度不匹配 |
| LPHS | 离线数据预处理 | 视频帧、landmark、粗 3DMM 姿态和几何 | 稳定的相机 `R*, T*` 与 transforms 文件 | 大姿态、自遮挡和跟踪漂移造成的全头抖动 |

它们最终在渲染阶段汇合：QGSFM 内部先由 SQAF 将音频与 AU 映射为空间调制特征，再由 GSAM
依据 Gaussian 的空间表征和 canonical scale 选择调制分辨率；LPHS 提供稳定的共享 camera
transform。原始 Mouth branch 继续预测口腔内部位置形变。这样形成“稳定全局坐标 + 尺度适配
局部形变”的两级方法。

---

## 15. 论文问题定义与叙事建议

### 15.1 建议的核心问题

论文不宜把问题简单写成“TalkingGaussian 的调制图分辨率不够”或“我们增加若干模块”。这种表述
容易被审稿人理解为局部工程改动，而且 LPHS 与 Face branch 看起来缺少联系。

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
                LPHS pose stabilization
```

两个创新点覆盖三个相互关联的层次：QGSFM 联合处理 **condition alignment** 和
**scale alignment**，LPHS 处理 **coordinate reliability**。

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

前两个模块都依赖 canonical Gaussian 坐标和每帧相机姿态。当大 yaw 导致自遮挡时，不可见
landmark 和漂移的 optical flow 会污染粗 3DMM pose。相机误差会被训练过程吸收到 Gaussian
deformation 中，使网络用局部非刚性形变补偿全局刚性误差，并在渲染中表现为整体头部抖动。

由此提出第三个观察：

> Scale-aware local motion remains ill-posed when the global camera coordinate is temporally unreliable.

第二个创新点 LPHS 在训练前估计 pose-dependent visibility 和 tracking reliability，通过 weighted
robust reprojection 与 adaptive SE(3) temporal optimization 得到稳定相机外参。

#### 第五段：方法闭环

最终方法形成以下闭环：

```text
LPHS stabilizes the coordinate system
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
2. **Large-Pose-Aware Head Stabilizer (LPHS).** We develop an offline pose refinement method that combines
   pose-dependent visibility, bidirectional-flow reliability, semantic confidence, keyframe re-anchoring,
   and adaptive SE(3) temporal optimization to provide stable camera transforms under large head poses.

中文表述可以写成：

1. 提出查询引导尺度感知人脸调制方法（QGSFM）：首先基于可学习三平面 query 对音频和 AU
   条件进行逐位置门控融合，再通过共享多分辨率调制金字塔和 Gaussian 级路由器，为不同空间
   位置与尺度的 Gaussian 自适应混合调制分辨率；
2. 提出大姿态感知头部稳定器（LPHS），联合利用姿态可见性、双向光流可靠性、关键帧重锚定和
   自适应 SE(3) 时序约束，生成稳定的训练与渲染相机轨迹。

### 15.4 摘要中的一句话方法概括

可以使用下面这句作为摘要方法部分的基础：

> We present a reliable and scale-aligned Gaussian motion framework that first stabilizes large-pose
> camera trajectories with LPHS, then spatially fuses audio and AU conditions through SQAF, and finally
> uses GSAM to route each facial Gaussian to a learned mixture of multi-resolution modulation fields.

对应中文：

> 我们提出一种可靠且尺度对齐的 Gaussian 运动建模框架：首先稳定大姿态视频中的相机轨迹，
> 随后通过 SQAF 对音频和 AU 条件进行空间门控融合，最后由 GSAM 为每个面部 Gaussian
> 自适应选择多分辨率调制场的组合。

### 15.5 标题方向

如果实验结果同时证明局部质量和大姿态稳定性，可以考虑：

- **Reliable and Scale-Aligned Gaussian Motion Fields for Talking Head Synthesis**
- **ScaleTalkGaussian: Scale-Aligned Motion Fields with Large-Pose Stabilization**
- **Stable Gaussian Talking Heads via Query-Guided Modulation and Scale-Aware Routing**

第一种最稳妥，能够覆盖两个创新点且不把论文标题绑定在某一个实现细节上。第二种更强调方法名称，
但需要全文统一使用该名称。第三种更具体，适合实验主要围绕 Face branch 和稳定性展开的情况。

### 15.6 论文中需要避免的表述

- 不要把 learned plane query 描述成显式人脸语义分割。当前 gate 是从重建监督中学习的空间权重，
  没有区域语义标签；
- 不要声称 router 必然建立“大 Gaussian→低分辨率、小 Gaussian→高分辨率”的单调映射。当前
  router 同时读取空间特征和尺度描述符，输出是学习到的混合权重；
- 不要把 LPHS 描述成 jointly optimizing 3D geometry and pose。当前实现固定 3DMM geometry，
  优化每帧姿态的 SE(3) 增量；
- 不要声称 Face branch 动态修改 opacity 或 SH color。网络虽然输出 `d_opa`，当前 renderer 没有
  应用它，实际动态更新的是 position、rotation 和 scale；
- 不要仅凭总体 PSNR/LPIPS 证明尺度路由或头部稳定。每个主张都需要对应证据。

### 15.7 建议的实验叙事与消融

为了验证两个创新点及 QGSFM 内部两个子模块的作用，建议至少设置以下递进消融：

| 实验 | SQAF | Modulation pyramid | Scale router | LPHS | 回答的问题 |
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
- LPHS：按 yaw 区间报告 LVD、tOF、Flow Warp Error 或 pose acceleration；与粗 3DMM
  tracking、普通平滑和固定语义权重对比；展示大姿态自遮挡序列；
- 完整模型：PSNR、SSIM、LPIPS、FID、CSIM 等总体指标，以及 lip synchronization 指标。总体
  图像质量与时间稳定性应分别报告，避免单一指标承担全部结论。

只有当这些消融支持对应假设后，摘要和结论中才使用“improves spatial specialization”、
“adapts modulation scale”或“reduces large-pose jitter”等结果性表述。
