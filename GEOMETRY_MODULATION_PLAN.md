# TalkingGaussian Stage 2.1 Geometry Modulation Plan

## Motivation

TalkingGaussian 的 face branch 已经采用 deformation-based 思路，通过 `MotionNetwork` 根据音频和 AU 表情特征预测 Gaussian 的几何偏移。当前实现中，空间特征由三组 2D hash-grid encoder 生成：

```python
feat_xy = self.encoder_xy(xy, bound=bound)
feat_yz = self.encoder_yz(yz, bound=bound)
feat_xz = self.encoder_xz(xz, bound=bound)
enc_x = torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
```

随后音频特征和上脸 AU 特征在点级别参与 MLP 预测：

```python
enc_w = enc_a * aud_ch_att
enc_e = enc_e * eye_att
h = torch.cat([enc_x, enc_w, enc_e], dim=-1)
```

这种做法虽然有 region attention，但每个 Gaussian 仍主要独立解释条件特征并回归偏移。改进目标是：把点级条件融合改为连续空间调制场，让 Gaussian 点从由音频和上脸特征共同生成的低分辨率几何调制图中采样调制参数，从而提升局部空间一致性并减少 audio 与 upper-face 特征纠缠。

## Design Choice

本方案不直接引入 dense triplane `T_can`。当前项目使用的是 hash-grid encoder，而不是显式三平面特征图。因此第一版改造应保留原有 hash-grid 结构，只在 hash-grid 采样后的点特征上做 FiLM 调制。

推荐改造对象：

```text
scene/motion_net.py -> MotionNetwork
```

暂不改：

```text
MouthMotionNetwork
GaussianModel
gaussian_renderer
train_fuse.py
```

第一版只改 face branch 的几何 motion field，不预测动态 `SH`、`opacity` 或外观调制。

## Proposed Module

新增模块可命名为 `AudioUpperFaceGeometryModulator` 或 `AUGMT`。

输入：

```text
audio feature a_n          # 来自 AudioNet + AudioAttNet
upper-face feature e_n     # 来自 AU / blink / upper-face encoder
plane query Q^xy,Q^yz,Q^xz # 可学习低分辨率空间 query
```

输出：

```text
gamma_beta: [B, 3, 2, C_plane, Hm, Wm]
```

其中：

```text
3       -> xy, yz, xz
2       -> gamma, beta
C_plane -> 每个平面的 hash feature 维度，当前建议 12
Hm,Wm   -> 调制图分辨率，建议 16 或 32
```

融合模块采用 AVIGATE-style gated fusion 思想：三平面空间 query 分别从 audio 和 upper-face 特征中提取候选控制信息，再用空间 gate 决定每个位置更依赖 audio 还是 upper-face。

## Feature Modulation

原始平面特征：

```python
feat_xy = self.encoder_xy(xy, bound=bound)
feat_yz = self.encoder_yz(yz, bound=bound)
feat_xz = self.encoder_xz(xz, bound=bound)
```

从低分辨率调制图采样：

```python
gamma_xy, beta_xy = sample_modulation(gamma_beta[:, 0], xy)
gamma_yz, beta_yz = sample_modulation(gamma_beta[:, 1], yz)
gamma_xz, beta_xz = sample_modulation(gamma_beta[:, 2], xz)
```

残差 FiLM 调制：

```python
feat_xy_dyn = (1 + lambda_gamma * torch.tanh(gamma_xy)) * feat_xy + lambda_beta * beta_xy
feat_yz_dyn = (1 + lambda_gamma * torch.tanh(gamma_yz)) * feat_yz + lambda_beta * beta_yz
feat_xz_dyn = (1 + lambda_gamma * torch.tanh(gamma_xz)) * feat_xz + lambda_beta * beta_xz

enc_x_dyn = torch.cat([feat_xy_dyn, feat_yz_dyn, feat_xz_dyn], dim=-1)
```

对应公式：

```text
h_i,dyn^p = (1 + lambda_gamma * tanh(Delta gamma_i^p)) * h_i^p
            + lambda_beta * beta_i^p
```

其中 `p in {xy, yz, xz}`。

## Motion Prediction

调制后再预测几何偏移：

```python
h = torch.cat([enc_x_dyn, enc_w, enc_e], dim=-1)
h = self.sigma_net(h)

d_xyz = h[..., :3] * 1e-2
d_rot = h[..., 3:7]
d_scale = h[..., 8:11]
```

第一版建议保留弱条件残差路径，避免一次性去掉原始条件注入导致训练不稳定：

```python
h = torch.cat([enc_x_dyn, 0.1 * enc_w, 0.1 * enc_e], dim=-1)
```

完成 ablation 后，再比较：

```text
A. enc_x + enc_w + enc_e                       # 原始
B. enc_x_dyn + enc_w + enc_e                   # 保守版
C. enc_x_dyn + 0.1*enc_w + 0.1*enc_e           # 推荐第一版
D. enc_x_dyn                                   # 完全调制版
```

## Initialization

调制图输出头必须零初始化：

```python
nn.init.zeros_(self.out_conv.weight)
nn.init.zeros_(self.out_conv.bias)
```

训练初始时：

```text
gamma = 0
beta = 0
enc_x_dyn = enc_x
```

这样模型刚进入 motion 阶段时等价于原始 TalkingGaussian，不会破坏已经学到的 canonical Gaussian。

## Training Strategy

保留 `train_face.py` 中的 warm-up：

```python
warm_step = 3000
```

训练流程：

```text
iteration < warm_step:
    render()，只训练静态 Gaussian

iteration >= warm_step:
    render_motion()，启用 MotionNetwork 和几何调制
```

调制强度建议 warm-up：

```text
lambda_gamma(t) = min(1, t / K)
lambda_beta(t)  = min(1, t / K)
K = 2000 ~ 5000
```

这里的 `t` 可从 `iteration - warm_step` 开始计数。

## Regularization

为了保证调制图平滑，加入 TV loss：

```text
L_TV = sum_p (
    ||grad_x gamma^p||_1 + ||grad_y gamma^p||_1
  + ||grad_x beta^p ||_1 + ||grad_y beta^p ||_1
)
```

推荐权重：

```text
lambda_TV = 1e-5 ~ 1e-4
```

为了减少 audio 和 upper-face gate 纠缠，可加入 gate overlap loss：

```text
L_gate = sum_p ||W_audio^p * W_upper^p||_1
```

推荐权重：

```text
lambda_gate = 1e-5 ~ 1e-4
```

如果后续构造了嘴部和上脸区域先验 mask，还可以增加：

```text
L_audio_upper = ||W_audio * M_upper||_1
L_upper_mouth = ||W_upper * M_mouth||_1
```

## Implementation Steps

1. 在 `scene/motion_net.py` 中新增 `AudioUpperFaceGeometryModulator`。
2. 在 `MotionNetwork.__init__()` 中实例化该模块。
3. 将 `encode_x()` 拆成返回三个平面特征：

   ```python
   feat_xy, feat_yz, feat_xz = self.encode_planes(xyz)
   ```

4. 在 `forward()` 中先编码 audio 和 upper-face，再生成 `gamma_beta`。
5. 用 `grid_sample` 根据 `xy/yz/xz` 从调制图中采样 `gamma/beta`。
6. 对 `feat_xy/feat_yz/feat_xz` 做残差 FiLM 调制。
7. 将调制后的 `enc_x_dyn` 送入 `sigma_net`。
8. 在 `train_face.py` 中加入 `L_TV` 和可选 `L_gate`。
9. 做 ablation：原始模型、保守调制版、弱残差版、完全调制版。

## Expected Benefits

该方案保持 TalkingGaussian 的 persistent Gaussian 和 deformation-only 框架不变，只改造 face branch 的条件注入方式。预期收益包括：

- 将 point-wise 条件融合改成连续空间调制场。
- 减少 audio 与 upper-face 特征在 face branch 中的模态纠缠。
- 让相邻 Gaussian 采样相近的调制参数，提高局部形变一致性。
- 保持外观参数不随音频直接变化，降低错误光影或颜色扰动。
- 通过 zero initialization 保证训练初始行为与原模型一致，便于稳定 ablation。

## First-Version Recommendation

第一版建议采用最小改动路线：

```text
保留 hash-grid encoder
保留 mouth branch
保留 fuse branch
只在 face branch 中新增 low-res gamma/beta 几何调制图
只调制 hash-grid 采样后的 feat_xy/feat_yz/feat_xz
不预测 Delta SH / Delta opacity
```

这一路线最贴合当前代码结构，也最容易验证 Stage 2.1 几何调制是否真正提升 TalkingGaussian 的动态稳定性。
