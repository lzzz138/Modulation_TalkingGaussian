# PCV-DF（实验方法）

固定 CFHS 相机，Mouth 50k → Face 50k → Fuse 10k。Face 在6k–9k渐增姿态残差。
相机旋转编码取 `world_view_transform[:3,:3]`（C2W旋转）前两列；逐点视线为点到相机单位方向。
姿态和可见性来自当前相机与模型，不读取目标图像或验证标签。

```bash
PYTHON=/home/lzq/miniconda3/envs/talking_gaussian/bin/python \
bash scripts/train_pcv_df.sh /path/to/CFHS/data output/posevideo5_pcv 2
# 可选第四参数：同数据 B 组未 Fuse 的 Mouth 50k checkpoint
```

脚本先完成原 Mouth，然后训练 Face；使用全新输出目录。`PCV_CONDITION` 可为
`none`、`pose`、`pose_visibility`。前两种条件不运行深度预渲染；参数量相同，计算成本不同。
完整模式短边128预渲染，使用相同投影矩阵、按宽高比缩放输出分辨率；
深度除以alpha，低alpha/非有限/越界点置零。近裁剪采用现有CUDA核的0.2阈值。
没有可靠法线，未使用Gaussian旋转轴冒充表面法线。

残差上限每轴0.001世界坐标单位，不代表毫米。权重为零时恢复原几何。
残差只加到XYZ，共享空间/音频/AU特征仍有梯度，因此共同主干更新可能间接改变其他输出。
预渲染完全无梯度，最终RGB路径正常参与增密。基础模型只前向一次。

## 保存、恢复与消融

Face保存版本化 `pcv_face_v1`，包含Gaussian及其optimizer、motion及optimizer/scheduler、
完整抽帧顺序和栈、随机状态、数据身份。保存发生在该步增密与优化完成后。
在3k、6k、原定期检查点及结束保存。恢复用 `--start_checkpoint`，所有训练配置严格匹配。
`--pcv_stop_after 6000` 在完成该步后保存并退出，不改变50k调度，便于建立共同起点。

从同一个6k完整checkpoint分叉时加 `--pcv_fork_6k`，只允许改变以下三项：
`pcv_condition`、`pcv_max_displacement`、`pcv_reg_weight`。之后正常恢复不需要该标志。

四项对照均使用PCV入口和同一个6k状态：

- B路径：`--pcv_max_displacement 0 --pcv_reg_weight 0`，不运行预渲染或残差。
- 等容量残差：`--pcv_condition none`。
- 姿态残差：`--pcv_condition pose`。
- 完整模式：`--pcv_condition pose_visibility`。

Mouth checkpoint复用需源目录cfg_args能验证同一transforms、AU、音频身份。
复用工具保存provenance，Fuse严格检查Face已完成训练、Mouth确为原始50k模型。
Fuse每1000步保存 `pcv_fuse_v1`，支持 `--start_checkpoint`；几何和运动从首次前向冻结。
合成自动从Fuse检查点恢复结构配置，显式传入冲突参数会报错。
原B/D模型使用原入口。禁止将PCV接入在线UVW位姿循环。

## 诊断和验证边界

`pcv_diagnostics.jsonl` 每200步记录当前训练帧门控、残差、饱和比例、覆盖、显存和额外前向时间。
`pcv_yaw_bins.json` 保存训练相机角度的圆周中心与三分位边界；验证分箱必须复用该文件。
这些yaw是相机约定下的相对角度，不是经过独立标注的真实解剖学头姿。

首轮仅在posevideo5比较图像/口部/时间指标、分箱指标和推理FPS。
同音频、AU固定的相机扫描需额外检查轮廓跳变；重建改善不证明真实三维形变恢复。
短训练使用人工推进迭代编号的边界fixture只能验证流程，不是效果实验。
