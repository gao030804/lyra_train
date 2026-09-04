# SoundStream Stage-2 GAN 15 万步训练日志总结

## 1. 日志与实验结论

分析日志：

```text
E:/lyra/results/24/s12/gan-pretrain-dscnn-relu-fp-64d-8q-from-s1-20260902-224609-s2-baseline-150000steps-6gpu-4s.log
```

服务器结果目录：

```text
/home/deploy/gyh/lyra_md/results/gan-pretrain-dscnn-relu-fp-64d-8q-from-s1-20260902-224609-s2-baseline-150000steps-6gpu-4s
```

本轮训练正常到达 `149999` step，并打印 `training complete (maximum steps)`，随后完成 100 条 held-out 测试。日志没有显示导致训练失败的异常。

核心结论：

- 三段式 15 万步训练流程完整执行。
- Encoder 和 RVQ 全程冻结，仅训练 Decoder；因此 Stage-1 编码表示与码本没有漂移。
- 完整 GAN 候选从 step 17000 开始产生。
- `best_full_gan_balanced.pt` 最后一次在 step 34000 刷新。
- step 34000 之后直到 step 150000 都没有产生更好的完整 GAN 候选。
- 延长到 15 万步没有导致崩溃，但在当前评价函数下，主要有效收益已经在前 3.4 万步完成。
- 最后一步 `soundstream.149999.pt` 只是训练终点，不能代替验证选出的最佳 checkpoint。

## 2. 模型与数据配置

| 项目 | 配置 |
|---|---:|
| 训练阶段 | `gan_pretrain` |
| 采样率 | 16 kHz |
| 训练数据 | LibriSpeech `train-clean-100` |
| 音频文件数 | 28,539 |
| 训练/验证/测试划分 | 90% / 5% / 5%，按说话人划分 |
| 单段长度 | 4 秒 |
| GPU 数量 | 6 |
| 每 GPU batch size | 4 |
| 全局 batch size | 24 |
| 最大训练步数 | 150,000 |
| checkpoint 保存间隔 | 5,000 step |
| 最佳模型验证间隔 | 500 step |
| Encoder | Block3/4 使用 DSCNN，激活为 ReLU |
| Encoder 状态 | 全程冻结 |
| RVQ 状态 | 全程冻结，EMA 与 dead-code replacement 关闭 |
| RVQ 数量 | 8 |
| Codebook size | 256 |
| 理论码率 | 3.2 kb/s |
| EMA 模型 | 关闭 |

码率计算：

```text
16000 / (2×4×5×8) = 50 latent frames/s
50 × 8 quantizers × log2(256) = 3200 bit/s
```

## 3. Stage-2 训练策略

### 3.1 冻结与释放

```text
step 0~1999：
    Generator 冻结
    判别器预热

step 2000 起：
    Decoder Generator 释放
    GAN adversarial/feature loss 开始渐增

整个 Stage-2：
    Encoder 冻结
    RVQ 冻结
    Decoder 可训练
```

Generator 学习率在释放后从 `1e-7` 线性增加到 `5e-7`，并在 step 5000 达到目标值。

### 3.2 三段式 15 万步调度

| 阶段 | Step 范围 | Generator LR | Adversarial 最大权重 | Feature 最大权重 |
|---|---:|---:|---:|---:|
| Phase 1 | `[0, 50000)` | 初始目标 `5e-7`，之后可受 plateau 调度影响 | `2e-4` | `1.5` |
| Phase 2 | `[50000, 100000)` | `2e-7` | 延续阶段配置 | 延续阶段配置 |
| Phase 3 | `[100000, 150000)` | `1e-7` | `1e-4` | `1.0` |

GAN 从 step 2000 开始，用 15000 step 完成 ramp；约在 step 17000 达到 `gan_ramp=1.0`。

### 3.3 重建损失权重

| 损失 | 权重/策略 |
|---|---:|
| Waveform | `10.0` |
| Mel | `1.1` |
| SI-SDR | `0.05` |
| Correlation | `0.02` |
| 双尺度 spectral envelope | `0.05` |
| Formant peak | 最大 `0.01`，前 5000 step ramp |
| Voiced high-band | `0.02` |
| Voiced-HF retention | `0.01` |
| Noise floor | `0.03` |
| MR-STFT | 最大 `0.02`，step 2000~10000 ramp |
| Active spectral detail | 最大 `0.01`，step 2000~10000 ramp |
| Frame phase | `0` |

MR-STFT 使用 6 个尺度：

```text
FFT size：64 / 128 / 256 / 512 / 1024 / 2048
权重：    0.25 / 0.50 / 0.75 / 1.00 / 1.00 / 0.75
```

## 4. 判别器配置与观察

| 判别器分支 | 更新间隔 | 损失权重 | 学习率 |
|---|---:|---:|---:|
| Waveform scale 1.0 | 2 | 1.0 | `5e-7` |
| Waveform scale 0.5 | 2 | 0.25 | `5e-7` |
| Waveform scale 0.25 | 2 | 0.25 | `2.5e-7` |
| STFT discriminator | 4 | 0.5 | `2.5e-7` |

STFT 判别器采用 real-only R1：

```text
gamma = 0.005
interval = 32
```

训练后段没有看到判别器完全饱和，但 waveform scale 1.0 和 scale 0.5 的梯度频繁触发裁剪。由于训练和验证没有发散，这不是失败；但它说明当前判别器更新偏强，后续若继续优化 Stage-2，可单独测试降低 waveform 判别器学习率或更新频率，不能同时修改多项配置。

## 5. Checkpoint 选择策略

Stage-2 初始化模型单独保存为：

```text
baseline_init.pt
```

它只是 Stage-1 初始化基线，不会被错误标记为已经训练的 Stage-2 最佳模型。

候选规则：

- 普通最佳候选从 step 5000 后开始选择。
- 质量 hard-stop 从 step 20000 后才开始统计，但本轮配置关闭了实际 hard-stop。
- `best_gan_balanced.pt` 要求 `gan_ramp >= 0.5`。
- `best_full_gan_balanced.pt` 要求 `gan_ramp = 1.0`。
- AC320、comb、高频指标仅用于诊断，不作为 checkpoint 硬门槛。

完整 GAN 最佳候选更新过程：

| Step | 文件 | Validation score | GAN ramp |
|---:|---|---:|---:|
| 17,000 | `best_full_gan_balanced.pt` | 2.713551 | 1.0 |
| 17,500 | `best_full_gan_balanced.pt` | 2.712614 | 1.0 |
| 31,000 | `best_full_gan_balanced.pt` | 2.712209 | 1.0 |
| **34,000** | **`best_full_gan_balanced.pt`** | **2.711411** | **1.0** |

step 34000 后没有再次刷新该文件。

## 6. 关键验证节点

分数越低越好；SI-SDR 和 correlation 越高越好。

| Step | Score | Aligned SI-SDR | Aligned Corr | F1 MAE | F2 MAE | F3 MAE | Voiced-HF ratio | AC320 | Comb median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 初始化 | 2.681463 | 3.266 dB | 0.798 | 47.0 Hz | 161.2 Hz | 379.1 Hz | -1.74 dB | 0.1157 | 4.91 dB |
| 9,500 | 2.716186 | 3.191 dB | 0.795 | 47.0 Hz | 160.0 Hz | 365.6 Hz | -0.36 dB | 0.1273 | 5.01 dB |
| 17,000 | 2.713551 | 3.199 dB | 0.796 | 47.0 Hz | 160.2 Hz | 365.1 Hz | -0.35 dB | 0.1276 | 4.99 dB |
| **34,000** | **2.711411** | **3.200 dB** | **0.796** | **47.2 Hz** | **160.2 Hz** | **364.7 Hz** | **-0.36 dB** | **0.1143** | **4.71 dB** |
| 50,000 | 2.715430 | 3.203 dB | 0.796 | 47.3 Hz | 160.2 Hz | 365.1 Hz | -0.28 dB | 0.1045 | 4.55 dB |
| 100,000 | 2.716766 | 3.207 dB | 0.796 | 47.4 Hz | 159.8 Hz | 366.3 Hz | -0.22 dB | 0.1015 | 4.38 dB |
| 149,500 | 2.715161 | 3.200 dB | 0.796 | 47.3 Hz | 159.5 Hz | 365.9 Hz | -0.19 dB | 0.1045 | 4.44 dB |

相对初始化，step 34000 的主要变化为：

- Aligned SI-SDR 下降 `0.066 dB`，仍在允许的 `0.15 dB` retention 范围内。
- Aligned correlation 约下降 `0.002`。
- F1 基本不变。
- F2 仅轻微改善。
- F3 MAE 从 `379.1 Hz` 降到 `364.7 Hz`，改善约 `14.4 Hz`。
- Voiced-HF ratio 从 `-1.74 dB` 改善到 `-0.36 dB`，高频能量明显接近目标。
- 综合重建 score 没有超过初始化，因此是否接受 Stage-2 必须结合盲听。

虽然 step 100000 的 AC320 和 comb 诊断数值比 step 34000 更低，但它的选择 score 更差；由于这两个指标已明确设为诊断项，不能据此覆盖 `best_full_gan_balanced.pt`。

## 7. 100 条 Held-out 测试结果

最终测试使用：

```text
best_full_gan_balanced.pt
```

而不是：

```text
soundstream.149999.pt
```

| 指标 | 测试结果 | 方向/含义 |
|---|---:|---|
| Score | 2.707407 | 越低越好 |
| Mel | 2.277863 | 越低越好 |
| MR-STFT | 0.967767 | 越低越好 |
| Waveform reconstruction | 0.017401 | 越低越好 |
| MSE | 0.001178 | 越低越好 |
| SI-SDR | 4.589792 dB | 越高越好 |
| Aligned SI-SDR | 4.598589 dB | 越高越好 |
| Correlation | 0.837915 | 越高越好 |
| Aligned correlation | 0.838659 | 越高越好 |
| RMS ratio | 0.973786 | 理想值接近 1 |
| Reconstruction peak | 0.508534 | 未接近满幅 |
| Clip fraction | 0.000118% | 很低 |
| Jump ratio | 0.848071 | 未触发 clean gate |
| Click score | 4.645207 | 未触发 clean gate |
| Voiced-HF ratio | +0.072 dB | 接近 0 dB |
| Voiced-HF error | 0.8609 | 诊断项 |
| Voiced-HF deficit | 0.3504 | 诊断项 |
| Voiced-HF retention | 4.6061 | 诊断项 |
| 7~7.8 kHz voiced ratio | -1.633 dB | 诊断项 |
| Quiet 7~7.8 kHz excess | +0.975 dB | 诊断项 |
| Spectral centroid delta | +13.4 Hz | 接近目标 |
| Spectral slope delta | +0.102 | 诊断项 |
| AC320 | 0.093269 | 仅诊断，不作硬门槛 |
| Phase peak | -9.585 dB | 诊断项 |
| Comb median | 5.176 dB | 仅诊断，不作硬门槛 |
| Active codes | 0.999512 | RVQ 使用健康 |
| Perplexity | 203.52 / 256 | 约为理论最大值的 79.5% |

Held-out 测试表明输出响度、削波和 RVQ 使用情况健康。但日志没有提供同一批 100 条音频的 `baseline_init.pt` 测试结果，因此不能仅凭本表声称 Stage-2 在 held-out 集上优于 Stage-1。

## 8. 稳定性判断

### 正常项

- 完整跑满 150,000 step。
- 未观察到训练中止或数值发散。
- Encoder/RVQ 冻结状态符合预期。
- RVQ validation 指标稳定：active codes 约 `0.992`，perplexity 约 `199.4`。
- 100 条 held-out 测试 active codes 为 `0.999512`，perplexity 为 `203.52`。
- 所列验证节点均为 `clean_ok=1`、`rvq_ok=1`。
- 重建音频削波比例很低。

### 需要关注

- 最佳完整 GAN checkpoint 停留在 step 34000，后续训练没有刷新。
- 重建 score 相对初始化略有退化。
- GAN 的主要客观收益集中在高频能量和 F3，而不是 SI-SDR。
- waveform 判别器前两尺度频繁梯度裁剪，可能限制后半程收益。
- EMA 关闭，最终只能在 online 权重中选择候选。

## 9. Checkpoint 使用建议

### 正式 Stage-2 候选

```text
best_full_gan_balanced.pt
```

理由：GAN ramp 已达到 1.0，并且是满足 clean、quality-retention 和 RVQ 条件的最低 selection score 完整 GAN 候选。

### Stage-1 对照基线

```text
baseline_init.pt
```

用途：与 Stage-2 最佳候选做相同长音频、相同推理参数的客观指标和盲听对照。

### 仅诊断

```text
soundstream.149999.pt
```

用途：确认长时间训练后是否产生听感漂移。它不是验证最佳点，不应直接用于部署。

## 10. 下一步操作建议

对完全相同的 10 条长测试音频，使用相同的：

```text
weights=auto
bitrate=3200
block_seconds=5
context_ms=60
```

比较：

```text
baseline_init.pt
best_full_gan_balanced.pt
soundstream.149999.pt
```

优先判断 `baseline_init.pt` 与 `best_full_gan_balanced.pt`：

- 元音是否更自然、饱满。
- 齿音和清辅音是否更清楚但不过亮。
- 高频恢复是否引入砂纸感或金属感。
- 音色是否稳定，是否出现说话人特征漂移。
- 是否存在 click、断裂或流式块边界伪影。

如果 `best_full_gan_balanced.pt` 在盲听中稳定胜出，可以接受 step 34000 的 Stage-2 模型。若盲听没有稳定收益，则不建议继续延长同一配置的训练；下一轮应单独调整判别器强度或 GAN/feature loss 配比，并将最大步数缩短到约 50,000 后重新验证。

## 11. 最终结论

本轮 15 万步 Stage-2 GAN 训练在工程上是成功的：流程完整、模型稳定、RVQ 健康，并成功生成和测试了完整 GAN 最佳候选。但从验证轨迹看，后 11.6 万步没有进一步改善选择指标。

推荐保留：

```text
baseline_init.pt
best_full_gan_balanced.pt
soundstream.149999.pt
held_out_test_report.txt
```

当前最合理的正式候选为 `best_full_gan_balanced.pt`；是否最终替代 Stage-1，需要由相同长音频的盲听和配对指标共同决定。
