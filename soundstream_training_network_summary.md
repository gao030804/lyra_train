# 当前 SoundStream 训练网络总结

> 本文档依据当前 `E:\lyra` 中的 `train_soundstream.py`、`audiolm_pytorch/soundstream.py`、`audiolm_pytorch/trainer.py` 和 `run_stages_0_to_4.sh` 整理。

## 1. 任务目标

当前网络是面向 16 kHz 单声道语音的因果 SoundStream 风格神经音频编解码器，采用纯卷积编码器/解码器和 23 级残差向量量化（RVQ）。

- 输入：16 kHz 单声道波形
- 基础训练片段：2 秒，即 32,000 samples
- 长流式微调片段：4 秒，即 64,000 samples
- 总下采样倍数：`2 × 4 × 5 × 8 = 320`
- 帧率：`16000 / 320 = 50 frames/s`
- 每帧最多输出 23 个 8-bit RVQ索引
- 最大载荷码率：`50 × 23 × 8 = 9,200 bit/s`
- 主干约束：纯卷积，`use_local_attn=False`
- 卷积模式：因果卷积，`pad_mode="constant"`

## 2. 总体数据流

```text
波形 [B, 1, T]
  │
  ├─ CausalConv1d, kernel=7, 1→16
  │
  ├─ EncoderBlock, stride=2,  16→32
  ├─ EncoderBlock, stride=4,  32→64
  ├─ EncoderBlock, stride=5,  64→128
  ├─ EncoderBlock, stride=8, 128→256
  │
  ├─ CausalConv1d, kernel=3, 256→64
  │
  ├─ GroupedResidualVQ
  │    ├─ codebook_dim=64
  │    ├─ codebook_size=256
  │    ├─ num_quantizers=23
  │    └─ groups=1
  │
  ├─ CausalConv1d, kernel=7, 64→256
  │
  ├─ DecoderBlock, stride=8, 256→128
  ├─ DecoderBlock, stride=5, 128→64
  ├─ DecoderBlock, stride=4,  64→32
  ├─ DecoderBlock, stride=2,  32→16
  │
  └─ CausalConv1d, kernel=7, 16→1
       ↓
重建波形 [B, 1, T]
```

对于 2 秒输入，时间维大致变化为：

```text
32000 → 16000 → 4000 → 800 → 100 frames
```

量化前潜变量约为 `[B, 64, 100]`，RVQ索引对应每帧最多23级量化结果。

## 3. 编码器与解码器

### 3.1 EncoderBlock

每个编码块包含：

1. 三个残差单元，dilation依次为`1、3、9`；
2. 每个残差单元使用：
   - `CausalConv1d(kernel=7, dilation=d)`
   - ELU
   - `CausalConv1d(kernel=1)`
   - ELU
3. 最后使用`CausalConv1d(kernel=2×stride, stride=stride)`完成下采样和通道扩展。

### 3.2 DecoderBlock

每个解码块包含：

1. `CausalConvTranspose1d(kernel=2×stride, stride=stride)`；
2. 三个dilation为`1、3、9`的残差单元。

编码器和解码器均不启用Local Transformer、GateLoop和Squeeze-Excite。

## 4. RVQ量化器

当前量化器配置：

| 项目 | 当前值 |
|---|---:|
| 类型 | `GroupedResidualVQ` |
| 潜变量维度 | 64 |
| 码本大小 | 256 |
| RVQ级数 | 23 |
| 分组数 | 1 |
| 码本EMA decay | 0.95 |
| K-means初始化 | 开启 |
| dead-code阈值 | 2 |
| quantize dropout | 关闭 |
| rotation trick | 开启 |
| 多GPU码本同步 | 8-GPU阶段开启 |

码率支持：

| 使用量化器数 | 载荷码率 |
|---:|---:|
| 8 | 3.2 kbps |
| 15 | 6.0 kbps |
| 23 | 9.2 kbps |

`active_codes`表示每层256个码字中，在验证样本里至少出现一次的平均比例；`perplexity`表示考虑使用频率后的有效码字数量。

## 5. 损失函数

### 5.1 重建损失

波形损失：

```text
wave_loss =
    waveform_L1
  + 0.5 × waveform_MSE
  + 0.1 × RMS_energy_loss
```

多尺度Mel频谱损失使用窗口长度：

```text
64, 128, 256, 512, 1024, 2048 samples
```

每个尺度同时计算线性Mel L1和log-Mel L1。

### 5.2 总生成器损失

```text
generator_total =
    recon_weight × wave_loss
  + 1.0 × multi_spectral_mel_loss
  + adversarial_weight × adversarial_loss
  + feature_weight × feature_matching_loss
  + 0.1 × commitment_loss
  + streaming_boundary_loss（仅流式阶段）
```

阶段0、1使用`recon_weight=10`；阶段2～4使用`recon_weight=1`。

### 5.3 判别器

GAN阶段使用：

- 三个多尺度时域判别器：scale `1、0.5、0.25`
- 一个Complex STFT判别器：
  - `n_fft=1024`
  - `hop_length=256`
  - `win_length=1024`
- Hinge GAN损失
- 最大对抗权重：`0.001`
- 最大feature matching权重：`5.0`

阶段0、1关闭GAN，因此`adv=0`、`feature=0`是预期行为。

## 6. 五阶段训练流程

| 阶段 | 模式 | 步数 | 单卡batch | 片段 | G学习率 | D学习率 | GAN计划 |
|---|---|---:|---:|---:|---:|---:|---|
| 0 `overfit` | 小数据链路验证 | 5,000 | 4 | 2 s | 3e-4 | 无 | 关闭 |
| 1 `recon_pretrain` | 全数据重建预训练 | 60,000 | 4 | 2 s | 2e-4 | 无 | 关闭 |
| 2 `gan_pretrain` | 非流式GAN预训练 | 30,000 | 4 | 2 s | 5e-5 | 1e-4 | 0～10k线性ramp |
| 3 `stream_finetune` | 2秒流式微调 | 20,000 | 4 | 2 s | 3e-5 | 5e-5 | 5k后启动，10k ramp |
| 4 `stream_finetune_long` | 4秒长流式微调 | 5,000 | 2 | 4 s | 1e-5 | 2e-5 | 1k后启动，2k ramp |

除阶段0外，默认使用8 GPU：

```text
per-GPU batch = 4
world size = 8
global batch = 32
```

阶段4的global batch为`2 × 8 = 16`。

## 7. 阶段衔接

```text
阶段0：独立链路验证，不向阶段1传递权重

阶段1 recon_pretrain
  ↓ best_selected.pt
阶段2 gan_pretrain
  ↓ best_selected.pt
阶段3 stream_finetune
  ↓ best_selected.pt
阶段4 stream_finetune_long
```

阶段1从随机初始化开始；阶段2～4只加载前一阶段选中的模型权重，不继承前一阶段优化器。

## 8. EMA策略

### 阶段0

```text
beta = 0.95
update_after_step = 0
update_every = 1
```

### 阶段1～4

```text
beta = 0.999
update_after_step = 0
update_every = 1
```

EMA在`accelerator.prepare()`完成DDP放置后创建，并显式移动到当前rank设备。

RVQ码本自身已经维护EMA状态，因此外层模型EMA不会再次平滑以下buffer，而是每步从online模型直接复制：

```text
rq.*._codebook.initted
rq.*._codebook.cluster_size
rq.*._codebook.embed_avg
rq.*._codebook.embed
```

神经网络参数仍按正常EMA规则更新。

## 9. 验证与checkpoint选择

固定验证集默认累计26个batch。每次验证同时计算online和EMA：

- validation score
- RMS ratio
- 零时延correlation
- 零时延SI-SDR
- ±40 ms对齐后的correlation
- ±40 ms对齐后的SI-SDR
- active code ratio
- codebook perplexity

质量门槛：

| 阶段 | aligned correlation | aligned SI-SDR |
|---|---:|---:|
| 阶段0 | ≥ 0.5 | ≥ -5 dB |
| 阶段1 | ≥ 0.5 | ≥ -5 dB |
| 阶段2～4 | ≥ 0.5 | ≥ 0 dB |

选择规则：

1. 分别判断online和EMA是否达到当前阶段质量门槛；
2. 在合格候选中选择validation score更低者；
3. 保存为`best_selected.pt`；
4. 日志打印`selected=online`或`selected=ema`。

同时保留：

- `best.pt`：同一选中step的online权重
- `best_ema.pt`：同一选中step的EMA权重
- `best_selected.pt`：该step中真正被选择的权重
- `latest.pt`：恢复训练用完整状态
- `soundstream.<step>.pt`：周期完整checkpoint

首个合格checkpoint产生前不累计early-stopping patience。

## 10. 保存与早停

| 阶段 | 周期checkpoint | 验证间隔 | 最早启用早停 | patience |
|---|---:|---:|---:|---:|
| 0 | 500 | 100 | 0 | 30 |
| 1 | 2,000 | 250 | 10,000 | 40 |
| 2 | 2,000 | 250 | 5,000 | 30 |
| 3 | 1,000 | 250 | 5,000 | 30 |
| 4 | 1,000 | 250 | 1,000 | 15 |

中断恢复使用`latest.pt`。只有已写入checkpoint的状态会保留，中断后尚未保存的step需要重新计算。

## 11. 流式阶段

阶段3、4使用`FrameStreamingSoundStream`：

- 内部帧长：320 samples，即20 ms
- 通过每层因果卷积状态实现连续流式处理
- 不依赖上一段PCM上下文
- 默认`stream_context_frames=0`
- 使用帧边界损失抑制20 ms边界伪影
- 默认boundary loss weight：0.1
- 默认boundary radius：8 samples

阶段4结束后执行连续、有状态的整文件流式测试。

## 12. 数据处理

- 支持FLAC、WAV、MP3和WebM
- 多声道输入先平均为单声道
- 全部重采样到16 kHz
- 长音频裁剪为阶段指定长度
- 不足指定长度的音频在末尾补零
- 阶段0使用固定裁剪并限制文件数量
- 阶段1～4按speaker划分：
  - train 90%
  - validation 5%
  - test 5%
- 训练片段最低RMS阈值：-45 dB

## 13. 当前主要产物

```text
results/
├─ overfit-64d-23q/
├─ recon-pretrain-64d-23q/
├─ gan-pretrain-64d-23q/
├─ stream-finetune-64d-23q/
└─ stream-finetune-long-64d-23q/
```

部署或进入下一阶段默认使用：

```text
best_selected.pt
```

恢复训练使用：

```text
latest.pt
```

## 14. 当前需要重点观察的指标

1. `selected=online/ema`：确认EMA修复后是否逐渐追上online。
2. `aligned_corr`与`aligned_si_sdr`：判断时域重建质量。
3. `active_codes`与`perplexity`：判断码本是否退化。
4. 阶段2的`gan_ramp`、判别器loss和feature loss：确认GAN渐进启用。
5. 阶段3、4的boundary loss及连续流式试听：确认帧边界没有明显爆音。

