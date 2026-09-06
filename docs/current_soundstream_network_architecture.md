# 当前 SoundStream 训练网络结构

> 依据当前 `train_soundstream.py` 和 `audiolm_pytorch/soundstream.py` 整理。  
> 适用结构：16 kHz、下采样 320、64 维 latent、16 级 RVQ、每级 16 个码字、Encoder Block2/3/4 使用 DSCNN 与低秩逐点卷积、Decoder 使用 ConvTranspose1d。

## 1. 总体数据流

```text
音频 [B, 1, T]
  │
  ▼
Encoder
  │  [B, 64, T/320]
  ▼
转置为 [B, T/320, 64]
  │
  ├─ bypass-RVQ：直接送入 Decoder
  │
  └─ 正常模式：16级 Residual VQ
         ├─ 每级 16 个 64维码字
         ├─ 输出 16 个 4-bit index/帧
         └─ 量化 latent [B, T/320, 64]
  │
  ▼
Decoder
  │
  ▼
重建音频 [B, 1, T]
```

基础配置：

| 项目 | 当前值 |
|---|---:|
| 采样率 | 16,000 Hz |
| 输入通道 | 1 |
| 基础通道数 | 16 |
| `channel_mults` | `(2, 4, 8, 16)` |
| Encoder 通道 | `16 → 32 → 64 → 128 → 256 → 64` |
| `strides` | `(2, 4, 5, 8)` |
| 总下采样倍率 | `2×4×5×8 = 320` |
| latent 帧率 | `16000/320 = 50 帧/秒` |
| latent 维度 | 64 |
| 激活函数 | ReLU |
| Local Attention | 关闭 |
| Squeeze-Excite | 关闭 |
| Decoder 上采样 | Causal ConvTranspose1d |
| 显式插值平滑 | 关闭 |

## 2. 因果卷积规则

`CausalConv1d` 在输入左侧补历史样本，不读取未来输入：

```text
causal_padding = dilation × (kernel_size - 1) + (1 - stride)
```

离线训练使用左侧 padding；流式执行使用等长历史缓存。因此离线和流式卷积具有相同的因果感受野。

每个 Encoder/Decoder Block 都包含三个残差单元，dilation 依次为：

```text
(1, 3, 9)
```

残差形式为：

```text
y = x + residual_scale × F(x)
```

Encoder 的 `residual_scale=1`。Decoder 支持逐 Block residual scale；Stage-1 可以从 0.2 逐步升到 1.0，后续阶段从 checkpoint 恢复这些值。

## 3. Encoder 结构

以 1 秒、16,000 个采样点为例：

| 位置 | 张量形状 | 时间长度/秒 |
|---|---|---:|
| 输入 | `[B,1,T]` | 16,000 |
| Initial Conv 后 | `[B,16,T]` | 16,000 |
| Block1 后 | `[B,32,T/2]` | 8,000 |
| Block2 后 | `[B,64,T/8]` | 2,000 |
| Block3 后 | `[B,128,T/40]` | 400 |
| Block4 后 | `[B,256,T/320]` | 50 |
| Final Conv 后 | `[B,64,T/320]` | 50 |

### 3.1 Initial Conv

```text
Causal Conv1d: 1 → 16, K=7, S=1
```

### 3.2 Encoder Block1：标准卷积

输入/输出：`16 → 32`，下采样倍率 2。

```text
3 × ResidualUnit(16通道, K7, dilation=1/3/9)
    ├─ Conv1d 16→16, K7, dilation=d
    ├─ ReLU
    ├─ Conv1d 16→16, K1
    └─ ReLU + residual

Downsample Conv1d 16→32, K4, S2
```

### 3.3 Encoder Block2/3/4：DSCNN + 低秩逐点卷积

这三个 Block 使用 revision 3。Depthwise 和 Pointwise 之间没有额外 ReLU；激活仍位于完整的低秩 Pointwise 之后。

每个残差单元：

```text
x
├─ Depthwise Conv1d C→C, K7, groups=C, dilation=d
├─ LowRank Pointwise C→R→C, K1 + K1（中间无激活）
├─ ReLU
├─ LowRank Pointwise C→R→C, K1 + K1（中间无激活）
├─ ReLU
└─ 与 x 相加
```

每个下采样层：

```text
Depthwise Conv1d Cin→Cin, K=2S, stride=S
  → LowRank Pointwise Cin→Rdown→Cout
```

| Block | Cin→Cout | Stride | 残差 PW rank | 下采样 PW rank |
|---|---:|---:|---:|---:|
| Block2 | 32→64 | 4 | 8 | 16 |
| Block3 | 64→128 | 5 | 16 | 32 |
| Block4 | 128→256 | 8 | 32 | 64 |

### 3.4 Final Conv

```text
Causal Conv1d: 256 → 64, K=3, S=1
```

## 4. Encoder 卷积参数量和 MAC

计算约定：

- Weight 与 Bias 分开统计；MAC 不包含 Bias、ReLU、padding 和残差加法。
- `MAC/s = 每个输出位置的权重乘加数 × 每秒输出位置数`。
- 低秩逐点卷积 `Cin→R→Cout` 的 Weight 为 `Cin×R + R×Cout`。

| 模块 | Weight | Bias | 卷积参数合计 | MAC/s | 20 ms MAC |
|---|---:|---:|---:|---:|---:|
| Initial Conv | 112 | 16 | 128 | 1.792 M | 0.03584 M |
| Encoder Block1 | 8,192 | 128 | 8,320 | 114.688 M | 2.29376 M |
| Encoder Block2 | 5,536 | 448 | 5,984 | 33.536 M | 0.67072 M |
| Encoder Block3 | 20,416 | 896 | 21,312 | 29.9776 M | 0.599552 M |
| Encoder Block4 | 78,464 | 1,792 | 80,256 | 22.0672 M | 0.441344 M |
| Final Conv | 49,152 | 64 | 49,216 | 2.4576 M | 0.049152 M |
| **Encoder 合计** | **161,872** | **3,344** | **165,216** | **204.5184 M** | **4.090368 M** |

说明：Block2 虽然时间分辨率较高，但低秩 DSCNN 已把它的计算量显著降低；Block4 参数较多，但只在 400 Hz 和 50 Hz 的低时间分辨率上计算。

## 5. RVQ 结构

当前 RVQ 参数：

| 项目 | 当前值 |
|---|---:|
| RVQ 级数 | 16 |
| 每级码字数 | 16 |
| 每个码字维度 | 64 |
| 每个 index 位宽 | 4 bit |
| 每帧 index 数 | 16 |
| 帧率 | 50 Hz |
| 码率 | `50×16×4 = 3,200 bit/s` |
| FP32 码本存储 | `16×16×64×4 = 65,536 B` |
| INT8 部署码本存储 | `16×16×64 = 16,384 B` |

第 `q` 级处理残差：

```text
r0 = z
kq = argmin_k ||rq - e(q,k)||²
qq = e(q,kq)
r(q+1) = rq - qq
最终量化输出 = q0 + q1 + ... + q15
```

最近邻 `argmin` 不可导。当前实现采用：

- 标准 STE：联合训练时令量化层对 Encoder 的近似导数为恒等映射；
- Commitment loss：约束 Encoder latent 靠近选中的码字；
- EMA（decay=0.99）：更新码本聚类中心；
- K-means 初始化；
- dead-code threshold=2；
- 多卡训练时同步码本统计。

RVQ 仅在正常模式工作。`bypass_rvq=True` 时：

```text
quantized = encoder_latent
indices = -1
commitment_loss = 0
```

## 6. Decoder 结构

当前生产训练路径使用 `decoder_upsample_mode=convtranspose`，没有 linear/cubic 插值平滑。

```text
量化或 bypass latent [B,64,T/320]
  │
  ├─ Causal Conv1d 64→256, K7
  │
  ├─ Decoder Block4: ConvTranspose1d 256→128, K16, S8
  │    └─ 3×标准 ResidualUnit(128, K7/K1, dilation=1/3/9)
  │
  ├─ Decoder Block3: ConvTranspose1d 128→64, K10, S5
  │    └─ 3×标准 ResidualUnit(64, K7/K1, dilation=1/3/9)
  │
  ├─ Decoder Block2: ConvTranspose1d 64→32, K8, S4
  │    └─ 3×标准 ResidualUnit(32, K7/K1, dilation=1/3/9)
  │
  ├─ Decoder Block1: ConvTranspose1d 32→16, K4, S2
  │    └─ 3×标准 ResidualUnit(16, K7/K1, dilation=1/3/9)
  │
  └─ Causal Conv1d 16→1, K7
       ↓
     重建音频 [B,1,T]
```

Decoder 的残差单元仍是标准卷积：

```text
Conv1d C→C, K7, dilation=d
 → ReLU
 → Conv1d C→C, K1
 → ReLU
 → residual add
```

## 7. Decoder 卷积参数量和 MAC

ConvTranspose1d 的 MAC 按稳态输入贡献计算；因果裁剪边界会使极短片段的实际值略有差异。

| 模块 | Weight | Bias | 卷积参数合计 | MAC/s | 20 ms MAC |
|---|---:|---:|---:|---:|---:|
| Decoder Initial Conv | 114,688 | 256 | 114,944 | 5.7344 M | 0.114688 M |
| Decoder Block4（×8） | 917,504 | 896 | 918,400 | 183.5008 M | 3.670016 M |
| Decoder Block3（×5） | 180,224 | 448 | 180,672 | 229.376 M | 4.58752 M |
| Decoder Block2（×4） | 40,960 | 224 | 41,184 | 229.376 M | 4.58752 M |
| Decoder Block1（×2） | 8,192 | 112 | 8,304 | 114.688 M | 2.29376 M |
| Decoder Final Conv | 112 | 1 | 113 | 1.792 M | 0.03584 M |
| **Decoder 合计** | **1,261,680** | **1,937** | **1,263,617** | **764.4672 M** | **15.289344 M** |

编解码器卷积合计：

| 范围 | Weight | Bias | 卷积参数合计 | MAC/s | 20 ms MAC |
|---|---:|---:|---:|---:|---:|
| Encoder + Decoder | 1,423,552 | 5,281 | 1,428,833 | 968.9856 M | 19.379712 M |

这些数字不包含 RVQ 距离搜索、判别器、STFT/Mel/共振峰损失、FiLM 参数、激活、残差加法和数据搬运。

## 8. 判别器与训练损失

判别器只在 GAN 阶段参与训练，不属于部署编码器/解码器：

- Multi-Scale waveform discriminators；
- STFT discriminator；
- 判别器特征匹配 loss；
- generator adversarial loss。

重建侧主要损失包括：

- waveform L1；
- multi-spectral / Mel loss；
- MR-STFT loss，并记录各 STFT scale；
- SI-SDR 和 correlation；
- 双尺度倒谱 spectral envelope；
- F1/F2/F3 formant peak；
- voiced high-band、noise-floor、click 等辅助项。

## 9. 当前四阶段训练中的数据流和冻结关系

| 阶段 | 实际数据流 | Encoder | RVQ | Decoder | Discriminator |
|---|---|---|---|---|---|
| bypass Stage-1 | `Encoder→Decoder` | 训练 | 绕过 | 训练 | 不训练 |
| bypass Stage-2 | `Encoder→Decoder` | 冻结 | 绕过 | 训练 | 训练 |
| RVQ Stage-1 校准 | `Encoder→RVQ` | 冻结 | 仅 EMA 更新 | 不执行、冻结 | 不训练 |
| RVQ Stage-2 | `Encoder→RVQ→Decoder` | 冻结 | 冻结 | 训练 | 训练 |

RVQ 校准阶段没有反向传播和优化器更新，只允许码本 EMA、cluster size、embed average 和 dead-code replacement 状态变化。最后阶段必须训练 Decoder，否则加入量化误差后生成器没有任何可适配参数。

## 10. 离线与流式模型

普通 `SoundStream` 对完整训练片段执行因果卷积。`FrameStreamingSoundStream` 继承相同的 Encoder、RVQ 和 Decoder 权重，但按 320 个采样点（20 ms）维护状态：

- Encoder 保存每层历史卷积输入；
- 每个 20 ms 输入帧产生一个 64 维 latent；
- RVQ 对每个 latent 帧产生 16 个 index；
- ConvTranspose1d 保存重叠输出状态；
- 连续帧之间不重新清空状态。

当前四阶段脚本使用普通 `recon_pretrain` / `gan_pretrain` 模型；它没有执行独立的 streaming fine-tune 阶段。

## 11. 部署边界

本文描述的是当前 PyTorch 训练网络。部署到 RTL 前仍需单独确认：

1. Conv/Depthwise/低秩逐点卷积的实际权重排列；
2. Bias、activation scale、requant、rounding 和 INT8 饱和规则；
3. 残差支路 scale 对齐；
4. 16×16×64 RVQ 码本的 INT8 scale 与索引打包；
5. PyTorch 浮点/量化参考与 RTL 的逐层整数对拍。

模型能够严格加载 checkpoint 只说明参数名称和形状兼容，不能代替上述 bit-exact 验证。
