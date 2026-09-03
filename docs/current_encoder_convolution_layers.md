# 当前 SoundStream 编码器卷积结构与参数量

## 1. 本文对应的实际训练配置

本文描述当前 `train_soundstream.py` 构造的非量化 SoundStream 编码器，不包含 Decoder、RVQ、判别器及损失网络。

| 配置项 | 当前值 |
|---|---:|
| 输入 | 单声道 16 kHz 波形 |
| `channels` | 16 |
| `channel_mults` | `(2, 4, 8, 16)` |
| 各 Block 通道 | `16→32→64→128→256` |
| `strides` | `(2, 4, 5, 8)` |
| 总下采样倍数 | `2×4×5×8 = 320` |
| 编码帧率 | `16000/320 = 50 Hz` |
| `codebook_dim` | 64 |
| 残差单元 dilation | `(1, 3, 9)` |
| DSCNN Block（零基索引） | `(2, 3)`，即 Block3/4 |
| DSCNN revision | 2：`Depthwise → Pointwise`，二者之间无激活 |
| 激活函数 | ReLU |
| Squeeze-Excite / Local Attention / Gate Loop | 当前均未启用 |

将 Depthwise 和 Pointwise 视为独立物理卷积后，当前 Encoder 共包含 **38 个 Conv1d**。

以 4 秒、16 kHz 输入为例，时间长度依次为：

```text
64000 → 64000 → 32000 → 8000 → 1600 → 200 → 200
 输入     Stem      B1       B2      B3     B4    64维潜变量
```

## 2. 卷积和参数量计算规则

PyTorch `Conv1d` 权重布局和参数公式：

### 普通 Conv1d

```text
weight shape = [Cout, Cin, K]
weight params = Cout × Cin × K
bias params = Cout
```

### Depthwise Conv1d

`groups=Cin`，并且当前 depth multiplier 为 1：

```text
weight shape = [Cin, 1, K]
weight params = Cin × K
bias params = Cin
```

### Pointwise Conv1d

```text
weight shape = [Cout, Cin, 1]
weight params = Cout × Cin
bias params = Cout
```

本文参数量均包含 Bias。MAC/s 按每个输出采样位置执行一次卷积计算估算，不计 Bias、ReLU、padding、残差相加和内存搬运。

## 3. 因果卷积方式

所有编码卷积均由 `CausalConv1d` 包装。左侧 padding 为：

```text
causal_padding = dilation × (K - 1) + (1 - stride)
```

只补历史样本，不使用未来样本。普通推理默认使用 reflect padding；流式推理则缓存相同长度的历史输入。

残差时域卷积 `K=7` 时：

| dilation | 左侧历史/padding |
|---:|---:|
| 1 | 6 |
| 3 | 18 |
| 9 | 54 |

每个下采样卷积采用 `K=2×stride`，其左侧 padding 恰好等于 stride，因此对可整除输入产生严格的 `T/stride` 输出长度。

## 4. 编码器总体结构

```text
Input [B,1,T]
  │
  ├─ Stem: Causal Conv1d 1→16, K7, S1
  │
  ├─ Encoder Block1: 16→32, S2，普通卷积
  │    ├─ Residual Unit, dilation 1
  │    ├─ Residual Unit, dilation 3
  │    ├─ Residual Unit, dilation 9
  │    └─ Causal Conv1d 16→32, K4, S2
  │
  ├─ Encoder Block2: 32→64, S4，普通卷积
  │    ├─ Residual Unit, dilation 1
  │    ├─ Residual Unit, dilation 3
  │    ├─ Residual Unit, dilation 9
  │    └─ Causal Conv1d 32→64, K8, S4
  │
  ├─ Encoder Block3: 64→128, S5，DSCNN
  │    ├─ DSCNN Residual Unit, dilation 1
  │    ├─ DSCNN Residual Unit, dilation 3
  │    ├─ DSCNN Residual Unit, dilation 9
  │    └─ Depthwise 64→64, K10, S5 → Pointwise 64→128
  │
  ├─ Encoder Block4: 128→256, S8，DSCNN
  │    ├─ DSCNN Residual Unit, dilation 1
  │    ├─ DSCNN Residual Unit, dilation 3
  │    ├─ DSCNN Residual Unit, dilation 9
  │    └─ Depthwise 128→128, K16, S8 → Pointwise 128→256
  │
  └─ Final: Causal Conv1d 256→64, K3, S1
       ↓
     Encoder latent [B,64,T/320]
```

## 5. 残差单元内部过程

### Block1/2：普通卷积残差单元

```text
x
 └─ Causal Conv1d(C→C, K7, dilation=d)
     → ReLU
     → Causal Conv1d(C→C, K1)
     → ReLU
     → 与 x 相加
```

### Block3/4：DSCNN 残差单元

```text
x
 └─ Depthwise Causal Conv1d(C→C, K7, groups=C, dilation=d)
     → Pointwise Causal Conv1d(C→C, K1)
     → ReLU
     → 第二个 Causal Conv1d(C→C, K1)
     → ReLU
     → 与 x 相加
```

注意：Depthwise 和其紧随的 Pointwise 之间没有 ReLU。第一个 ReLU 位于完整 DSCNN 之后。第二个 `1×1` 卷积是原 Residual Unit 自带的通道变换层，并未因 DSCNN 改造而删除。因此 Block3/4 的每个残差单元实际包含三次卷积：一个 Depthwise 和两个 Pointwise。

## 6. 各物理卷积层参数量

### 6.1 Stem

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Stem Conv | `[16,1,7]` | `7/1/1/1` | 112 | 16 | **128** |

### 6.2 Encoder Block1：16→32，stride=2

每个残差单元包含普通 `K7` 卷积和 `K1` 卷积。

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Residual-1 temporal | `[16,16,7]` | `7/1/1/1` | 1,792 | 16 | 1,808 |
| Residual-1 pointwise | `[16,16,1]` | `1/1/1/1` | 256 | 16 | 272 |
| Residual-2 temporal | `[16,16,7]` | `7/1/3/1` | 1,792 | 16 | 1,808 |
| Residual-2 pointwise | `[16,16,1]` | `1/1/1/1` | 256 | 16 | 272 |
| Residual-3 temporal | `[16,16,7]` | `7/1/9/1` | 1,792 | 16 | 1,808 |
| Residual-3 pointwise | `[16,16,1]` | `1/1/1/1` | 256 | 16 | 272 |
| Downsample | `[32,16,4]` | `4/2/1/1` | 2,048 | 32 | 2,080 |
| **Block1总计** |  |  | **8,192** | **128** | **8,320** |

### 6.3 Encoder Block2：32→64，stride=4

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Residual-1 temporal | `[32,32,7]` | `7/1/1/1` | 7,168 | 32 | 7,200 |
| Residual-1 pointwise | `[32,32,1]` | `1/1/1/1` | 1,024 | 32 | 1,056 |
| Residual-2 temporal | `[32,32,7]` | `7/1/3/1` | 7,168 | 32 | 7,200 |
| Residual-2 pointwise | `[32,32,1]` | `1/1/1/1` | 1,024 | 32 | 1,056 |
| Residual-3 temporal | `[32,32,7]` | `7/1/9/1` | 7,168 | 32 | 7,200 |
| Residual-3 pointwise | `[32,32,1]` | `1/1/1/1` | 1,024 | 32 | 1,056 |
| Downsample | `[64,32,8]` | `8/4/1/1` | 16,384 | 64 | 16,448 |
| **Block2总计** |  |  | **40,960** | **256** | **41,216** |

### 6.4 Encoder Block3：64→128，stride=5，DSCNN

三个残差单元只有 dilation 不同，因此参数量相同。

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Residual-1 depthwise | `[64,1,7]` | `7/1/1/64` | 448 | 64 | 512 |
| Residual-1 DSCNN pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Residual-1 second pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Residual-2 depthwise | `[64,1,7]` | `7/1/3/64` | 448 | 64 | 512 |
| Residual-2 DSCNN pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Residual-2 second pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Residual-3 depthwise | `[64,1,7]` | `7/1/9/64` | 448 | 64 | 512 |
| Residual-3 DSCNN pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Residual-3 second pointwise | `[64,64,1]` | `1/1/1/1` | 4,096 | 64 | 4,160 |
| Downsample depthwise | `[64,1,10]` | `10/5/1/64` | 640 | 64 | 704 |
| Downsample pointwise | `[128,64,1]` | `1/1/1/1` | 8,192 | 128 | 8,320 |
| **Block3总计** |  |  | **34,752** | **768** | **35,520** |

### 6.5 Encoder Block4：128→256，stride=8，DSCNN

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Residual-1 depthwise | `[128,1,7]` | `7/1/1/128` | 896 | 128 | 1,024 |
| Residual-1 DSCNN pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Residual-1 second pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Residual-2 depthwise | `[128,1,7]` | `7/1/3/128` | 896 | 128 | 1,024 |
| Residual-2 DSCNN pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Residual-2 second pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Residual-3 depthwise | `[128,1,7]` | `7/1/9/128` | 896 | 128 | 1,024 |
| Residual-3 DSCNN pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Residual-3 second pointwise | `[128,128,1]` | `1/1/1/1` | 16,384 | 128 | 16,512 |
| Downsample depthwise | `[128,1,16]` | `16/8/1/128` | 2,048 | 128 | 2,176 |
| Downsample pointwise | `[256,128,1]` | `1/1/1/1` | 32,768 | 256 | 33,024 |
| **Block4总计** |  |  | **135,808** | **1,536** | **137,344** |

### 6.6 最终潜变量投影

| 层 | Weight shape | K/S/D/G | Weight | Bias | 合计 |
|---|---|---|---:|---:|---:|
| Final Conv | `[64,256,3]` | `3/1/1/1` | 49,152 | 64 | **49,216** |

## 7. 参数量与MAC汇总

| 部分 | Weight | Bias | 总参数 | 估算 MAC/s |
|---|---:|---:|---:|---:|
| Stem | 112 | 16 | 128 | 1.792 M |
| Encoder Block1 | 8,192 | 128 | 8,320 | 114.688 M |
| Encoder Block2 | 40,960 | 256 | 41,216 | 229.376 M |
| Encoder Block3（DSCNN） | 34,752 | 768 | 35,520 | 55.373 M |
| Encoder Block4（DSCNN） | 135,808 | 1,536 | 137,344 | 42.138 M |
| Final Conv | 49,152 | 64 | 49,216 | 2.458 M |
| **卷积总计** | **268,976** | **2,768** | **271,744** | **445.824 M MAC/s** |

4 秒训练片段约执行 `1.783 G MAC`，这里仍不含 RVQ、Decoder和损失计算。

## 8. DSCNN改造带来的减少

若 Block3/4 保持原普通卷积，它们的卷积参数量分别为：

| Block | 原普通卷积参数 | 当前 DSCNN 参数 | 减少 |
|---|---:|---:|---:|
| Block3 | 180,736 | 35,520 | 80.35% |
| Block4 | 918,528 | 137,344 | 85.05% |
| 整个 Encoder 卷积 | 1,198,144 | 271,744 | **77.32%** |

由于 Block3/4 已处在较低时间分辨率，参数减少比例高于实际运算量减少比例。整个 Encoder 从约 `761.190 M MAC/s` 降为 `445.824 M MAC/s`，估算下降约 **41.43%**。

## 9. 当前硬件映射时的重点

1. 普通卷积权重可按 `[Cout, Cin, K]` 展开；Depthwise 权重必须单独按 `[Cin, 1, K]` 处理。
2. DSCNN 不能被误认为只有两层：残差单元中还有原结构保留的第二个 `C→C, K1` 卷积。
3. Depthwise 与第一个 Pointwise 之间没有 ReLU；ReLU 位于第一个 Pointwise 之后。
4. dilation 只改变取样地址和历史缓存长度，不改变卷积核参数量。
5. stride 下采样只发生在每个 Block 最后一组卷积；前三个残差单元保持时间长度不变。
6. 本文总数只统计卷积 Weight 与 Bias，不包含归一化、RVQ codebook、FiLM以及其他非卷积参数。
