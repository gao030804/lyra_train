# RVQ INT8 PTQ 与逐级整数参考

从最终 Encoder QAT checkpoint 开始；冻结全部模型参数和码本 EMA，不重新训练。
工具读取实际 RVQ 层数、每层 K 与 lookup 维度，不采用文档中旧的 16×16×64 常量。
当前 9 级 [256,128×8]、32D 的码本原始存储为 40,960 字节；norm 为 5,120 字节。

## 文件

- `tools/analyze_rvq_codebook_quantization.py`：校准scale、独立验证、试听WAV、质量报告与条件导出。
- `tools/integer_rvq_reference.py`：NumPy纯整数参考与二进制打包。
- `tests/test_integer_rvq_reference.py`：边界、独立穷举距离/PyTorch与打包验证。

## 校准与验证数据

准备两个 CSV（或TSV）manifest，每行一条音频，列名 `source`，值为真实路径。
建议校准用训练流程原有validation split，验证用原有test split；两份数据必须互不重叠。
同一音频不得使用拷贝改名后同时放入两份manifest。工具拒绝重复路径及交集。
每条固定取中心4秒，独立零状态前向；不是stateful整句测试。可用 `--segment-seconds` 调整。
如要公平比较现有10条长音频，将其 `selected_test_files.csv` 作为evaluation manifest，
校准manifest另用validation文件。保存manifest以保证实验可重复。

## 执行

```bash
cd /home/deploy/gyh/lyra_md
conda activate lyra
python -m unittest discover -s tests -p test_integer_rvq_reference.py -v

# CAL_MANIFEST、EVAL_MANIFEST 指向上面准备好的真实文件。
python tools/analyze_rvq_codebook_quantization.py \
  --checkpoint results/hardware-qat-w8a16-reference-fix-20260922-222030/best_full_qat.pt \
  --calibration-manifest "$CAL_MANIFEST" \
  --evaluation-manifest "$EVAL_MANIFEST" \
  --output-dir "results/rvq-int8-ptq-$(date +%Y%m%d-%H%M%S)" \
  --device cuda --segment-seconds 4 \
  --percentiles 100 99.99 99.9 99.5 --export
```

输出目录必须是新目录，防止失败后的旧参数包被误用。
按stage逐个搜索scale，每个候选运行完整RVQ和Decoder。选择先满足音频保持条件，
再比较完整VQout NMSE、前三层index flip和码本元素NMSE。这是一次坐标搜索，非全局最优保证。
默认质量阈值为平均aligned SI-SDR下降不超过0.1 dB、平均相关系数下降不超过0.01、
平均VQout NMSE不超过0.05；这些是初始实验阈值，不是已验证的硬件部署标准。
逐文件指标也保存以检查平均数掩盖的退化。SI-SDR/相关使用±20ms范围内有符号互相关对齐。

## 输出

- `rvq_ptq_report.json`：每个scale候选、每文件指标、q00等index flip、码本与selected-codeword NMSE、
  VQout NMSE、残差RMS/饱和计数、音频保持结果。
- `00_original.wav`、`00_fp_codebook.wav`、`00_int8_codebook.wav`等：同一音频对照，FLOAT WAV避免写文件截断。
- 只有 `--export` 且独立验证通过才生成 `export/`：
  `rvq_codebook_int8.bin`、`rvq_codebook_packed_4x8_int8.bin`、`rvq_codebook_norm_int32.bin`、
  `rvq_stage_requant_params.bin`、`rvq_manifest.json`。
- `golden_00.npz`等：输入A16 query、每级residual、所有码字score、减法后残差、indices及最终INT16 latent。
  张量采用 `[B,T,D]`，score采用 `[B,T,Kq]`，indices采用 `[B,T,Q]`。

## 数值契约与边界

每stage一个对称scale，zero=0，INT8码本，INT16查询；
score=N−2R·E使用signed INT32，tie选择最小index。
减法先保留17位，stage scale比值转成INT32 multiplier和0..63 shift，
INT64乘积、half-away-from-zero舍入后饱和INT16。最后一级减法残差仅为诊断。
编码端输出与解码端输出均按indices取同一INT8码本，逐级requant到固定output scale，
INT32求和后饱和INT16。不能使用输入减最终残差代替解码定义。

二进制均为小端；原始码本为stage/codeword/dimension排列。4×8版本为
output_group/reduction_group/4/8；两种权重布局择一使用，padding不能参与argmin。
每级requant记录恰好10字节：`<iBiB` (next multiplier, next shift, output multiplier, output shift)。
最后一级next参数为(1,0)，不执行下一stage。

工具验证 NumPy dot-product 与独立NumPy平方距离、PyTorch平方距离整数路径的indices与output一致；
另外用indices-only查表验证解码端一致，并检查FP32 master码本未变化。
这不等价于原FP32码本与INT8码本index相同；两者差异在index flip中报告。

当前只量化RVQ codebook与RVQ边界残差；64→32、32→64外部projection和音频Decoder仍是浮点。
输入需先按manifest.input_scale转换到lookup域A16。真实Encoder到RVQ的整数投影接口需另行实现。
现有训练checkpoint不修改；码本导出包与golden不是训练checkpoint。
RTL对拍尚需testbench接入导出格式，报告始终标明 rtl_verified=false。
若PTQ验证失败先分析报告再决定codebook-aware微调；此次未重新开启EMA或增加训练阶段。
