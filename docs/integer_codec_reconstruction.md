# 导出参数的整数编码、浮点解码软件验证

## 范围和边界

入口：`tools/reconstruct_integer_codec.py`；整数计算核心：`tools/integer_codec_package.py`。

- Encoder 的图结构来自匹配 checkpoint，但卷积权重、bias、M/S 从导出二进制读取；Residual alignment 读取 manifest。仅支持 W8A16/ACC40、零填充、共享 scale 的 ReLU。
- 校验 checkpoint SHA，并在临时目录重新生成 Encoder 包，检查导出二进制及计算元数据一致。不会覆盖原包。
- 64→32 输入投影从同一 checkpoint 量化为 W8A16/ACC40，输出 scale 对齐 RVQ；保存到本轮 `projection_export/` 后重新读回，实际使用读回的参数。
- RVQ 读取 V1 INT8 码本、INT32 norm、各级 M/S；检查 SHA、原始/packed 布局、norm、manifest 与二进制的一致性。与独立平方距离 oracle 比较。
- 保存索引并重新加载查表，得到 INT16 RVQ 向量。按校准后的 output_scale 反量化，然后使用冻结的浮点 32→64 输出投影和 Decoder。不要混用浮点码本解码。
- 不训练、不调 scale、不选 checkpoint、不修改质量门槛，也不授予部署资格。当前候选最终音质测试未通过，仍是诊断包。

`offline` 整段执行。`compare` 在同一 PCM 上额外运行 320-sample 分帧编码和有状态 Decoder：逐层整数输出、投影输出、索引和 RVQ 输出必须完全相同；浮点 Decoder 使用 `atol=1e-5, rtol=1e-4` 比较。关闭 TF32。匹配失败不得称为通过。

Encoder 每层保存有限长度历史；支持本模型 320 样本帧对应的整除步幅，不支持任意非对齐 chunk。Decoder 使用原有 `stream_module` 的状态接口（ConvTranspose overlap 等），不每帧重置。每个文件重新初始化一次状态。

起点为冷启动；报告裁剪不带额外前文。输入只接受 16 kHz，单声道或多声道均值。不自动重采样。PCM 按 half-away 舍入至 INT16。尾部补零至 320 的倍数；离线/分帧执行相同 padding，试听及音质指标只计有效样本。长度不符合预期会报错，不静默裁剪对齐。未量化 Decoder，不是全整数解码器。

## 同步与检查

将上述两个 tools 文件和 `tests/test_integer_codec_package.py` 同步到服务器对应位置（不是只复制本文档）。

```bash
cd /home/deploy/gyh/lyra_md
conda activate lyra
python -m py_compile tools/integer_codec_package.py tools/reconstruct_integer_codec.py
python -m unittest discover -s tests -p test_integer_codec_package.py -v
python tools/reconstruct_integer_codec.py --help
```

本机的合成参数单元测试不是服务器真实模型测试。若报 Encoder metadata mismatch / 缺少 alignment，先检查同步版本，再用匹配 QAT checkpoint 导出至新的 Encoder 目录；不要删除一致性检查。

## 公共参数

在同一个 Bash 会话设置：

```bash
ENCODER_DIR="results/hardware-export-w8a16-reference-fix"
RVQ_DIR="results/rvq-output-cal-step50-20260927-155239/rvq_diagnostic_export"
DECODER_CKPT="results/hardware-qat-w8a16-reference-fix-20260922-222030/best_full_qat.pt"
TEST_REPORT="results/rvq-calibrated-test-20260927-155841/calibrated_test_report.json"
COMMON=(
  --encoder-dir "$ENCODER_DIR"
  --rvq-dir "$RVQ_DIR"
  --checkpoint "$DECODER_CKPT"
  --report "$TEST_REPORT"
  --device cuda
)
mkdir -p logs
```

## 一条 4 秒离线重建

复用旧报告编号 02 的文件和 start_sample，而不是重新随机裁剪：

```bash
OUT="results/integer-codec-one-$(date +%Y%m%d-%H%M%S)"
LOG="logs/$(basename "$OUT").log"
nohup setsid env CUDA_VISIBLE_DEVICES=0 python -u tools/reconstruct_integer_codec.py \
  "${COMMON[@]}" --mode offline --file-index 2 --num-files 1 \
  --segment-seconds 4 --output-dir "$OUT" </dev/null >"$LOG" 2>&1 &
echo "PID=$! OUT=$OUT LOG=$LOG"
tail -n 80 -F "$LOG"
```

此次输出编号重新从 `00` 开始，对应源报告编号 02；以 report.json 的 path/start_sample 为准。

纯整数 Encoder 使用 CPU NumPy，浮点 Decoder 使用指定设备；不是高性能实时实现。第一次单文件可能需要较长时间，不能用这个脚本的运行速度估计 RTL 速度。

## 同一批 10 条、4 秒离线重建

第一条正常后再启动，不要同时占用同一 GPU：

```bash
OUT="results/integer-codec-ten-$(date +%Y%m%d-%H%M%S)"
LOG="logs/$(basename "$OUT").log"
nohup setsid env CUDA_VISIBLE_DEVICES=0 python -u tools/reconstruct_integer_codec.py \
  "${COMMON[@]}" --mode offline --num-files 10 \
  --segment-seconds 4 --output-dir "$OUT" </dev/null >"$LOG" 2>&1 &
echo "PID=$! OUT=$OUT LOG=$LOG"
tail -n 80 -F "$LOG"
```

## 分帧一致性：先一条，再十条完整长音频

先单条 4 秒验证：

```bash
OUT="results/integer-codec-stream-one-$(date +%Y%m%d-%H%M%S)"
LOG="logs/$(basename "$OUT").log"
nohup setsid env CUDA_VISIBLE_DEVICES=0 python -u tools/reconstruct_integer_codec.py \
  "${COMMON[@]}" --mode compare --file-index 2 --num-files 1 \
  --segment-seconds 4 --output-dir "$OUT" </dev/null >"$LOG" 2>&1 &
echo "PID=$! OUT=$OUT LOG=$LOG"
tail -n 80 -F "$LOG"
```

通过后，改为 `--mode compare --file-index 0 --num-files 10 --segment-seconds 0`，并设置一个新的 OUT/LOG。**segment-seconds=0 表示从文件开头处理完整文件，忽略裁剪起点**；此时不能与旧 4 秒结果直接做数值差比较。

## 输出与判断

每次新目录，已有目录拒绝覆盖。每个 `00/`、`01/` 等目录包含：

- `original.wav`：显式 PCM16 化后的原始参考，以 FLOAT WAV 保存。
- `qat_fp_codebook.wav`：完整 QAT Encoder + 浮点投影、码本 + 浮点 Decoder 参考。
- `integer_offline.wav`：导出整数参数整段重建。
- `integer_stream.wav`：仅 compare 模式，连续有状态分帧重建。
- `indices.npy`：码字索引，不是已打包传输比特流。
- `integer_latents.npz`：整数 Encoder、投影、RVQ 输出及索引。

顶层 `report.json` 包含文件/裁剪、参数 SHA、图结构、饱和计数、SI-SDR、aligned SI-SDR、相关性、MSE/MAE、峰值、音频 clip 比例、VQ NMSE、分帧一致性。Encoder 中存在饱和不自动判为失败（可能是已校准裁剪），如实记录。

完成时打印 `Software diagnostic complete`。compare 模式还应检查 `integer_stream_exact=true` 和 `decoder_stream_close=true`。浮点 Decoder 超出容差会保留 WAV/report 并退出失败；整数不一致直接指出 frame/layer 并停止，不生成通过结论。

验证的是软件数值、导出包和流式一致性，不是 RTL bit-exact，也不是音质合格证明。
