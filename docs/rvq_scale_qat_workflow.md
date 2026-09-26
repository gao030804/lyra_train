# Projection整数化与RVQ scale-only QAT

## 2026-09-27 批量微调与双门槛（最新）

不再仅在通过时保存：每次验证保存`scales_step_XXXXX.pt`（含step0）、
`best_candidate_scales.pt`（全部step音频优先）、`best_audio_candidate_scales.pt`（仅音频门槛通过）。
它们只是诊断候选，不能直接作为通过验收的参数；candidate可能是step0。
`best_scales.pt`严格要求mean/P90 NMSE<=0.04/0.08；
`best_experimental_scales.pt`使用0.05/0.10，其他SI-SDR/零饱和条件相同。
默认`--gate-profile strict`不变；显式`--gate-profile experimental`才允许以实验候选进入最终测试。
最终测试仍只评验证集选定的一个候选，不用测试集重新选参数。
实验导出写入`experimental_export/`，严格导出写入`export/`；manifest同时记录strict retention、
experimental retention、门槛和`deployment_approved=false`，RTL验证仍需单独完成。
不能把旧step200日志恢复成checkpoint；需要重新运行才能保存其新实验对应参数。

新版默认batch-size=4（逐文件梯度累积取平均，文件NMSE等权）、lr=0.0003、
max-ratio=1.08、eval-every=100、distance-weight=0.1、codebook-weight=0.02。
steps仍默认0以保留PTQ优先政策；训练需显式`--steps 1000`。
仅优化scale，码本及EMA保持冻结。日志输出平均loss、加权辅助loss、实际batch和梯度范数，
验证summary增加q00/q01/q02的flip与逐帧能量加权flip。

下一轮推荐在原max-scale初始化和原三份CSV基础上添加：

```bash
--frontend rvq-only --scale-mode v1 \
--steps 1000 --eval-every 100 --batch-size 4 \
--lr 0.0003 --max-ratio 1.08 \
--distance-weight 0.1 --codebook-weight 0.02 \
--gate-profile experimental
```

step0复现检查继续强制执行；通过实验门槛不代表听感或硬件已验收。
若新一轮仍长期停留在约0.047，不应自动继续堆步数；先试听已保存候选再决定。

## 2026-09-27 RVQ-only 隔离实验

新增 `--frontend rvq-only --scale-mode v1`，固定原QAT浮点执行Encoder和浮点Projection输出，
不运行整数Encoder、不再次做PCM舍入；和独立PTQ使用相同的中心裁剪及浮点解码数值路径。
默认frontend仍为integer，不影响原完整链路实验。
RVQ-only step0强制逐文件验证路径、start_sample、SI-SDR/corr delta、VQout NMSE、饱和数
与初始化PTQ报告一致（浮点rtol=1e-4、atol=1e-5）；不一致即停止，禁止带着不同baseline训练。
使用max-scale报告 `results/rvq-v1-maxscale-20260926-235517/rvq_ptq_report.json`，
并继续使用 `results/rvq-scale-qat-20260924-092506-manifests` 中三个CSV。
建议首轮 `--steps 500 --eval-every 100 --lr 0.001 --max-ratio 1.15`。
只有9个scale可训练，输出scale和码本保持冻结；没有合格checkpoint则不评测试、不导出。
RVQ-only导出仅含RVQ，projection_quantized=false，golden输入是RVQ INT16 query，
不含伪造的整数Encoder/Projection输出，不代表完整端到端整数链路已通过。
如需继续训练，不支持resume，必须显式新建实验；保留原max-scale结果不覆盖。

## 2026-09-26 更新（优先于下文旧 V2 说明）

默认 `--scale-mode v1 --steps 0`：只做完整整数前端/PTQ验证，不训练。
V1 每级 codebook/residual 共用 scale；V2 分离 scale 必须显式指定 `--scale-mode v2`。
显式设置 `--steps 2000` 才允许验证失败后训练；step0通过时仍跳过训练，直接评最终测试。
测试失败不导出，不能根据测试集结果反复选 scale。
先缓存验证集，只有需要训练时才缓存训练集；旧日志中固定32条训练缓存优先的顺序已改变。
V1训练时 residual scale 随 codebook scale 变化；Projection只更新输出单位的乘子/移位，
权重、Bias和几何映射不变。导出Projection与最佳scale使用一致的输出单位。
合格候选改为音频优先（mean/P10/worst SI-SDR、correlation），再比较VQout NMSE。
报告新增分来源饱和计数、真正按逐帧能量加权的flip、selected-codeword误差和能量。

推荐首先重跑独立 `analyze_rvq_codebook_quantization.py` 的 V1 PTQ。
该工具保持浮点Projection，不与完整整数前端报告混为同一实验；已增加max-scale校准基线、
P10/worst门槛及输入/残差/输出零饱和门槛。旧报告的PASS不能自动等同新版门槛PASS。
只训练9个scale无法保证补偿整数Encoder/Projection的误差，完整前端失败应先定位边界。

9级K=[256,128×8]、D=32：INT8为40960字节，FP32为163840字节。
这不包含norm、requant、Projection、运行state。两个codebook布局文件是同一数据的不同排列，
不要默认两份都烧录。

新增 `tools/compare_rvq_integer_goldens.py`：

```bash
python tools/compare_rvq_integer_goldens.py \
  --golden "$PTQ_DIR/golden_00.npz" \
  --rtl-dump "$RTL_DUMP" \
  --report "$COMPARE_REPORT"
```

RTL_DUMP必须是真实仿真转换得到的NPZ（不能用golden冒充），数组名/shape与golden一致，
包括indices、output_int16及各stage residual/scores/after_subtract。
缺失数组、shape错误、浮点payload均拒绝；任何逐元素不一致返回失败。
PTQ/QAT报告的RTL mismatch默认null而不是0，未执行RTL不能声称bit-exact。
该比较器不负责运行RTL或验证dump来源；目前仍需外部仿真器产生数据。

同步清单还需加入 `tools/compare_rvq_integer_goldens.py` 和 `tests/test_rvq_golden_compare.py`。
回归测试：两个原RVQ测试以及 `python -m unittest discover -s tests -p test_rvq_golden_compare.py -v`。

## 实现范围

新增独立 `tools/train_rvq_quant_scales.py`，从已通过Encoder W8A16/ACC40 QAT的checkpoint和
`rvq_ptq_report.json`读取初值。只训练每级一个delta，当前结构共9个参数。
Encoder、两端Projection、Decoder、FP32 master码本全部冻结；码本前向freeze_codebook=True，eval模式禁止EMA更新。
训练结束核对master码本数值没有变化。

参数化 s=s0*exp(log(1.15)*tanh(delta))，严格范围为 **[s0/1.15,s0*1.15]**，
下界约0.869565*s0，并非0.85*s0。round使用half-away STE，FP32 master不经过detach整个量化结果的写法。
损失为 VQout NMSE + 0.2*top-8距离KL + 0.02*码本NMSE。
前三层KL权重1/1/0.8，随后0.65/0.55/0.45/0.35/0.30/0.25。
KL用teacher top-k平均距离归一化后再加temperature，避免不同残差能量使固定温度失效。
学生hard argmin选择用于前向重建，距离KL提供码字排序的可导监督。
当前实现Phase A；是否加入后续小权重音频重建微调，根据真实整数验证结果决定。

## 固定residual scale的硬件含义

原PTQ格式v1共享codebook/residual scale，scale-only阶段采用格式v2：
固定每级residual存储scale为原PTQ值r_q，学习codebook scale s_q。

1. R_q按r_q/s_q重定标并饱和到搜索INT16向量。
2. score = norm(INT8 codeword) - 2*INT16 search dot INT8 codeword，INT32 score。
3. selected INT8 codeword按s_q/r_q转回残差单位；与原R_q相减保留17位。
4. 按固定r_q/r_(q+1)重定标到下级INT16。
5. indices-only lookup按s_q/s_out求和重建，INT32 accumulation、INT16输出。

每级新增10字节 `rvq_search_subtract_requant_params.bin`：小端 `<iBiB>`。
原 `rvq_stage_requant_params.bin` 仍为10字节/stage，记录固定残差级间转换与输出转换。
RTL需要支持这些额外转换，不能把格式v2作为原共享scale的v1直接读取。

## Projection与前端

`tools/rvq_scale_qat.py`包含真实模块树的整数Encoder执行、64→32 Linear的W8A16/ACC40 PTQ。
Encoder从真实s16 PCM /32768开始，递归处理Conv、低秩、depthwise、ReLU、Residual。
Projection权重per-Cout INT8、Bias ACC40装入INT64容器、输出为固定r_0的INT16。
ACC40乘INT32乘子使用Python大整数容器，防止72位乘积溢出INT64。
Projection参数固定，末端32→64和音频Decoder仍为浮点。
整数前端按文件中心片段独立零状态计算并缓存；目前不是stateful长流验证。
缓存阶段CPU整数参考较慢，会逐文件打印进度。

## Linux操作

同步以下文件：
`tools/train_rvq_quant_scales.py`、`tools/rvq_scale_qat.py`、`tools/integer_rvq_reference.py`、
`tools/analyze_rvq_codebook_quantization.py`、`tools/export_encoder_integer_goldens.py`、
`tools/prepare_rvq_scale_manifests.py` 和两个RVQ测试文件。

```bash
cd /home/deploy/gyh/lyra_md
conda activate lyra
python -m unittest discover -s tests -p 'test*rvq*reference.py' -v
python -m unittest discover -s tests -p test_rvq_scale_qat.py -v

RUN_TAG="rvq-scale-qat-$(date +%Y%m%d-%H%M%S)"
MANIFEST_DIR="results/${RUN_TAG}-manifests"
python tools/prepare_rvq_scale_manifests.py \
  --audio-dir data/librispeech/LibriSpeech/train-clean-100 \
  --output-dir "$MANIFEST_DIR"

SCALE_LOG="logs/${RUN_TAG}.log"
nohup setsid env CUDA_VISIBLE_DEVICES=0 python tools/train_rvq_quant_scales.py \
  --checkpoint results/hardware-qat-w8a16-reference-fix-20260922-222030/best_full_qat.pt \
  --ptq-report results/rvq-int8-ptq-20260923-210227/rvq_ptq_report.json \
  --train-manifest "$MANIFEST_DIR/train.csv" \
  --validation-manifest "$MANIFEST_DIR/validation.csv" \
  --test-manifest "$MANIFEST_DIR/test.csv" \
  --output-dir "results/$RUN_TAG" \
  --steps 2000 --eval-every 250 --lr 0.001 --max-ratio 1.15 \
  --segment-seconds 4 --device cuda \
  </dev/null >"$SCALE_LOG" 2>&1 &
echo "PID=$! LOG=$SCALE_LOG"
tail -F "$SCALE_LOG"
```

默认32训练文件、20验证文件、10测试文件。划分沿用seed42说话人90%/5%/5%，
测试选最长10条。第一次用于小规模实验；最终应扩大验证与测试集。
训练/验证/测试路径和音频内容哈希不允许重叠。PTQ报告只能提供初值，禁止用测试指标选择训练checkpoint。

## 选择与输出

step0也参加验证；最优scale可能仍是初始PTQ，不能假定训练必然改善。
每250步真实整数验证；只有同时满足以下实验阈值才进入best候选：
平均SI-SDR delta>=-0.05 dB，P10>=-0.20 dB，worst>=-0.30 dB；
mean VQout NMSE<=0.04，P90<=0.08；Projection、RVQ搜索/残差/输出均无INT16饱和。
候选按mean VQout NMSE、worst SI-SDR、P10 SI-SDR排序。
这些是初始实验阈值，不代表已通过硬件部署验收。

- `best_scales.pt`：验证集选中scale参数，不是完整SoundStream checkpoint。
- `latest_scales.pt`：末次scale与optimizer状态，当前CLI暂未实现resume选项。
- `validation_XXXXX.json`：逐文件与分位数、margin/weighted flip。
- `final_test_report.json`：选好best后只在测试集评一次；测试失败不导出。
- `export/`：Projection权重/Bias/multiplier/shift与RVQ格式v2二进制、两份manifest。
- `golden_XX.npz`：同一段PCM、整数Encoder输出、Projection输出、各级残差/score/index、最终latent。
- `XX_original.wav`、`XX_fp_codebook.wav`、`XX_scale_qat.wav`：固定中心片段试听。

保真参照是同一冻结QAT模型的FP32 Projection+FP32码本输出，故报告包含前端整数化与Projection量化影响。
每次验证检查整数dot-product与独立平方距离搜索以及indices-only查表一致。
训练STE前向不等于整数乘子近似后的输出；只有真实整数验证参与选checkpoint。
尚未使用真实checkpoint进行端到端运行，也未进行RTL对拍；实际改善需服务器验证。
