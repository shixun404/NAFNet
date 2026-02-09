```bash
(py311) ➜  NAFNet git:(main) ✗ python scripts/profile_sidd_breakdown.py \
  --width 16 \
  --enc 1,1,1,1 \
  --dec 1,1,1,1 \
  --middle 12 \
  --device cuda
[Config]
width=16, enc=1,1,1,1, dec=1,1,1,1, middle=12, input=(1, 3, 256, 256), fp16=False, channels_last=False, compile=False
[Model]
macs:   2.47 GMac
params: 6.28 M
[Runtime]
avg_latency_ms: 12.4592

[Module Time Breakdown] (top 20)
module                                                 total_ms       avg_ms      calls
encoders.3.0.norm1                                       14.554      0.14554        100
encoders.0.0.norm1                                       14.524      0.14524        100
middle_blks.0.norm1                                      14.426      0.14426        100
encoders.2.0.norm1                                       14.409      0.14409        100
encoders.1.0.norm1                                       14.320      0.14320        100
decoders.0.0.norm1                                       14.192      0.14192        100
decoders.2.0.norm1                                       14.069      0.14069        100
decoders.3.0.norm1                                       14.064      0.14064        100
encoders.3.0.norm2                                       14.054      0.14054        100
middle_blks.0.norm2                                      14.013      0.14013        100
decoders.1.0.norm1                                       14.012      0.14012        100
encoders.2.0.norm2                                       14.006      0.14006        100
encoders.0.0.norm2                                       13.928      0.13928        100
decoders.2.0.norm2                                       13.920      0.13920        100
middle_blks.1.norm1                                      13.894      0.13894        100
encoders.1.0.norm2                                       13.888      0.13888        100
decoders.1.0.norm2                                       13.879      0.13879        100
middle_blks.1.norm2                                      13.854      0.13854        100
middle_blks.6.norm2                                      13.842      0.13842        100
decoders.3.0.norm2                                       13.822      0.13822        100
/pscratch/sd/s/swu264/conda/py311/lib/python3.11/site-packages/torch/profiler/profiler.py:217: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(

[Operator Time Breakdown] (top 20)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     927.028us        22.13%     927.028us       4.500us           206  
                                aten::cudnn_convolution        11.33%       3.262ms        14.83%       4.271ms      38.824us     903.224us        21.56%     903.224us       8.211us           110  
                                             aten::mean         3.94%       1.134ms         6.17%       1.777ms      17.772us     899.962us        21.48%     899.962us       9.000us           100  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     748.185us        17.86%     748.185us       9.352us            80  
                                              aten::mul         4.60%       1.325ms         7.68%       2.213ms      15.805us     542.812us        12.96%     542.812us       3.877us           140  
                                             aten::add_         3.21%     925.571us         5.30%       1.525ms      14.391us     467.386us        11.15%     467.386us       4.409us           106  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     421.434us        10.06%     421.434us       4.214us           100  
                                              aten::add         3.24%     931.868us         5.38%       1.548ms      12.388us     396.824us         9.47%     396.824us       3.175us           125  
                                              aten::sub         2.08%     600.411us         3.42%     985.385us      12.317us     340.192us         8.12%     340.192us       4.252us            80  
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     247.869us         5.92%     247.869us       7.511us            33  
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     225.694us         5.39%     225.694us       7.783us            29  
                                aten::_conv_depthwise2d         1.21%     349.126us         2.35%     676.441us      33.822us     223.069us         5.32%     223.069us      11.153us            20  
void at::native::(anonymous namespace)::conv_depthwi...         0.00%       0.000us         0.00%       0.000us       0.000us     223.069us         5.32%     223.069us      11.153us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     190.465us         4.55%     190.465us       2.930us            65  
                                              aten::div         1.12%     322.369us         1.83%     525.937us      13.148us     177.119us         4.23%     177.119us       4.428us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     177.119us         4.23%     177.119us       4.428us            40  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     151.777us         3.62%     151.777us       7.589us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     121.378us         2.90%     121.378us       3.034us            40  
                                              aten::pow         3.70%       1.066ms         4.58%       1.319ms      32.976us     113.533us         2.71%     113.533us       2.838us            40  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     113.533us         2.71%     113.533us       2.838us            40  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 28.797ms
Self CUDA time total: 4.190ms
```

## 结论摘要

- 当前基线（`width=16, enc/dec=1, middle=12, 256x256, batch=1, FP32`）：
  - `MACs = 2.47 GMac`
  - `Params = 6.28 M`
  - `avg_latency = 12.46 ms`（约 `80.3 FPS`）
- 主要瓶颈类型：
  - 卷积：`aten::cudnn_convolution`（~21.6% Self CUDA）
  - LayerNorm/逐点运算：`mean/reduce/mul/add/sub/div/pow + elementwise_kernel` 占比很高
- 优化方向：这是“卷积 + 大量小算子混合”结构，通常通过 `FP16 + channels_last + compile` 获得最稳健收益；再做结构侧裁剪（如 `middle`）获得最大幅度收益。

## 结果解读（为什么还有优化空间）

1. 模块统计里 `norm1/norm2` 全部靠前，说明 LayerNorm2d 与其配套逐点算子是热点。
2. 算子统计中 `elementwise/reduce` 和 `mean/div/pow` 占比高，说明 kernel launch 与 memory-bound 开销明显。
3. `cudnn_convolution` 仍是第一梯队热点，表示卷积算子本身也值得通过低精度和内存格式优化。

## 可量化优化空间（基于该 profile 的保守估计）

- 不改模型结构，仅做推理工程优化：
  - 目标：`12.46 ms -> 8.5~10.5 ms`
  - 对应加速：`1.2x ~ 1.45x`
- 轻量结构改动（将 `middle=12 -> 8`，其余不变）：
  - 目标：在工程优化基础上再降 `15%~25%` 延迟
- 极致轻量（`middle=12 -> 1`）：
  - 可能接近 `2x` 级别额外加速（但画质风险最高，需重新评估 SIDD 指标）

## 优化建议（按优先级）

1. 先做低风险工程优化（推荐第一批）
   - 启用 `--fp16 --channels_last`
   - 再启用 `--compile`
   - 预期收益：`15%~45%`（与驱动/显卡/PyTorch版本相关）

2. 减少中间块数量（推荐第二批）
   - 从 `middle=12` 做阶梯实验：`12 -> 10 -> 8 -> 6`
   - 每一步记录：`latency + PSNR/SSIM`
   - 预期收益：基本与 `middle` 近似线性相关（中段是最重路径）

3. 进一步内核融合/部署优化（推荐第三批）
   - 目标：减少 `LayerNorm + pointwise` 的分散 kernel
   - 路线：`torch.compile` 持续优化，或 TensorRT/ONNX 融合
   - 预期收益：在前两批基础上再拿 `5%~20%`

## 建议实验矩阵（直接执行）

```bash
# A. FP32 baseline
python scripts/profile_sidd_breakdown.py --width 16 --enc 1,1,1,1 --dec 1,1,1,1 --middle 12 --device cuda

# B. AMP + channels_last
python scripts/profile_sidd_breakdown.py --width 16 --enc 1,1,1,1 --dec 1,1,1,1 --middle 12 --device cuda --fp16 --channels_last

# C. AMP + channels_last + compile
python scripts/profile_sidd_breakdown.py --width 16 --enc 1,1,1,1 --dec 1,1,1,1 --middle 12 --device cuda --fp16 --channels_last --compile

# D. 结构侧（示例：middle=8）
python scripts/profile_sidd_breakdown.py --width 16 --enc 1,1,1,1 --dec 1,1,1,1 --middle 8 --device cuda --fp16 --channels_last --compile
```

## 目标建议

- 短期目标（不改模型结构）：把延迟从 `12.46 ms` 压到 `<=10 ms`。
- 中期目标（小幅结构改动）：在可接受画质损失下达到 `<=8 ms`。

## 注意事项

- 当前 `Module Time Breakdown` 来自 Python hook，会引入额外同步开销；定位“热点类别”有效，但绝对数值以 `Operator Time Breakdown` 和端到端 `avg_latency_ms` 为准。
- 最终优化决策请以“同一输入、同一设备、重复 3 次以上”的中位数作为比较基准。
