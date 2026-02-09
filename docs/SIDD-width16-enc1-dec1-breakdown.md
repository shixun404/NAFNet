# NAFNet 在 SIDD 配置（`width=16`, `enc_blk_nums=dec_blk_nums=[1,1,1,1]`）下的输入输出、数学过程与算子对应

本文基于仓库实现 `basicsr/models/archs/NAFNet_arch.py` 与 `basicsr/models/archs/arch_util.py`，描述该配置的一次前向传播。

## 1. 配置与问题定义

- 任务：RGB 图像去噪（SIDD）。
- 模型：`NAFNet`。
- 关键配置：
  - `img_channel = 3`
  - `width = 16`
  - `enc_blk_nums = [1,1,1,1]`
  - `dec_blk_nums = [1,1,1,1]`
  - `middle_blk_num = 12`（沿用 SIDD 默认配置）
- 典型输入：`x_in ∈ R^{B×3×H×W}`，例如 `B=1, H=W=256`。
- 输出：`x_out ∈ R^{B×3×H×W}`（与输入同尺寸，残差学习）。

## 2. 整体输入输出与形状流

### 2.1 padding 规则

模型内部 `padder_size = 2^4 = 16`（因为有 4 次下采样）。  
若 `H`/`W` 不是 16 的倍数，会右侧/底部补零：

- `H' = ceil(H/16)*16`
- `W' = ceil(W/16)*16`

`256x256` 时不补零（`H'=H, W'=W`）。

### 2.2 各 stage 形状（以 `B=1, H=W=256` 为例）

1. `intro`：`Conv3x3, 3->16`  
   输出：`[1,16,256,256]`
2. 编码器 + 下采样：
   - `enc0`（1 个 NAFBlock, `C=16`）-> `[1,16,256,256]`
   - `down0`（`Conv2x2 s=2, 16->32`）-> `[1,32,128,128]`
   - `enc1`（`C=32`）-> `[1,32,128,128]`
   - `down1`（`32->64`）-> `[1,64,64,64]`
   - `enc2`（`C=64`）-> `[1,64,64,64]`
   - `down2`（`64->128`）-> `[1,128,32,32]`
   - `enc3`（`C=128`）-> `[1,128,32,32]`
   - `down3`（`128->256`）-> `[1,256,16,16]`
3. `middle`：12 个 NAFBlock（`C=256`）  
   输出：`[1,256,16,16]`
4. 解码器 + 上采样：
   - `up0`: `Conv1x1 (256->512)` + `PixelShuffle(2)` -> `[1,128,32,32]`
   - 加 skip（来自 `enc3`）-> `[1,128,32,32]`
   - `dec0`（1 个 NAFBlock, `C=128`）
   - `up1` -> `[1,64,64,64]` + skip(`enc2`) + `dec1`(`C=64`)
   - `up2` -> `[1,32,128,128]` + skip(`enc1`) + `dec2`(`C=32`)
   - `up3` -> `[1,16,256,256]` + skip(`enc0`) + `dec3`(`C=16`)
5. `ending`：`Conv3x3, 16->3`，再与 padded 输入做残差相加。  
6. 裁剪回原尺寸：`x[:, :, :H, :W]`。

## 3. NAFBlock 的完整数学运算（单块）

下面以输入 `u ∈ R^{B×C×H×W}` 表示一个 `NAFBlock(C)` 的前向。该实现中 `DW_Expand=2`, `FFN_Expand=2`。

### 3.1 LayerNorm2d（按通道维归一化）

实现来自 `LayerNormFunction`，对每个像素位置 `(h,w)` 在通道维求均值方差：

1. `μ = mean_c(u)`  
2. `σ^2 = mean_c((u-μ)^2)`  
3. `u_hat = (u-μ)/sqrt(σ^2+eps)`  
4. `v = w_ln * u_hat + b_ln`

其中 `w_ln, b_ln` 为可学习参数，形状均为 `[C]`（广播到 `[1,C,1,1]`）。

### 3.2 空间混合 + 简化通道注意力（SCA）分支

1. `a = Conv1x1(v)`，`C -> 2C`  
2. `b = DWConv3x3(a)`，`2C -> 2C`（depthwise）
3. `b1, b2 = chunk(b, 2, dim=channel)`  
4. `g = b1 * b2`（SimpleGate）
5. `s = Conv1x1(AvgPool(g))`，其中 `AvgPool` 为全局平均池化到 `[B,C,1,1]`
6. `t = g * s`（通道缩放，广播乘法）
7. `p = Conv1x1(t)`，`C -> C`
8. 第一残差：`y = u + beta * p`，`beta ∈ R^{1×C×1×1}` 可学习

### 3.3 FFN 分支

1. `q = LN(y)`（同上）
2. `r = Conv1x1(q)`，`C -> 2C`
3. `r1, r2 = chunk(r, 2, dim=channel)`
4. `h = r1 * r2`（SimpleGate）
5. `z = Conv1x1(h)`，`C -> C`
6. 第二残差：`out = y + gamma * z`，`gamma ∈ R^{1×C×1×1}` 可学习

这就是一个 NAFBlock 的全部数学流程。

## 4. 全网络前向的数学表达（高层）

记：

- `E_i`：第 `i` 个 encoder block（本配置每个 `E_i` 都是 1 个 NAFBlock）
- `D_i`：第 `i` 个 decoder block（同上）
- `Down_i`：步长 2 的 `Conv2x2`
- `Up_i`：`Conv1x1 + PixelShuffle(2)`

前向流程：

1. `x0 = Intro(Pad(x_in))`
2. `s0 = E0(x0)`, `x1 = Down0(s0)`
3. `s1 = E1(x1)`, `x2 = Down1(s1)`
4. `s2 = E2(x2)`, `x3 = Down2(s2)`
5. `s3 = E3(x3)`, `x4 = Down3(s3)`
6. `m = Middle(x4)`，其中 `Middle = NAFBlock ∘ ... ∘ NAFBlock`（12 次）
7. `y3 = D0(Up0(m) + s3)`
8. `y2 = D1(Up1(y3) + s2)`
9. `y1 = D2(Up2(y2) + s1)`
10. `y0 = D3(Up3(y1) + s0)`
11. `x_pad_out = Ending(y0) + Pad(x_in)`
12. `x_out = Crop(x_pad_out, H, W)`

## 5. 数学步骤到 PyTorch 算子的对应

下表是核心映射（对应你 profile 中常见 `aten::*`）：

- `Conv1x1 / Conv3x3 / Conv2x2 / DWConv3x3`：`aten::cudnn_convolution`（或相关卷积 kernel）
- `PixelShuffle(2)`：`aten::pixel_shuffle`
- `mean`（LN 与全局池化）：`aten::mean` + `reduce_kernel`
- 减法 `(x-μ)`：`aten::sub`
- 平方 `pow(2)`：`aten::pow`
- 除法 `/sqrt(...)`：`aten::div` + `aten::sqrt`
- 加法（残差、偏置等）：`aten::add` / `aten::add_`
- 乘法（SimpleGate、通道缩放、`beta/gamma`）：`aten::mul`
- 切分 `chunk`：`aten::chunk`（及底层 view/slice）
- padding：`aten::pad`（`F.pad`）
- 裁剪：`aten::slice`

## 6. 该配置下的 block 数量与算子结构规模

- Encoder block 数：`1+1+1+1 = 4`
- Middle block 数：`12`
- Decoder block 数：`1+1+1+1 = 4`
- 全网 NAFBlock 总数：`20`

每个 NAFBlock 固定包含：

- LayerNorm2d：2 次
- 卷积：5 次（`conv1, conv2(dw), conv3, conv4, conv5`）
- SimpleGate 乘法：2 次
- 残差门控乘法：2 次（`beta`, `gamma`）
- 以及若干 `add/sub/mean/pow/div/sqrt`

因此总算子热点通常会集中在：

- 大量卷积（尤其 1x1 与 depthwise 3x3）
- LayerNorm2d 相关的 `mean/sub/pow/div/sqrt`
- 逐点算子 `mul/add`

## 7. 输入输出语义总结

- 输入：噪声图像 `x_in`
- 输出：去噪后图像 `x_out`
- 网络学习的是残差修正：`x_out ≈ x_in + f_theta(x_in)`（在 padded 空间中计算，最后裁剪回原图大小）

## 8. 从 input 到 output 的 Torch Pseudocode（逐步带维度注释）

```python
# ------------------------------------------------------------
# Config (this document's target setting)
# width = 16
# enc_blk_nums = [1, 1, 1, 1]
# dec_blk_nums = [1, 1, 1, 1]
# middle_blk_num = 12
# ------------------------------------------------------------

def layernorm2d(u, w_ln, b_ln, eps=1e-6):
    # u: [B, C, H, W]
    mu = u.mean(dim=1, keepdim=True)                    # [B, 1, H, W]
    var = (u - mu).pow(2).mean(dim=1, keepdim=True)     # [B, 1, H, W]
    u_hat = (u - mu) / (var + eps).sqrt()               # [B, C, H, W]
    y = w_ln.view(1, C, 1, 1) * u_hat + b_ln.view(1, C, 1, 1)  # [B, C, H, W]
    return y                                             # [B, C, H, W]

def naf_block(u, params):
    # u: [B, C, H, W]
    v = layernorm2d(u, params.w1, params.b1)            # [B, C, H, W]
    a = conv1x1(v, out_channels=2*C)                    # [B, 2C, H, W]
    b = dwconv3x3(a, channels=2*C)                      # [B, 2C, H, W]
    b1, b2 = b.chunk(2, dim=1)                          # [B, C, H, W], [B, C, H, W]
    g = b1 * b2                                         # [B, C, H, W]

    s = adaptive_avg_pool2d(g, output_size=1)           # [B, C, 1, 1]
    s = conv1x1(s, out_channels=C)                      # [B, C, 1, 1]
    t = g * s                                           # [B, C, H, W] (broadcast)
    p = conv1x1(t, out_channels=C)                      # [B, C, H, W]
    y = u + params.beta * p                             # [B, C, H, W], beta:[1,C,1,1]

    q = layernorm2d(y, params.w2, params.b2)            # [B, C, H, W]
    r = conv1x1(q, out_channels=2*C)                    # [B, 2C, H, W]
    r1, r2 = r.chunk(2, dim=1)                          # [B, C, H, W], [B, C, H, W]
    h = r1 * r2                                         # [B, C, H, W]
    z = conv1x1(h, out_channels=C)                      # [B, C, H, W]
    out = y + params.gamma * z                          # [B, C, H, W], gamma:[1,C,1,1]
    return out                                          # [B, C, H, W]

def forward(x_in):
    # x_in: [B, 3, H, W]
    B, _, H, W = x_in.shape

    # pad to multiple of 16 (because 4 downsamples => padder_size=16)
    Hp = ((H + 15) // 16) * 16
    Wp = ((W + 15) // 16) * 16
    x_pad = pad_to_size(x_in, Hp, Wp)                   # [B, 3, Hp, Wp]

    # intro
    x = conv3x3(x_pad, out_channels=16)                 # [B, 16, Hp, Wp]

    # ---------------- Encoder path ----------------
    # stage 0 (C=16)
    s0 = naf_block(x, enc0_blk0)                        # [B, 16, Hp, Wp]
    x = conv2x2_stride2(s0, out_channels=32)            # [B, 32, Hp/2, Wp/2]

    # stage 1 (C=32)
    s1 = naf_block(x, enc1_blk0)                        # [B, 32, Hp/2, Wp/2]
    x = conv2x2_stride2(s1, out_channels=64)            # [B, 64, Hp/4, Wp/4]

    # stage 2 (C=64)
    s2 = naf_block(x, enc2_blk0)                        # [B, 64, Hp/4, Wp/4]
    x = conv2x2_stride2(s2, out_channels=128)           # [B, 128, Hp/8, Wp/8]

    # stage 3 (C=128)
    s3 = naf_block(x, enc3_blk0)                        # [B, 128, Hp/8, Wp/8]
    x = conv2x2_stride2(s3, out_channels=256)           # [B, 256, Hp/16, Wp/16]

    # ---------------- Middle path ----------------
    # 12 x NAFBlock at C=256
    x = naf_block(x, mid_blk0)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk1)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk2)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk3)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk4)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk5)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk6)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk7)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk8)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk9)                          # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk10)                         # [B, 256, Hp/16, Wp/16]
    x = naf_block(x, mid_blk11)                         # [B, 256, Hp/16, Wp/16]

    # ---------------- Decoder path ----------------
    # up0: C 256 -> 128
    x = conv1x1(x, out_channels=512)                    # [B, 512, Hp/16, Wp/16]
    x = pixel_shuffle(x, upscale_factor=2)              # [B, 128, Hp/8, Wp/8]
    x = x + s3                                          # [B, 128, Hp/8, Wp/8]
    x = naf_block(x, dec0_blk0)                         # [B, 128, Hp/8, Wp/8]

    # up1: C 128 -> 64
    x = conv1x1(x, out_channels=256)                    # [B, 256, Hp/8, Wp/8]
    x = pixel_shuffle(x, upscale_factor=2)              # [B, 64, Hp/4, Wp/4]
    x = x + s2                                          # [B, 64, Hp/4, Wp/4]
    x = naf_block(x, dec1_blk0)                         # [B, 64, Hp/4, Wp/4]

    # up2: C 64 -> 32
    x = conv1x1(x, out_channels=128)                    # [B, 128, Hp/4, Wp/4]
    x = pixel_shuffle(x, upscale_factor=2)              # [B, 32, Hp/2, Wp/2]
    x = x + s1                                          # [B, 32, Hp/2, Wp/2]
    x = naf_block(x, dec2_blk0)                         # [B, 32, Hp/2, Wp/2]

    # up3: C 32 -> 16
    x = conv1x1(x, out_channels=64)                     # [B, 64, Hp/2, Wp/2]
    x = pixel_shuffle(x, upscale_factor=2)              # [B, 16, Hp, Wp]
    x = x + s0                                          # [B, 16, Hp, Wp]
    x = naf_block(x, dec3_blk0)                         # [B, 16, Hp, Wp]

    # ending + global residual
    x = conv3x3(x, out_channels=3)                      # [B, 3, Hp, Wp]
    x = x + x_pad                                       # [B, 3, Hp, Wp]

    # crop back to original size
    x_out = x[:, :, :H, :W]                             # [B, 3, H, W]
    return x_out                                        # [B, 3, H, W]
```

---

如果你后续要做算子级优化，这份文档可直接作为性能分析基线，配合 `scripts/profile_sidd_breakdown.py` 的 `Operator Time Breakdown` 一起使用。
