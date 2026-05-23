# EEG-JEPA vs Laya 架构对比（最新版）

## 整体架构

| 组件 | Laya | Ours (EEG-JEPA v4) | 状态 |
|---|---|---|---|
| **Training paradigm** | LeJEPA (single encoder + StopGrad) | LeJEPA (single encoder + StopGrad) | ✅ 相同 |
| **EMA target** | 无 | 无 | ✅ 相同 |
| **Anti-collapse** | SIGReg (λ=0.05) | SIGReg (var+cov, λ=0.05) | ✅ 相同 |
| **Masking** | Block masking 60%, 500ms-1s blocks | Block masking 60%, ~500ms blocks | ✅ 相同 |
| **Loss** | L2 (MSE) | L2 (MSE) | ✅ 相同 |

## Tokenizer

| | Laya | Ours | 差异 |
|---|---|---|---|
| **Per-channel encoding** | Depthwise 1D Conv (kernel=25, stride=25) | MLP (Linear→GELU→Linear→LN) | 不同实现，同理念 |
| **Patch size** | 25 samples (100ms @ 250Hz) | 26 samples (100ms @ 256Hz) | ≈相同 |
| **Channel embedding** | Fixed Fourier features (3D electrode coordinates) | Learnable embedding per electrode | **我们更简单** |
| **Channel mixing** | Dynamic Channel Mixer (16 queries + cross-attention) | Dynamic Channel Mixer (16 queries + cross-attention) | ✅ 相同 |
| **Query Specialization Loss** | 有 (penalize WW^T off-diagonal) | 有 (penalize WW^T off-diagonal) | ✅ 相同 |
| **Channel mixer dim** | 32 | 32 | ✅ 相同 |
| **N_queries** | 16 | 16 | ✅ 相同 |

## Encoder

| | Laya | Ours | 差异 |
|---|---|---|---|
| **Architecture** | Transformer | Transformer | ✅ 相同 |
| **Layers** | 12 | 6 | **我们更小** |
| **d_model** | 384 | 256 | **我们更小** |
| **Heads** | 6 | 8 | 不同 |
| **MLP ratio** | 4x | 4x | ✅ 相同 |
| **Position encoding** | RoPE (旋转位置编码) | Absolute (learnable) | **不同** |
| **参数量** | ~30M (估计) | ~5M (估计) | **我们小 6x** |

## Predictor

| | Laya | Ours | 差异 |
|---|---|---|---|
| **Layers** | 4 | 3 | 接近 |
| **Dim** | 128 | 128 | ✅ 相同 |
| **Heads** | 4 | 4 | ✅ 相同 |
| **Position** | RoPE | Absolute (learnable) | **不同** |

## Training

| | Laya | Ours | 差异 |
|---|---|---|---|
| **Optimizer** | cosine schedule + warmup | cosine schedule + warmup | ✅ 相同 |
| **Learning rate** | 1e-4 | 3e-4 | 不同 |
| **Batch size** | 256/GPU | 8/GPU | **差 32x** |
| **Total steps** | 10K-20K | ~5600 (50 epochs) | 不同 |
| **Weight decay** | 0.05 | 0.05 | ✅ 相同 |
| **Precision** | bf16-mixed | fp32 | **不同** |
| **Input crop** | Random 16s crop from 120s | Fixed 4s trial | **不同** |
| **Sampling rate** | 250 Hz | 256 Hz | ≈相同 |

## Pretraining Data

| | Laya-S | Laya-B | Ours |
|---|---|---|---|
| **Hours** | 3,000 | 29,109 | ~10 (PhysioNet only) |
| **Subjects** | 20,940 | 20,940 | 109 |
| **Datasets** | TUH+HBN+NMT+EEGDash+MOABB | same | PhysioNet |
| **Channel topologies** | 17 different | 17 different | 1 (64ch) |

## 剩余差异总结（可能影响性能的）

### 已对齐 ✅
- LeJEPA paradigm (no EMA)
- StopGrad on targets
- SIGReg regularization
- Block masking 60%
- Dynamic Channel Mixer (16 queries)
- Query Specialization Loss
- L2 prediction loss

### 未对齐 ⚠️
1. **RoPE vs Absolute position** — Laya 用旋转位置编码，我们用绝对位置
2. **模型规模** — 12层/384d vs 6层/256d（小 6 倍）
3. **Batch size** — 256 vs 8（差 32 倍，影响 BatchNorm 和 SIGReg 的统计质量）
4. **数据规模** — 3000h vs 10h（差 300 倍）
5. **Input crop** — 16s random crop vs 4s fixed trial
6. **Electrode encoding** — Fourier features (3D coordinates) vs learnable embedding
7. **Training precision** — bf16 vs fp32

### 最可能影响性能的前 3 个因素
1. **数据规模**（300x 差距）
2. **Batch size**（32x 差距，直接影响 SIGReg 的 variance/covariance 估计质量）
3. **模型规模**（6x 差距）
