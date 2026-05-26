# EEG Foundation Model 调研：谁用了频率分解和空间 masking？

## 按模型整理

### CBraMod (ICLR 2025)
- **Criss-Cross Attention**: 空间注意力和时间注意力分开并行处理
- **Tokenization**: patch-based，每个 patch = (几个通道 × 一段时间)
- **空间处理**: 有！空间注意力单独建模通道间关系
- **频率分解**: 无，直接处理原始信号
- **Masking**: masked patch reconstruction
- **Positional encoding**: asymmetric conditional positional encoding
- **⚠ 和我们的关系**: 它的"空间注意力"类似我们的 channel mixer，但它没有脑区 masking

### REVE (NeurIPS 2025)
- **4D Positional Encoding**: 利用电极的真实 3D 坐标 + 时间索引
- **空间处理**: 有！用电极坐标编码空间信息
- **Masking**: **block masking in both spatial AND temporal dimensions**
- **频率分解**: 无
- **⚠ 和我们的关系**: REVE 已经做了空间+时间联合 masking！但它用的是电极坐标，不是脑区分组

### LUNA (NeurIPS 2025)
- **Latent query architecture**: 用 cross-attention 压缩多通道到 latent space
- **空间处理**: 有，通过 latent queries 隐式处理
- **频率分解**: 无
- **Masking**: masked patch reconstruction

### NeurIPT (NeurIPS 2025)
- **PMoE (Progressive Mixture-of-Experts)**: 不同层用不同 expert 处理不同时间特征
- **空间处理**: **Intra-Inter Lobe Pooling (IILP)** — 按脑叶分组池化！
- **频率分解**: 无直接分解，但 PMoE 可能隐式处理不同频段
- **3D 电极坐标**: 用于跨设备迁移
- **⚠ 和我们的关系**: IILP 类似我们的脑区分组，但它用在 fine-tune 阶段不是预训练

### FoME
- **双域输入**: **时间序列 + 功率谱密度（PSD）同时输入**
- **空间处理**: 有 spatial encoder（通道维度）
- **频率分解**: **有！PSD 作为额外输入**
- **架构**: temporal encoder → reorganize by channel → spatial encoder
- **⚠ 和我们的关系**: FoME 已经用了频谱信息！但它是把 PSD 作为单独输入，不是在 tokenizer 里分解

### BrainRVQ
- **Vector Quantization**: 类似 LaBraM 的 VQ 方法
- **双域 VQ**: **时间和频率域分别做 vector quantization**
- **⚠ 和我们的关系**: 频率域 VQ 和我们的频率感知 tokenizer 理念类似

### EEG-X (2025)
- **Location-based channel embedding**: 用电极位置编码空间信息
- **Noise-aware masking**: 同时在 raw 和 latent space 做 masking
- **频率分解**: 无

### ST-EEGFormer (NeurIPS 2025 Challenge Winner)
- **Spatial + Temporal tokenization**: 沿空间和时间维度分割
- **Masking**: **75% 随机 masking 所有 token（空间+时间）**
- **频率分解**: 无
- **⚠ 和我们的关系**: 它的 token 本身就包含空间维度，所以 random masking 自然包含了空间 masking

### Laya (2026)
- **频率分解**: 无
- **空间处理**: Dynamic Channel Mixer (16 queries)
- **Masking**: 只有时间 block masking
- **无脑区 masking**

---

## 关键发现总结

### 频率分解：少数模型用了，但方式不同

| 模型 | 频率处理方式 |
|---|---|
| FoME | PSD 作为额外输入（双流） |
| BrainRVQ | 频率域 VQ（离散化） |
| LaBraM | 预测频谱（VQ-NSP） |
| **其他所有模型** | **无频率分解** |
| **我们提议** | **LearnableFilterBank 在 tokenizer 内分解** |

→ **频率感知 tokenizer 有差异化空间**。现有方法要么不用频率，要么把频率作为单独分支。我们在 tokenizer 内部做频段分解，更端到端。

### 空间/脑区 Masking：已有模型做了，但方式不同

| 模型 | 空间 masking 方式 |
|---|---|
| REVE | block masking in spatial + temporal（电极级别） |
| ST-EEGFormer | 随机 masking spatial-temporal tokens |
| **其他所有模型** | **只有时间 masking** |
| **我们提议** | **脑区级别 masking（按解剖分组）** |

→ **脑区 masking 有差异化**。REVE 在电极级别做空间 masking（随机选几个电极 mask），我们在脑区级别做（mask 整个 frontal 或 parietal 区），有更强的神经科学先验。

---

## 我们的差异化定位

```
已有方法的空白：
1. 没有模型同时做 脑区masking + 频率分解 + LeJEPA
2. REVE 有空间 masking 但没有频率分解
3. FoME 有频率信息但没有空间 masking
4. Laya 两个都没有

我们的组合：
  LeJEPA (理论保证)
  + 频率感知 Tokenizer (EEG 频段先验)
  + 脑区 Masking (EEG 空间先验)
  = 唯一一个同时利用 EEG 时间+频率+空间三个维度的 LeJEPA
```
