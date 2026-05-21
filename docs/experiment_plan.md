# EEG-JEPA 实验计划

## Part 1: 各模型预训练和测试数据集总览

### 预训练数据

| 模型 | 发表 | 预训练方法 | 预训练数据 | 规模 |
|---|---|---|---|---|
| **LaBraM** | ICLR 2024 | VQ-VAE + masked discrete token prediction | TUEG子集(TUSZ/TUEP/TUAR/TUSL) + PhysioNet + SEED + 自采等18个数据集 | **2,535h, ~500+人** |
| **LUNA** | NeurIPS 2025 | Masked patch reconstruction | TUEG + Siena | **21,000+h** |
| **CBraMod** | ICLR 2025 | Masked patch reconstruction (criss-cross attention) | 大规模EEG语料 | **9,000+h, 1.1M样本** |
| **REVE** | NeurIPS 2025 | Block masking reconstruction (4D positional encoding) | 92个数据集, 25,000 subjects | **60,000+h, 25K人** |
| **Laya** | 2026 | LeJEPA (latent prediction + SIGReg) | TUH + HBN + NMT + EEGDash + MOABB | **29,109h, 20,940人** |
| **Ours** | - | LeJEPA-style (latent prediction + SIGReg) | PhysioNet + Cho2017 + Lee2019 + HBN(200人) | **~200h, ~415人** |

### 下游评测数据集（EEG-Bench 标准）

**BCI 任务（Motor Imagery）：**

| 任务 | 数据集 | 类别数 |
|---|---|---|
| LH vs RH MI | BCI-IV-2a, BCI-IV-2b, PhysioNet, Cho2017, Liu2022, Schirrmeister2017, Weibo2014, Zhou2016, Kaya2018 | 2 |
| RH vs Feet MI | Weibo2014, PhysioNet, BCI-IV-2a, Barachant2012, Faller2012, Scherer2015, Zhou2016, Kaya2018 | 2 |
| 4-Class MI | BCI-IV-2a, Kaya2018 | 4 |
| 5-Finger MI | Kaya2018 | 5 |

**Clinical 任务：**

| 任务 | 数据集 |
|---|---|
| Abnormal | TUAB |
| Epilepsy | TUEP |
| Seizure | CHB-MIT / TUSZ |
| Artifact (Binary) | TUAR |
| Artifact (Multi) | TUAR |
| Sleep Stages | Sleep-EDF |
| mTBI | Singh2018 |
| Parkinson's | Cavanagh2018 |
| Schizophrenia | Cavanagh2019 |
| OCD | Gruendler2009 |

---

## Part 2: 我们要用的数据集

### 预训练数据（按阶段扩展）

| 阶段 | 数据集 | 规模 | 状态 |
|---|---|---|---|
| Phase 0 | PhysioNet MI | 109人, ~10h | ✅ 完成 |
| Phase 1 | + Cho2017, Lee2019_MI | 215人, ~40h | ✅ 完成 |
| Phase 2 | + HBN (200人) | 415人, ~200h | ✅ 已下载 |
| Phase 3 | + TUH (申请中) | 1000+人, 1000+h | ⏳ 等审批 |
| Phase 4 | + 更多 MOABB | 500+人 | 可选 |

### 下游评测数据（EEG-Bench）

| 数据集 | 状态 | 用于任务 |
|---|---|---|
| BCI-IV-2a/2b (MOABB) | ✅ 已有 | lr, rf, 4class |
| PhysioNet MI (MOABB) | ✅ 已有 | lr, rf |
| Cho2017 (MOABB) | ✅ 已有 | lr |
| Weibo2014 (MOABB) | ✅ 已有 | lr, rf |
| Liu2022 | ❌ 下载损坏 | lr |
| Schirrmeister2017 | ❌ 服务器网络超时 | lr, rf |
| Kaya2018 (HaLT) | ❌ .mat文件截断 | lr, rf, 4class, 5finger |
| Zhou2016 | ❌ CNT格式错误 | lr, rf |
| Barachant2012/Faller2012/Scherer2015 | ❌ 不在MOABB | rf |
| TUAB | ❌ 需TUH申请 | abnormal |
| Singh2018 | ✅ 已有 (94G) | mtbi |
| Cavanagh2018a | ✅ 已有 (12G) | parkinsons |
| Brown2020 | ✅ 已有 (18G) | schizophrenia |

---

## Part 3: 实验计划

### 实验 1: 预训练 + EEG-Bench BCI 评测 [核心]

**目标**：在 EEG-Bench BCI 任务上对比我们和 baseline。

**方法**：
- 用 `run_eegbench.py --run_baselines` 在完全相同的数据子集上跑所有模型
- 模型：Ours (frozen probe), Random, LaBraM (fine-tune), CSP+LDA
- 任务：lr, rf, 4class（用可用的数据子集）
- 5次重复取 mean±std

**预期输出表格（对标 Laya Table 1）**：

| Task | Chance | CSP-LDA | LaBraM | Ours (frozen) | Laya-S* |
|---|---|---|---|---|---|
| LH vs RH | 0.500 | x.xxx | x.xxx | x.xxx | 0.510 |
| RH vs Feet | 0.500 | x.xxx | x.xxx | x.xxx | 0.586 |
| 4-Class MI | 0.250 | x.xxx | x.xxx | x.xxx | 0.273 |
| Mean | | x.xxx | x.xxx | x.xxx | 0.395 |

*Laya 数字引用自论文（完整数据集），标注†表示不完全可比

**GPU**: 1张卡，~2小时

### 实验 2: 数据规模消融 [核心]

**目标**：证明 scaling 对 EEG-JEPA 的效果。

**方法**：
- 在不同数据规模下预训练，然后统一评测
- 规模：109人 → 215人 → 415人 (→ 1000+人 if TUH available)
- 评测：EEG-Bench BCI frozen probe

**预期输出**：Scaling curve 图（x=数据量，y=accuracy）

**GPU**: 2张卡 × 4h/规模 × 3规模 = ~24 GPU-hours

### 实验 3: 预训练目标消融 [核心]

**目标**：证明 JEPA latent prediction > reconstruction > random。

**方法**：
- 同一数据（Phase 2, 415人）、同一架构
- 变体：
  - Random init (不预训练)
  - MAE (reconstruct raw EEG patches)
  - JEPA w/o SIGReg (去掉正则化)
  - JEPA w/ SIGReg (完整方法)
- 评测：EEG-Bench BCI frozen probe

**预期输出**：

| Method | LH vs RH | RH vs Feet | 4-Class | Mean |
|---|---|---|---|---|
| Random | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| MAE | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| JEPA w/o SIGReg | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| **JEPA w/ SIGReg** | 0.xxx | 0.xxx | 0.xxx | 0.xxx |

**GPU**: 2张卡 × 4h × 3变体 = ~24 GPU-hours

**需要额外写的代码**：MAE baseline（重建原始EEG patch，而不是预测latent）

### 实验 4: SIGReg/VICReg 必要性消融 [核心]

**目标**：证明 EEG 上正则化是必需的（这是我们的核心发现之一）。

**方法**：
- JEPA + SIGReg (λ=0.05)
- JEPA + VICReg (var + cov)
- JEPA 无正则化
- 对比维度坍缩指标 + 下游 accuracy

**预期输出**：

| Regularization | Effective Dims (90%) | Inter-sample Cos | LH vs RH |
|---|---|---|---|
| None | 4 | 0.99 | ~chance |
| VICReg | 136 | 0.23 | 0.xxx |
| SIGReg | xxx | 0.xxx | 0.xxx |

**GPU**: 已有大部分数据（之前的失败实验）

### 实验 5: 表示可视化 [必须]

**目标**：PCA/UMAP 展示不同类别的表示分布。

**方法**：
- 提取 frozen encoder 表示
- PCA/UMAP 降维到 2D
- 按 MI 类别着色
- 对比 random vs pretrained

**需要额外写的代码**：可视化脚本

**GPU**: 不需要训练，1张卡 10min

### 实验 6: 噪声鲁棒性 [建议]

**目标**：证明预训练提升了对噪声的鲁棒性（对标 Laya Fig.3）。

**方法**：
- 在测试数据上人工加噪声（Gaussian, 1/f, EMG, channel dropout）
- SNR 从 clean 到 10dB
- 对比 pretrained vs random 的 accuracy 下降曲线

**需要额外写的代码**：噪声注入 + 评测脚本

**GPU**: 不需要训练，1张卡 30min

### 实验 7: Clinical 任务评测 [加分]

**目标**：展示跨任务泛化能力。

**方法**：
- 用已有的 clinical 数据集（Singh2018/mTBI, Cavanagh2018/Parkinson's 等）
- Frozen probe 评测
- 如果 TUH 审批通过，加 TUAB(abnormal) 和 TUEP(epilepsy)

**GPU**: 1张卡，~1小时

---

## Part 4: 优先级排序

| 优先级 | 实验 | 耗时 | 需要新代码 |
|---|---|---|---|
| **P0** | 实验1: EEG-Bench BCI + baselines | 2h | ✅ 已写好 (run_eegbench.py) |
| **P0** | 实验2: 数据规模消融 | 24h | 部分已有（不同 phase 的 checkpoint） |
| **P0** | 实验3: 预训练目标消融 | 24h | 需写 MAE baseline |
| **P0** | 实验5: 表示可视化 | 10min | 需写可视化脚本 |
| **P1** | 实验4: SIGReg 消融 | 已有数据 | 只需整理 |
| **P1** | 实验6: 噪声鲁棒性 | 30min | 需写噪声脚本 |
| **P2** | 实验7: Clinical 任务 | 1h | 需 TUH 数据 |

---

## Part 5: 时间线

| 时间 | 任务 |
|---|---|
| **本周** | 实验1（EEG-Bench BCI 完整对比）+ 实验5（可视化） |
| **下周** | 实验2（scaling 消融）+ 实验3（MAE baseline + 目标消融） |
| **第3周** | 实验4（SIGReg 消融整理）+ 实验6（鲁棒性） |
| **第4周** | 实验7（Clinical，如果 TUH 到了）+ 论文写作 |
