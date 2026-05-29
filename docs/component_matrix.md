# 模型变体 × 组件对照矩阵

> 目的:厘清每个变体到底包含哪些组件,**避免把"频率 tokenizer"和"跨频预测"混为一谈**。
> 反映当前代码状态(true SIGReg 为默认、辅助损失已去 StopGrad、新增 `lejepa_crossfreq`)。

## 变体 × 组件

| 变体 (`--model`) | Tokenizer | 跨频预测<br>CrossFreqPredictor | Region 遮挡<br>RegionMasker | 时间 block 遮挡 | 主目标 predictor + StopGrad | 文件 |
|---|---|---|---|---|---|---|
| `jepa`(Laya 对照) | DynamicChannelMixer | ✗ | ✗ | ✓ | **✓**(transformer predictor + StopGrad) | eeg_jepa.py |
| `mae`(重建对照) | DynamicChannelMixer | ✗ | ✗ | ✓ | decoder 重建原始信号 | eeg_mae.py |
| `lejepa`(base) | DynamicChannelMixer | ✗ | ✗ | ✓ | ✗(MLP head,无 StopGrad) | eeg_lejepa.py |
| `lejepa_spectral` | **频率 tokenizer**(SpectralChannelMixer) | ✗ | ✗ | ✓ | ✗ | eeg_lejepa_spectral.py |
| `lejepa_region` | DynamicChannelMixer | ✗ | **✓** | ✓ | ✗ | eeg_lejepa_region.py |
| `lejepa_crossfreq`(新) | SpectralTokenizer | **✓** | ✗ | ✓ | ✗ | eeg_lejepa_crossfreq.py |
| `lejepa_full` | SpectralTokenizer | **✓** | **✓** | ✓ | ✗ | eeg_lejepa_full.py |

所有变体都额外带:**SIGReg/VICReg 正则**(`--reg_type`,默认 sigreg)+ **Query Specialization Loss**。

## ⚠️ 关键澄清:spectral tokenizer ≠ cross-frequency prediction

两件不同的事,只是都带"频率":

- **Spectral tokenizer(频率分带 tokenizer)**:在**分词阶段**把信号拆成 5 个频带的**表示方式**。出现在 `lejepa_spectral` / `lejepa_crossfreq` / `lejepa_full`。
- **Cross-frequency prediction(`CrossFrequencyPredictor`)**:一个**额外的辅助损失**——遮掉某频带,用其余频带在隐空间预测它。**只出现在 `lejepa_full` 和 `lejepa_crossfreq`**。

因此 `lejepa_spectral` 有频率 tokenizer 但**没有**跨频预测损失。`docs/ablation_sigreg.md` / 实验记录里 "`+ Spectral tokenizer: FT −3.3%`" 量的**只是换 tokenizer 的代价,不含跨频预测**。

## 哪个对照隔离哪个组件

| 对照 | 隔离出的效果 |
|---|---|
| `lejepa` vs `lejepa_spectral` | 频率 tokenizer 本身的代价 |
| `lejepa` vs `lejepa_region` | region 遮挡的效果 |
| `lejepa_crossfreq --freq_mask_weight 1` vs `--freq_mask_weight 0` | **跨频预测的纯效果**(同架构、仅开关辅助损失)← 唯一干净隔离 |
| `lejepa_full` vs `lejepa_crossfreq` | region 遮挡在"已有 spectral+跨频"之上的增量 |
| `lejepa` vs `jepa` | predictor + StopGrad 范式的影响(主结论:FT −16.3%) |
| 任意变体 `--reg_type sigreg` vs `vicreg` | 真 SIGReg vs var+cov |

> 注意:`lejepa_spectral` 用的是 `SpectralChannelMixer`,而 `lejepa_crossfreq`/`lejepa_full` 用的是 `SpectralTokenizer`——两者几乎相同但不是同一个类(`d_band` 下限 6 vs 8,且后者会额外吐 band tokens)。**跨脚本横比 spectral 与 crossfreq 时要注意这个 tokenizer 差异**;最干净的跨频对照是 `lejepa_crossfreq` 自身的 weight 1 vs 0。

## 组件设计要点与已知疑点

### Spectral tokenizer(`eeg_lejepa_full.py:59-154`)
可学习 filter bank(sinc 带通×Hamming 初始化)→ 逐带 MLP 编码到 `d_band=max(d_channel//n_bands,8)=8` → 加 band_embed → 拼成 40 维 → 压回 `d_channel=32` → 多 query 空间注意力 → d_model。
- **疑点**:替换了验证过的 DynamicChannelMixer;memory 显示单独上 spectral **FT −3.3%**(唯一有隔离证据的负向组件);分带再压回引入频率瓶颈。

### CrossFrequencyPredictor(`eeg_lejepa_full.py:161-225`)
遮 1~2 个频带,`context = 可见频带均值` → MLP → 与被遮频带算 MSE(已去 StopGrad)。
- **疑点 A(band 无关)**:`predicted` 只算**一个**向量却用于所有被遮频带,`band_mask_tokens` 定义了**未使用**,无 band-id 条件 → 学不到频带特异关系,只能预测"通用缺失带"≈频带均值。
- **疑点 A'(空间被平均)**:`band_tokens` 对通道做了 `mean`,MI 的 α/β 空间局部性(C3/C4)被糊掉。

### RegionMasker(full 版 `eeg_lejepa_full.py:232-298`;region 版 `eeg_lejepa_region.py`)
把 token 的 `d_model` 切成 `n_regions` 段(`d_per_region=256//5=51`),随机遮 1~2 段,用其余段均值经 MLP 预测被遮段(已去 StopGrad)。
- **疑点 D**:所谓"脑区"其实是**隐维切片**,类里的 `ELECTRODE_REGIONS` 电极映射是**死代码**;且"用其余维预测某段维"与 SIGReg 的**去相关目标直接冲突**。但 memory 显示 region 单独上反而 **+0.9%**,故大概率非主凶。

### 损失权重(`compute_loss`)
`(1-λ)·pred + λ·sigreg + freq_w·freq + region_w·region + 0.1·qspec`,默认 `freq_w=region_w=1.0`、`λ=0.05`。
- **疑点 C**:两个辅助损失权重(1.0)压过 SIGReg(0.05)约 **20×**,可能盖过主表示学习。`--freq_mask_weight` 已可调。

## 正则与 StopGrad 现状(当前代码)

| 项 | 现状 |
|---|---|
| 默认正则 | **true SIGReg**(随机投影 + Epps-Pulley 特征函数检验,`regularizers.py`) |
| 备选正则 | `--reg_type vicreg`(var+cov,旧实现,作 ablation) |
| 主目标 StopGrad | 仅 `jepa` / `mae` 有;所有 `lejepa*` 无 |
| 辅助损失 StopGrad | **已全部移除**(跨频/region target 不再 `.detach()`,改为 LeJEPA 一致) |

> 历史结果(旧 var+cov + 辅助 StopGrad)见 `ablation_sigreg.md` 与实验记录;新 SIGReg / 去 StopGrad / crossfreq 的结果待补。
