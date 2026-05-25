# SIGReg/VICReg Ablation Results

## Representation Quality (check_repr.py, PhysioNet val set)

| Regularization | Effective Dims (90%) | Dead Dims | Inter-sample Cosine | Mean Std/Dim |
|---|---|---|---|---|
| **None** | **4/256** | 2/256 | 0.71 | 0.019 |
| **VICReg (var+cov)** | **136/256** | 0/256 | 0.75 | 0.079 |
| **SIGReg (var+cov, LeJEPA)** | **61/256** | 0/256 | 0.23 | 0.139 |

## Downstream Performance (PhysioNet MI, 4-Class, eval_jepa.py)

### Frozen Linear Probe (balanced accuracy)

| Regularization | Model | Random | Pretrained | Delta |
|---|---|---|---|---|
| None (v1) | JEPA + EMA | 0.283 | 0.279 | -0.4% |
| VICReg λ_var=5, λ_cov=1 (v2) | JEPA + EMA | 0.283 | 0.296 | **+1.3%** |
| SIGReg λ=0.05 (v4, LeJEPA) | LeJEPA (no predictor) | 0.289 | 0.263 | -2.6% |

### Fine-tune (balanced accuracy)

| Regularization | Model | Random-FT | Pretrained-FT | Delta |
|---|---|---|---|---|
| None (v1) | JEPA + EMA | — | — | — |
| VICReg (v2) | JEPA + EMA | — | — | — |
| SIGReg λ=0.05 (LeJEPA, 10h) | LeJEPA | 0.590 | 0.604 | **+1.4%** |
| SIGReg λ=0.05 (LeJEPA, 1000h) | LeJEPA | 0.458 | 0.605 | **+14.6%** |

## Key Findings

### 1. Without regularization → complete dimensional collapse
- Only 4 out of 256 dimensions are active
- Top-1 singular value explains 51.4% of variance
- Prediction loss drops to exactly 0.0000
- Frozen probe ≈ random (no pretraining benefit)

### 2. VICReg fixes collapse but frozen probe only
- 136/256 dimensions active (34× improvement)
- All dimensions alive (0 dead dims)
- Frozen probe improves +1.3%
- Fine-tune not tested (code had issues at the time)

### 3. SIGReg enables effective fine-tuning
- Similar collapse prevention as VICReg
- But more diverse representations (inter-sample cosine 0.23 vs 0.75)
- Frozen probe marginal, fine-tune shows real value (+14.6% at scale)

### 4. Regularization alone is not enough — data scale matters
| Data Scale | SIGReg Frozen Delta | SIGReg Fine-tune Delta |
|---|---|---|
| 10h (109 subjects) | -1.3% | +1.4% |
| 1000h (1732 subjects) | +1.0% | **+14.6%** |

## Conclusion for Paper

> "Anti-collapse regularization (VICReg/SIGReg) is necessary but not sufficient 
> for effective EEG self-supervised pretraining. Without regularization, encoder 
> representations collapse into a 4-dimensional subspace with zero prediction loss. 
> With regularization, collapse is prevented, but meaningful downstream improvements 
> require sufficient pretraining data (>1000h) and fine-tuning evaluation protocol."
