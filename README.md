# SpecBraM

EEG pretraining by masked band-power prediction. The current model predicts five classical
band powers from masked channel/time patches using a linear patch tokenizer and a criss-cross
transformer. The target is a patch-local periodogram followed by `log(P + 1e-3)`; the current
training loss has no JEPA or SIGReg contribution.

## Current implementation

| Component | Paper configuration |
|---|---|
| Input | 256 Hz, 19 canonical EEG channels, 10-second windows |
| Patches | 200 samples, 12 patches/channel; the final 160 samples are discarded |
| Encoder | 12 criss-cross blocks, width 512, 8 attention heads |
| Masking | Whole channel/time patches, probability 0.3 |
| Target bands | 0.5–4, 4–8, 8–13, 13–30, 30–45 Hz (half-open intervals) |
| Target power | `2 * sum(abs(rfft(patch))**2) / 200**2` within each band |
| Loss | Mean squared log-power error on masked patches |
| Pretraining | 2,388 hours, same 2,100-patient subset for seeds 42/43/44 |

The legacy model constructors and training CLI retain older defaults for checkpoint
compatibility. **Use the explicit paper configuration**, rather than assuming that
`EEGLeJEPA_v3()` or the default training command constructs the current paper model.
Earlier Hann-filter/log1p experiments remain in historical results and are not the current
method. [MODEL_USAGE.md](MODEL_USAGE.md) explains checkpoint loading and feature extraction.

## Installation

Use Python 3.10 or newer and install a suitable PyTorch build for your CPU/CUDA setup, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

External baseline adapters additionally require the corresponding official repositories,
weights and dependencies (e.g. timm, safetensors, einops). These are not bundled. The
baseline directory and prepared non-sleep datasets can be set with the environment variables
`SPECBRAM_BASELINES` and `SPECBRAM_TASK_DATA`; the defaults describe the original experiment layout.
EEG datasets and model weights are not stored in Git.
Per-sample prediction/feature NPZ files are also kept out of Git; regenerate them with the
evaluation scripts' `--pred_dir` option when reproducing the cohort bootstrap.

## Reproduce pretraining

[configs/specbram_main.json](configs/specbram_main.json) records the main checkpoint's
settings. It is a reference config, not an automatically loaded CLI file. An equivalent
four-GPU launch is:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train_v2.py \
  --arch v3 --tokenizer_type linear --mask_axis time --band_mask_ratio 0.3 \
  --target_type fft_bandpower --fft_bands standard --fft_taper rect --target_log log \
  --jepa_weight 0 --cf_weight 1 --sigreg_lambda 0 \
  --d_model 512 --encoder_layers 12 --patch_len 200 --cf_d_band 64 \
  --epochs 105 --stop_after_epoch 16 --warmup_epochs 7 \
  --batch_size 128 --lr 0.0017 --weight_decay 0.05 --grad_clip 3 \
  --pretrain_n_subjects 2100 --subset_seed 42 --seed 42 \
  --normalization per_recording_robust --trial_duration_s 10 \
  --drop_short_recording_min 5 --trim_start_end_sec 60 --reject_abs_uv 150 \
  --num_workers 3 --tueg_dir /path/to/TUEG/edf \
  --data_cache_dir /path/to/cache --output_dir /path/to/output
```

Change `--seed` to 43 or 44 while keeping `--subset_seed 42`. The cosine schedule is
defined over 105 epochs and training is stopped at epoch 16; this is not a 16-epoch cosine
schedule. The original training implementation still evaluates auxiliary branches whose
loss weights are zero. Step-count comparisons should not be interpreted as measured
wall-clock or FLOP speedups.

## Evaluation and result provenance

The primary evaluation freezes the encoder, mean-pools tokens, standardizes features using
the training split, and selects class-balanced logistic-regression regularization on the
validation split. A sleep epoch averages three 10-second features. This differs from the
older frozen Transformer sequence head and from full fine-tuning.

- Sleep: `eval_linear_probe_v2.py`; main-model example:

  ```bash
  python eval_linear_probe_v2.py --dataset isruc --data_dir /path/to/ISRUC \
    --model specbram --checkpoints main=/path/to/checkpoint_ep16.pt \
    --modes avg3 --output /path/to/isruc_probe.json
  ```

- Other tasks: `build_task_npz.py`, `build_tuab_lp.py`, `build_tuev_lp.py`,
  `eval_linear_probe_tasks.py`, and `eval_linear_probe_tuab.py`.
- Subject/recording bootstrap: `paper/arxiv/bootstrap_metrics.py`.
- Aggregates and paper tables: `results/arxiv_data/`, `results/bootstrap/boot.json`,
  `paper/arxiv/make_arxiv_tables.py`.

The eight main datasets are ISRUC, HMC, TUAB, TUEV, MA, Mumtaz, FACED and PhysioNet-MI.
Main comparison rows use a single checkpoint with a test-cohort bootstrap confidence
interval; target ablations use three pretraining seeds and report their standard deviation.
These describe different sources of uncertainty. The reference per-class TUAB preparation
shares three patients between train and validation; its test patients are disjoint from both.
TUAB/TUEV patients were not excluded
from TUEG pretraining, so those tasks are not overlap-free transfer evaluations.

## Manuscript

The editable master is [paper/arxiv/specbram_arxiv.tex](paper/arxiv/specbram_arxiv.tex).
The compiled draft is [paper/arxiv/specbram_arxiv.pdf](paper/arxiv/specbram_arxiv.pdf).
The CVPR layout is [paper/cvpr27/specbram_cvpr27/main.tex](paper/cvpr27/specbram_cvpr27/main.tex).
These are research drafts, not a claim of conference acceptance. See
[paper/README.md](paper/README.md) for build instructions and the relationship between
current and historical versions.
