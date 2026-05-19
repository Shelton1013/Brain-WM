#!/bin/bash
# Install EEG-Bench dependencies into existing brainwm environment.
# Usage: conda activate brainwm && bash install_eeg_bench.sh

pip install \
    accelerate==0.21.0 \
    antropy==0.1.8 \
    beautifulsoup4==4.13.3 \
    bs4==0.0.2 \
    edfio==0.4.5 \
    edflib-python==1.0.8 \
    einops==0.7.0 \
    h5py==3.10.0 \
    lightning==2.5.1 \
    mne-bids==0.14 \
    moabb==1.1.1 \
    omegaconf==2.3.0 \
    openpyxl==3.1.5 \
    pandarallel==1.6.5 \
    pyhealth==1.1.4 \
    pyprep==0.4.3 \
    pyriemann==0.6 \
    pytorch-lightning==2.5.1 \
    resampy==0.4.3 \
    rfpimp==1.3.7 \
    scikit-image==0.25.0 \
    seaborn==0.12.2 \
    stochastic==0.7.0 \
    tensorboardx==1.8 \
    timm==0.4.12 \
    torchinfo==1.8.0 \
    torchmetrics==1.7.0 \
    wfdb==4.1.0

echo "Done. Verify: cd /home/share/data_makchen/peng/EEG-Bench && python -c 'import eeg_bench; print(\"OK\")'"
