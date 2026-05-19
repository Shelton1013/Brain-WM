#!/bin/bash
# Install ALL EEG-Bench pip dependencies into existing brainwm environment.
# Converted from EEG-Bench/environment.yml pip section.
# Usage: conda activate brainwm && bash install_eeg_bench.sh

pip install \
    accelerate==0.21.0 \
    antropy==0.1.8 \
    beautifulsoup4==4.13.3 \
    blobfile==3.0.0 \
    bs4==0.0.2 \
    colorama==0.4.6 \
    contourpy==1.3.1 \
    coverage==7.6.10 \
    cryptography==44.0.1 \
    dill==0.3.9 \
    diskcache==5.6.3 \
    docopt==0.6.2 \
    edfio==0.4.5 \
    edflib-python==1.0.8 \
    einops==0.7.0 \
    h5py==3.10.0 \
    imageio==2.36.1 \
    lightning==2.5.1 \
    lightning-utilities==0.14.2 \
    llvmlite==0.43.0 \
    lxml==5.3.1 \
    matplotlib==3.10.0 \
    memory-profiler==0.61.0 \
    mne==1.9.0 \
    mne-bids==0.14 \
    moabb==1.1.1 \
    msal==1.31.1 \
    ninja==1.11.1.3 \
    numba==0.60.0 \
    office365-rest-python-client==2.5.14 \
    omegaconf==2.3.0 \
    openpyxl==3.1.5 \
    pandarallel==1.6.5 \
    pandas==1.5.3 \
    pooch==1.8.2 \
    protobuf==3.20.0 \
    pyhealth==1.1.4 \
    pyprep==0.4.3 \
    pyriemann==0.6 \
    pywavelets==1.8.0 \
    pytorch-lightning==2.5.1 \
    rdkit==2024.9.5 \
    resampy==0.4.3 \
    rfpimp==1.3.7 \
    safetensors==0.5.3 \
    scikit-image==0.25.0 \
    scikit-learn==1.4.2 \
    seaborn==0.12.2 \
    stochastic==0.7.0 \
    tensorboardx==1.8 \
    timm==0.4.12 \
    tokenizers==0.15.2 \
    torchinfo==1.8.0 \
    torchmetrics==1.7.0 \
    transformers==4.17.0 \
    wfdb==4.1.0 \
    webdriver-manager==4.0.2 \
    websockets==14.2

echo "Done. Verify: cd /home/share/data_makchen/peng/EEG-Bench && python -c 'import eeg_bench; print(\"OK\")'"
