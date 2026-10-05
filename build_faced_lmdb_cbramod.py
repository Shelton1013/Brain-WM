"""Build CBraMod's FACED LMDB from our rebuilt Processed_data arrays.

Line-for-line CBraMod `preprocessing/preprocessing_faced.py` (same label table,
sorted-subject 80/20/23 split, 250->200 Hz FFT resample, 3 x 10 s windows,
[32, 10, 200] samples in µV; their loader divides by 100). The ONLY change is
the input: our sub-XXX.npy [28, 32, 7500] from preprocess_faced.py instead of
the official pickles, which are the same shape/units/order.

    python build_faced_lmdb_cbramod.py --proc_dir ~/home2/datasets/FACED_proc \
        --out ~/home2/datasets/FACED_cbramod_lmdb
"""
import argparse
import os
import pickle

import lmdb
import numpy as np
from scipy import signal

labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])

p = argparse.ArgumentParser()
p.add_argument("--proc_dir", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
root_dir = os.path.expanduser(a.proc_dir)
files = sorted(f for f in os.listdir(root_dir) if f.startswith("sub-") and f.endswith(".npy"))
files_dict = {"train": files[:80], "val": files[80:100], "test": files[100:]}
dataset = {"train": [], "val": [], "test": []}

db = lmdb.open(os.path.expanduser(a.out), map_size=6612500172)
for files_key in files_dict:
    for file in files_dict[files_key]:
        array = np.load(os.path.join(root_dir, file))
        eeg = signal.resample(array, 6000, axis=2)
        eeg_ = eeg.reshape(28, 32, 30, 200)
        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10 * j:10 * (j + 1), :]
                sample_key = f"{file}-{i}-{j}"
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(),
                        value=pickle.dumps({"sample": sample, "label": label}))
                txn.commit()
                dataset[files_key].append(sample_key)
        print(file, files_key, flush=True)

txn = db.begin(write=True)
txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
print({k: len(v) for k, v in dataset.items()})
