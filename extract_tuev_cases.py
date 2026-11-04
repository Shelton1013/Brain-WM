"""Extract the raw TUEV test windows of the case study (indices into the test split, in the
order of the saved predictions) for plotting."""
import json
import sys

import numpy as np

D = "/home/pxieaf/home2/datasets/lp_tasks/tuev/"
cases = json.load(open(sys.argv[1]))
meta = np.load(D + "meta.npz")
test = np.flatnonzero(meta["split"] == "test")
mono = np.load(D + "mono200.npy", mmap_mode="r")
out = {"mono_ch": meta["mono_ch"]}
for name, c in cases.items():
    g = test[c["idx"]]
    out[name] = np.asarray(mono[g], dtype=np.float32)          # [19, 1000] uV at 200 Hz
    out[name + "_y"] = int(meta["y"][g])
    out[name + "_rec"] = str(meta["rec"][g])
np.savez(sys.argv[2], **out)
print({k: (out[k + "_y"], out[k + "_rec"]) for k in cases})
