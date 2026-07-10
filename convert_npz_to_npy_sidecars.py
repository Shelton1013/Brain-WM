"""Convert chunked .npz TUEG cache -> per-chunk .npy sidecars for mmap loading.

The chunked-cache loader (dataset_multi.py) uses <stem>_trials.npy /
<stem>_sids.npy via np.load(mmap_mode='r') when present, and only falls back to
decompressing the whole .npz into RAM otherwise. A full TUEG 10 s cache is ~1.8 TB
compressed; without sidecars each DDP rank tries to decompress the WHOLE thing
into RAM and thrashes. This script writes the sidecars once (bounded RAM: one
chunk at a time; resumable: skips chunks already done).

    python convert_npz_to_npy_sidecars.py <chunk_dir>
    # e.g. .../dataset_cache_no_exclude/tueg_72c4397aabf30d5e
"""
import sys
import time
from pathlib import Path

import numpy as np


def main(chunk_dir: str, max_chunks: int = 0):
    d = Path(chunk_dir)
    npzs = sorted(d.glob("chunk_*.npz"))
    if not npzs:
        raise SystemExit(f"no chunk_*.npz under {d}")
    if max_chunks > 0:
        npzs = npzs[:max_chunks]
        print(f"converting FIRST {len(npzs)} chunks only (--max_chunks)")
    print(f"{len(npzs)} chunks in {d}")
    for i, npz in enumerate(npzs):
        stem = npz.stem                       # chunk_0000
        t_out = d / f"{stem}_trials.npy"
        s_out = d / f"{stem}_sids.npy"
        if t_out.exists() and s_out.exists():
            print(f"[{i+1}/{len(npzs)}] skip {stem} (sidecars exist)", flush=True)
            continue
        t0 = time.time()
        z = np.load(str(npz))
        trials = z["trials"]
        sids = z["subject_ids"]
        # write to a .tmp then rename so an interrupted run never leaves a
        # half-written sidecar that the loader would mmap as valid.
        np.save(str(t_out) + ".tmp.npy", trials)
        np.save(str(s_out) + ".tmp.npy", sids)
        Path(str(t_out) + ".tmp.npy").rename(t_out)
        Path(str(s_out) + ".tmp.npy").rename(s_out)
        gb = trials.nbytes / 1e9
        print(f"[{i+1}/{len(npzs)}] {stem}: {trials.shape} {trials.dtype} "
              f"-> {gb:.1f} GB in {time.time()-t0:.0f}s", flush=True)
        del z, trials, sids
    print("done — sidecars written; loader will mmap them now.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python convert_npz_to_npy_sidecars.py "
                         "<chunk_dir> [max_chunks]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)
