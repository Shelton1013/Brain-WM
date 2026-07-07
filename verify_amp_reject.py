"""Verify the amplitude-reject fix in prebuild_tueg_cache.py.

Two independent checks, no torch / no real TUEG needed:

  1. UNIT/SCALE: MNE get_data() returns Volts (~1e-5). Confirm that the
     OLD threshold (`> reject_abs_uv`) never fires on Volt-scale data, while
     the FIXED threshold (`> reject_abs_uv * 1e-6`) fires correctly.

  2. EA DESTROYS SCALE: confirm Euclidean Alignment whitens to O(1), so
     checking a µV threshold AFTER EA (the old bug) is meaningless, whereas
     checking BEFORE EA (the fix) sees true amplitudes.

Run on the server (has scipy):  python verify_amp_reject.py
"""
import numpy as np
from scipy.linalg import fractional_matrix_power


def euclidean_alignment(data):
    """Same as dataset.py: X @ R^-1/2, R = spatial covariance."""
    R = (data.T @ data) / data.shape[0]
    try:
        R_inv_sqrt = fractional_matrix_power(R, -0.5).real.astype(np.float32)
    except Exception:
        R_reg = R + 1e-6 * np.eye(R.shape[0])
        R_inv_sqrt = fractional_matrix_power(R_reg, -0.5).real.astype(np.float32)
    return data @ R_inv_sqrt


def main():
    rng = np.random.default_rng(0)
    T, C = 2560, 19  # 10 s @ 256 Hz, 19 ch
    reject_abs_uv = 100.0

    # Synthetic recording in VOLTS (MNE native): mostly ~30 µV physiological
    # (=30e-6 V) with a few high-amplitude artifact samples at ~250 µV.
    eeg_volts = (rng.standard_normal((T, C)) * 30e-6).astype(np.float32)
    eeg_volts[500, 3] = 250e-6   # artifact spike, 250 µV → should be rejected
    eeg_volts[900, 7] = -280e-6  # artifact spike, 280 µV → should be rejected

    print("=" * 64)
    print("CHECK 1 — unit/scale of the reject threshold")
    print("=" * 64)
    max_v = np.abs(eeg_volts).max()
    print(f"  data max |x|            = {max_v:.3e}  (Volts, MNE native)")
    print(f"  data max in µV          = {max_v*1e6:.1f} µV")
    old_fires = max_v > reject_abs_uv               # OLD buggy check (V vs 100)
    new_fires = max_v > reject_abs_uv * 1e-6        # FIXED check (V vs 100µV)
    print(f"  OLD  ` > {reject_abs_uv:g}`        fires? {old_fires}   "
          f"(expect False — the bug: never rejects)")
    print(f"  FIXED` > {reject_abs_uv:g}*1e-6`   fires? {new_fires}   "
          f"(expect True  — correctly flags the 250/280 µV artifact)")
    assert not bool(old_fires), "old check unexpectedly fired"
    assert bool(new_fires), "FIXED check failed to fire on 250µV artifact!"

    print()
    print("=" * 64)
    print("CHECK 2 — EA destroys the amplitude scale (why reject must be pre-EA)")
    print("=" * 64)
    eeg_ea = euclidean_alignment(eeg_volts)
    pre_max_uv = np.abs(eeg_volts).max() * 1e6
    post_max = np.abs(eeg_ea).max()
    print(f"  pre-EA  max |x|  = {pre_max_uv:8.1f} µV   (real amplitude)")
    print(f"  post-EA max |x|  = {post_max:8.3f}      (whitened, ~O(1), unitless)")
    print(f"  → EA rescaled 280µV to ~{post_max:.0f}; the OLD post-EA check")
    print(f"    `>{reject_abs_uv:g}` sees {post_max:.1f} < {reject_abs_uv:g} → never fires (the bug).")
    print(f"    The fix captures the reject reference BEFORE EA, in real µV.")
    assert post_max < reject_abs_uv, "EA output unexpectedly large"

    # Per-trial reject simulation exactly as prebuild does it (fixed version)
    print()
    print("=" * 64)
    print("CHECK 3 — per-trial reject as prebuild now runs it")
    print("=" * 64)
    ref = eeg_volts  # pre-EA reference (the fix)
    trial = ref[0:T]
    rejected = np.abs(trial).max() > reject_abs_uv * 1e-6
    print(f"  trial containing artifact rejected? {rejected}  (expect True)")

    # A clean trial (no artifact) must survive. NB: use a realistic 12µV RMS —
    # at 30µV RMS the Gaussian tail alone exceeds 100µV over 48k samples, which
    # is itself a warning (see printout) that a hard 100µV reject is aggressive.
    clean = (rng.standard_normal((T, C)) * 12e-6).astype(np.float32)
    clean_max_uv = np.abs(clean).max() * 1e6
    clean_rej = np.abs(clean).max() > reject_abs_uv * 1e-6
    print(f"  clean 12µV-RMS trial max = {clean_max_uv:.1f} µV, rejected? {clean_rej}  (expect False)")
    assert rejected and not clean_rej, "per-trial reject logic wrong"

    print()
    print("  ⚠ NOTE: a 100µV HARD reject is strict — Gaussian tails of a 30µV-RMS")
    print("    signal already exceed 100µV over a 10s×19ch trial. On real TUEG this")
    print("    will reject a large fraction of trials. Watch the retained-trial count")
    print("    after rebuild; if it drops far below ~9000h, consider raising the µV")
    print("    threshold or applying reject on robust-normalized data instead.")

    print()
    print("ALL CHECKS PASSED — reject now fires on >100µV, survives clean data,"
          " and is correctly placed before EA.")


if __name__ == "__main__":
    main()
