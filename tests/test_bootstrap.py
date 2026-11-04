import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper.arxiv import bootstrap_metrics as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_sleep_readout_does_not_silently_switch_to_center10(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            y = np.array([0, 0, 1, 1])
            subj = np.array(["a", "b", "c", "d"])
            for name, pred in [("m_s42__avg3", y), ("cbramod__avg3", y),
                               ("cbramod__center10", 1 - y)]:
                np.savez(p / f"isruc__{name}.npz", y=y, pred=pred, subj=subj, val_ba=1.)
            with contextlib.redirect_stdout(io.StringIO()):
                bootstrap.main(p, p / "out.json", B=10)
            result = json.loads((p / "out.json").read_text())
            self.assertEqual(result["isruc"]["models"]["cbramod"]["balanced_accuracy"]["point"], 1.)

    def test_duplicate_model_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ("m_s42__avg3", "cbramod", "cbramod__avg3"):
                np.savez(p / f"isruc__{name}.npz", y=[0, 1], pred=[0, 1],
                         subj=["a", "b"], val_ba=1.)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                bootstrap.main(p, p / "out.json", B=10)

    def test_multiclass_subject_is_not_fixed_by_singleton_majority_stratum(self):
        y = np.array([0, 0, 0, 1, 2, 3, 4] * 9 + [1, 1, 1, 0, 2, 3, 4])
        subj = np.repeat(np.arange(10), 7)
        majorities = [np.bincount(y[subj == s]).argmax() for s in range(10)]
        self.assertEqual(majorities, [0] * 9 + [1])
        draws, scheme, info = bootstrap.bootstrap_draws(y, subj, B=100)
        self.assertEqual(scheme, "ordinary_subject_class_coverage")
        self.assertEqual(info["rejected_missing_class"], 0)
        counts = [int((subj[d] == 9).sum()) // 7 for d in draws]
        self.assertGreater(np.std(counts), 0.)
        self.assertTrue(all(len(np.unique(y[d])) == 5 for d in draws))


if __name__ == "__main__":
    unittest.main()
