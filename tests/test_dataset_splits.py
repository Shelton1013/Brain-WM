from pathlib import Path
import unittest
import warnings

from build_tuab_lp import validate_patient_splits


class PatientSplitTests(unittest.TestCase):
    def test_test_patient_overlap_fails(self):
        records = [(Path("p1_s1.edf"), 0, "train"), (Path("p1_s2.edf"), 1, "test")]
        with self.assertRaises(ValueError):
            validate_patient_splits(records)

    def test_reference_validation_overlap_is_explicit(self):
        records = [(Path("p1_s1.edf"), 0, "train"), (Path("p1_s2.edf"), 1, "val"),
                   (Path("p2_s1.edf"), 0, "test")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(validate_patient_splits(records), 1)
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
