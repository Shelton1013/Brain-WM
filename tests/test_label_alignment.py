import unittest

from label_utils import map_epoch_labels


class LabelAlignmentTests(unittest.TestCase):
    def test_unknown_annotation_keeps_later_epoch_position(self):
        labels = map_epoch_labels(["0\n", "?\n", "5\n"], {"0": 0, "5": 4})
        self.assertEqual(labels, [0, None, 4])
        self.assertEqual([(i, y) for i, y in enumerate(labels) if y is not None], [(0, 0), (2, 4)])

    def test_known_labels_and_blank_formatting(self):
        self.assertEqual(map_epoch_labels(["0", " ", "5"], {"0": 0, "5": 4}), [0, 4])


if __name__ == "__main__":
    unittest.main()
