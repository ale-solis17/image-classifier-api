import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from app.services.service import classify_image
from app.services.train import _split


class FakeModel:
    def __init__(self, probs):
        self._probs = np.array([probs], dtype=np.float32)

    def predict(self, x, verbose=0):
        return self._probs


class SplitTests(unittest.TestCase):
    def test_stratified_split_keeps_classes_when_possible(self):
        paths = [f"img-{idx}" for idx in range(10)]
        y = [0] * 5 + [1] * 5

        train_paths, train_y, val_paths, val_y = _split(paths, y)

        self.assertEqual(len(train_paths) + len(val_paths), len(paths))
        self.assertCountEqual(train_paths + val_paths, paths)
        self.assertEqual(set(train_y), {0, 1})
        self.assertEqual(set(val_y), {0, 1})

    def test_stratified_split_reduces_validation_for_tiny_classes(self):
        paths = [f"img-{idx}" for idx in range(10)]
        y = [0] * 9 + [1]

        train_paths, train_y, val_paths, val_y = _split(paths, y)

        self.assertEqual(len(train_paths) + len(val_paths), len(paths))
        self.assertIn(1, train_y)
        self.assertNotIn(1, val_y)

    def test_stratified_split_fails_without_validation_samples(self):
        paths = ["img-0", "img-1"]
        y = [0, 1]

        with self.assertRaisesRegex(ValueError, "validacion estratificado"):
            _split(paths, y)


class ClassifyImageTests(unittest.TestCase):
    def test_accepts_prediction_with_threshold_and_margin(self):
        model = FakeModel([0.82, 0.11, 0.07])

        with patch("app.services.service.load_image", return_value=np.zeros((224, 224, 3), dtype=np.float32)):
            result = asyncio.run(classify_image("unused", model, ["a", "b", "c"]))

        self.assertEqual(result["label"], "a")

    def test_rejects_prediction_below_threshold(self):
        model = FakeModel([0.66, 0.20, 0.14])

        with patch("app.services.service.load_image", return_value=np.zeros((224, 224, 3), dtype=np.float32)):
            result = asyncio.run(classify_image("unused", model, ["a", "b", "c"]))

        self.assertEqual(result["label"], "No reconocido")

    def test_rejects_prediction_with_small_margin(self):
        model = FakeModel([0.74, 0.69, 0.01])

        with patch("app.services.service.load_image", return_value=np.zeros((224, 224, 3), dtype=np.float32)):
            result = asyncio.run(classify_image("unused", model, ["a", "b", "c"]))

        self.assertEqual(result["label"], "No reconocido")


if __name__ == "__main__":
    unittest.main()

