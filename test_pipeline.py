"""Tests for the e-waste classifier pipeline.

These cover the things that were actually broken when the project was picked up,
rather than asserting that a neural network produces particular logits - a test
that hard-codes expected probabilities would break on every retrain and prove
nothing. What is checked here is the surrounding machinery: the checkpoint
loads, the architecture matches the label count, predictions stay in range, and
the labels agree with the dataset layout.

Run with:  python test_pipeline.py
       or: pytest test_pipeline.py
"""

import random
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHECKPOINT = HERE / "ewaste_mobilenet_master_model.repaired.pth"
DATASET = HERE / "MASTER_e_waste_visual_dataset"

import predict as predictor  # noqa: E402


class TestLabels(unittest.TestCase):
    def test_labels_file_is_not_empty(self):
        self.assertGreater(len(predictor.load_labels()), 0)

    def test_labels_match_dataset_folders(self):
        """The label order is the model's class index, so the two must agree."""
        labels = predictor.load_labels()
        if not DATASET.exists():
            self.skipTest(f"{DATASET} not present")
        folders = sorted(p.name for p in DATASET.iterdir() if p.is_dir())
        self.assertEqual(sorted(labels), folders)

    def test_no_duplicate_labels(self):
        labels = predictor.load_labels()
        self.assertEqual(len(labels), len(set(labels)))


class TestCheckpoint(unittest.TestCase):
    """The checkpoint shipped corrupt: a stray .py at the zip root."""

    def setUp(self):
        if not CHECKPOINT.exists():
            self.skipTest("checkpoint not present; run repair_checkpoint.py")

    def test_checkpoint_loads(self):
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertIsInstance(state, dict)
        self.assertGreater(len(state), 100)

    def test_architecture_matches_label_count(self):
        model, labels = predictor.build_model(CHECKPOINT)
        self.assertEqual(model.classifier[3].out_features, len(labels))

    def test_original_checkpoint_is_still_documented_as_broken(self):
        """The original is kept for reference; it must stay unloadable."""
        original = HERE / "ewaste_mobilenet_master_model.pth"
        if not original.exists():
            self.skipTest("original checkpoint not present")
        with self.assertRaises(Exception):
            torch.load(original, map_location="cpu", weights_only=False)


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.exists() or not DATASET.exists():
            raise unittest.SkipTest("checkpoint or dataset not present")
        cls.model, cls.labels = predictor.build_model(CHECKPOINT)
        cls.transform = predictor.preprocessor()
        cls.images = sorted(
            p for p in DATASET.rglob("*")
            if p.suffix.lower() in predictor.IMAGE_SUFFIXES
        )

    def _images(self):
        if not self.images:
            self.skipTest("no images in dataset")
        return self.images

    def test_probabilities_are_a_distribution(self):
        for path in self._images()[:5]:
            tensor = self.transform(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs = self.model(tensor).softmax(dim=1)[0]
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
            self.assertTrue(bool((probs >= 0).all()))

    def test_predictions_are_in_range(self):
        """A predicted index must map to a real label.

        This is the check that catches a class-count mismatch, which is the
        failure mode of a checkpoint trained on an older label set.
        """
        for path in self._images()[:10]:
            tensor = self.transform(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                top = int(self.model(tensor).argmax())
            self.assertTrue(0 <= top < len(self.labels))

    def test_accuracy_meets_a_floor(self):
        """Guards against a checkpoint that loads but is untrained.

        The threshold is deliberately loose - the model scores 97% on this
        dataset - because the point is to catch a broken or randomly
        initialised model, not to pin a metric that will move on retraining.
        """
        images = self._images()
        random.seed(0)
        sample = images if len(images) <= 40 else random.sample(images, 40)

        correct = 0
        for path in sample:
            truth = path.parent.name
            if truth not in self.labels:
                continue
            tensor = self.transform(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                top = int(self.model(tensor).argmax())
            correct += self.labels[top] == truth

        self.assertGreater(correct / len(sample), 0.5,
                           f"only {correct}/{len(sample)} correct - "
                           "the checkpoint may be untrained or mismatched")

    def test_preprocessing_shape(self):
        tensor = self.transform(Image.new("RGB", (640, 480)))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))


class TestScriptArguments(unittest.TestCase):
    def test_missing_checkpoint_exits_cleanly(self):
        """A bad path should be a readable message, not a stack trace."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(HERE / "predict.py"),
             "--checkpoint", "/nonexistent.pth", "image.jpg"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("checkpoint not found", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
