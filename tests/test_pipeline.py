"""Synthetic API regression tests; these fixtures are never experiment data."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from traffic_signs.features import (
    extract_features, feature_vector, hog_visualization, load_rgb, preprocess,
)


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        # Keep test writes in the project instead of the Windows global Temp.
        output_root = Path(__file__).resolve().parents[1] / "outputs"
        output_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=output_root)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        # Deterministic colored rectangle with a non-square source and ROI.
        rgb = np.zeros((48, 40, 3), dtype=np.uint8)
        rgb[8:40, 10:30] = [255, 128, 32]
        self.path = self.folder / "synthetic_fixture.png"
        Image.fromarray(rgb).save(self.path)
        self.config = dict(image_size=32, use_roi=False, output_dir=str(self.folder))
        self.manifest = pd.DataFrame([dict(
            path=str(self.path), label=0, split="train", track="synthetic_fixture",
            x1=10, y1=8, x2=29, y2=39,
        )])

    def test_image_and_manifest_inference_are_identical(self):
        gray = preprocess(self.path, self.config)
        self.assertEqual(gray.shape, (32, 32))
        self.assertEqual(gray.dtype, np.float32)
        self.assertGreaterEqual(gray.min(), 0)
        self.assertLessEqual(gray.max(), 1)
        for kind, dimension in (("hog", 324), ("pixels", 1024)):
            batch = extract_features(self.manifest, self.config, kind=kind)
            self.assertEqual(batch.shape, (1, dimension))
            np.testing.assert_array_equal(batch[0], feature_vector(gray, self.config, kind))
        self.assertEqual(hog_visualization(gray, self.config).shape, gray.shape)

    def test_undistorted_baseline_and_order(self):
        baseline = preprocess(self.path, self.config)
        np.testing.assert_array_equal(
            baseline, preprocess(self.path, self.config, brightness=1.0, blur=0.0))
        dark = preprocess(self.path, self.config, brightness=.3)
        self.assertLess(dark.mean(), baseline.mean())
        self.assertFalse(np.array_equal(baseline, preprocess(self.path, self.config, blur=2)))
        np.testing.assert_array_equal(
            preprocess(self.path, self.config, brightness=.3, blur=2),
            np.asarray(load_rgb(self.path, self.config, brightness=.3, blur=2).convert("L"),
                       dtype=np.float32) / np.float32(255))

    def test_cache_separates_dimensions_distortions_and_roi(self):
        base = extract_features(self.manifest, self.config)
        np.testing.assert_array_equal(base, extract_features(self.manifest, self.config))
        self.assertEqual(len(list((self.folder / "cache").glob("*.npy"))), 1)
        larger = extract_features(self.manifest, dict(self.config, image_size=40))
        self.assertEqual(larger.shape, (1, 576))
        darker = extract_features(self.manifest, self.config, brightness=.3)
        roi = extract_features(self.manifest, dict(self.config, use_roi=True))
        self.assertFalse(np.array_equal(base, darker))
        self.assertFalse(np.array_equal(base, roi))
        self.assertEqual(len(list((self.folder / "cache").glob("*.npy"))), 4)

    def test_inclusive_roi_coordinates_and_required_metadata(self):
        config = dict(self.config, use_roi=True)
        gray = preprocess(self.path, config, roi=[10, 8, 29, 39])
        # Entire ROI has one color; an exclusive or shifted crop changes edges.
        np.testing.assert_array_equal(gray, np.full((32, 32), gray[0, 0], np.float32))
        self.assertGreater(gray[0, 0], .5)
        with self.assertRaises(ValueError):
            preprocess(self.path, config)
        for roi in ([-1, 8, 29, 39], [10, 8, 40, 39], [29, 8, 10, 39]):
            with self.assertRaises(ValueError):
                preprocess(self.path, config, roi=roi)

    def test_manifest_row_order_survives_cache(self):
        another = self.folder / "synthetic_second.png"
        Image.new("RGB", (32, 32), (0, 0, 0)).save(another)
        second = self.manifest.iloc[0].to_dict()
        second.update(path=str(another), label=1, track="synthetic_second")
        manifest = pd.concat([self.manifest, pd.DataFrame([second])], ignore_index=True)
        forward = extract_features(manifest, self.config)
        backward = extract_features(manifest.iloc[::-1], self.config)
        np.testing.assert_array_equal(forward[::-1], backward)
        self.assertEqual(len(list((self.folder / "cache").glob("*.npy"))), 2)

    def test_audited_content_hash_invalidates_disk_and_gray_caches(self):
        # PPM has fixed-length raw RGB bytes, so changing pixels preserves size.
        path = self.folder / "synthetic_audited.ppm"
        with Image.open(self.path) as image:
            image.save(path)
        before_stat = path.stat()
        manifest = self.manifest.copy()
        manifest.loc[0, "path"] = str(path)
        manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        before = {kind: extract_features(manifest, self.config, kind=kind)
                  for kind in ("hog", "pixels")}
        # Populate and reuse the bounded grayscale cache in addition to disk.
        np.testing.assert_array_equal(
            before["pixels"], extract_features(manifest, self.config, kind="pixels", cache=False))
        with Image.open(path) as image:
            changed = np.asarray(image).copy()
        changed[:24, :20] = [32, 255, 64]
        Image.fromarray(changed).save(path)
        os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        after_stat = path.stat()
        self.assertEqual(before_stat.st_size, after_stat.st_size)
        self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)
        new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertNotEqual(manifest.loc[0, "sha256"], new_hash)
        manifest["sha256"] = new_hash
        fresh_gray = preprocess(path, self.config)
        for kind in ("hog", "pixels"):
            after = extract_features(manifest, self.config, kind=kind)
            self.assertFalse(np.array_equal(before[kind], after))
            np.testing.assert_array_equal(after[0], feature_vector(fresh_gray, self.config, kind))
            np.testing.assert_array_equal(after, extract_features(manifest, self.config, kind=kind))
        self.assertEqual(len(list((self.folder / "cache").glob("*.npy"))), 4)


if __name__ == "__main__":
    unittest.main()
