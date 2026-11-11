"""Regression: a segmentation hole must not erase the mouth RGB."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from stylist.segmentation import FaceParsing
from stylist.seasonal_preprocessing import (
    OUTER_LIPS, SeasonalPreprocessing, prepare_seasonal_image,
)


class MouthPreservationTest(unittest.TestCase):
    def setUp(self):
        self.landmarks = [SimpleNamespace(x=.5, y=.5) for _ in range(468)]
        self.landmarks[1] = SimpleNamespace(x=0, y=0)
        self.landmarks[2] = SimpleNamespace(x=1, y=1)
        for i, index in enumerate(OUTER_LIPS):
            angle = 2 * np.pi * i / len(OUTER_LIPS)
            self.landmarks[index] = SimpleNamespace(
                x=.5 + .15 * np.cos(angle), y=.65 + .08 * np.sin(angle))
        labels = np.ones((100, 100), dtype=np.uint8)
        labels[54:77, 32:69] = 0  # SegFormer mistakenly calls the mouth background.
        self.parsing = FaceParsing(labels, {0: 'background', 1: 'skin.png'})
        self.settings = SeasonalPreprocessing(crop_scale=1, center_y_offset=0)

    def test_mouth_survives_background_misclassification(self):
        image = Image.new('RGB', (100, 100), (180, 90, 70))
        protected = np.asarray(prepare_seasonal_image(
            image, self.parsing, self.settings, self.landmarks))
        unprotected = np.asarray(prepare_seasonal_image(
            image, self.parsing, replace(self.settings, preserve_mouth_landmarks=False), self.landmarks))
        np.testing.assert_array_equal(protected[145, 112], [180, 90, 70])
        np.testing.assert_array_equal(unprotected[145, 112], [0, 0, 0])
        # Nearby background outside the lip contour stays masked.
        np.testing.assert_array_equal(protected[123, 75], [0, 0, 0])

    def test_original_black_mouth_is_not_repainted(self):
        pixels = np.full((100, 100, 3), 180, dtype=np.uint8)
        pixels[54:77, 32:69] = 0
        protected = np.asarray(prepare_seasonal_image(
            Image.fromarray(pixels), self.parsing, self.settings, self.landmarks))
        np.testing.assert_array_equal(protected[145, 112], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
