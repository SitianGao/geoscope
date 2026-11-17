import unittest
from PIL import Image
from geoscope_tools.statistics import ToolError, region_statistics


class StatisticsTests(unittest.TestCase):
    def test_roi_denominators(self):
        mask = Image.new('L', (10, 10))
        mask.paste(255, (0, 0, 2, 10))
        result = region_statistics(mask, 'roi', [0, 0, 5, 5])
        self.assertEqual((result['target_pixels_in_scope'], result['scope_pixels']), (10, 25))
        self.assertEqual((result['ratio_of_full_image'], result['ratio_within_scope']), (.1, .4))

    def test_empty_full_and_no_overlap(self):
        for value in (0, 255):
            with self.subTest(value=value):
                result = region_statistics(Image.new('L', (10, 10), value))
                self.assertEqual(result['ratio_within_scope'], int(bool(value)))
        mask = Image.new('L', (10, 10)); mask.paste(255, (0, 0, 2, 2))
        self.assertEqual(region_statistics(mask, 'roi', [5, 5, 10, 10])['target_pixels_in_scope'], 0)

    def test_roi_rejections(self):
        for roi in ([], [0, 0, 0, 5], [-1, 0, 5, 5], [0, 0, 11, 5], [5, 0, 2, 5], [0., 0, 5, 5], [False, 0, 5, 5]):
            with self.subTest(roi=roi), self.assertRaises(ToolError):
                region_statistics(Image.new('L', (10, 10)), 'roi', roi)

    def test_odd_halves_partition(self):
        mask = Image.new('L', (5, 3), 255)
        self.assertEqual(region_statistics(mask, 'left_half')['scope_pixels'], 6)
        self.assertEqual(region_statistics(mask, 'right_half')['scope_pixels'], 9)
        self.assertEqual(region_statistics(mask, 'top_half')['scope_pixels'], 5)
        self.assertEqual(region_statistics(mask, 'bottom_half')['scope_pixels'], 10)

    def test_probability_and_conflicting_scope_rejected(self):
        with self.assertRaises(ToolError): region_statistics(Image.new('L', (2, 2), 128))
        with self.assertRaises(ToolError): region_statistics(Image.new('L', (2, 2)), 'all', [0, 0, 1, 1])
