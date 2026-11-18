import unittest
from PIL import Image
from geoscope_tools.statistics import ToolError
from geoscope_tools.version_compare import mask_compare


class CompareTests(unittest.TestCase):
    def test_recomputable_diff(self):
        a = Image.new('L', (10, 10)); a.paste(255, (0, 0, 5, 5))
        b = Image.new('L', (10, 10)); b.paste(255, (2, 0, 7, 5))
        r, diff = mask_compare(a, b)
        self.assertEqual((r['intersection_pixels'], r['union_pixels'], r['added_pixels'], r['removed_pixels']), (15, 35, 10, 10))
        self.assertEqual(r['prediction_agreement_iou'], 15/35)
        self.assertEqual(diff.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(diff.getpixel((6, 0)), (0, 255, 0))

    def test_empty_rule_and_dimensions(self):
        a = Image.new('L', (3, 3))
        self.assertEqual(mask_compare(a, a)[0]['prediction_agreement_iou'], 1)
        self.assertEqual(mask_compare(a, Image.new('L', (3, 3), 255))[0]['prediction_agreement_iou'], 0)
        with self.assertRaises(ToolError): mask_compare(a, Image.new('L', (4, 3)))
