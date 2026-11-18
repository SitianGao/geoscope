import unittest
from PIL import Image
from geoscope_tools.region_analysis import connected_region_analysis, spatial_relation_analysis


class RegionsTests(unittest.TestCase):
    def test_two_blocks_and_filter(self):
        m = Image.new('L', (10, 10)); m.paste(255, (0, 0, 2, 2)); m.paste(255, (7, 7, 10, 10))
        r = connected_region_analysis(m, min_area_pixels=5)
        self.assertEqual((r['count_before_filter'], r['connected_components_count']), (2, 1))
        self.assertEqual(r['components'][0]['area_pixels'], 9)
        self.assertEqual(r['filtered_components'][0]['reason'], 'below_min_area_pixels')

    def test_diagonal_is_connected(self):
        m = Image.new('L', (3, 3)); m.putpixel((0, 0), 255); m.putpixel((1, 1), 255)
        self.assertEqual(connected_region_analysis(m)['connected_components_count'], 1)

    def test_scope_splits_component(self):
        m = Image.new('L', (5, 5)); m.paste(255, (0, 0, 1, 5)); m.paste(255, (4, 0, 5, 5)); m.paste(255, (0, 4, 5, 5))
        self.assertEqual(connected_region_analysis(m)['connected_components_count'], 1)
        self.assertEqual(connected_region_analysis(m, 'roi', [0, 0, 5, 3])['connected_components_count'], 2)

    def test_spatial_empty_and_pixel_directions(self):
        a = Image.new('L', (10, 10)); b = a.copy()
        self.assertIsNone(spatial_relation_analysis(a, b)['near'])
        a.paste(255, (0, 0, 2, 2)); b.paste(255, (8, 8, 10, 10))
        r = spatial_relation_analysis(a, b)
        self.assertFalse(r['intersects']); self.assertEqual(r['relative_center']['horizontal'], 'left')
