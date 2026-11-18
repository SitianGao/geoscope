"""GeoScope-AFR 确定性算法单测（合成概率图，无需 GPU / 模型）。

运行：PYTHONPATH=. python tests/test_geoscope_algorithms.py -v
"""
import unittest

import numpy as np

from geoscope_algorithms.adaptive_refinement import Budget, select_regions, tile_grid
from geoscope_algorithms.consistency import (
    DISAGREEMENT_NOTICE,
    composite_disagreement,
    disagreement_mask,
    region_disagreement_stats,
    review_regions,
)
from geoscope_algorithms.fusion import fuse, threshold_binary
from geoscope_algorithms.region_scoring import (
    AFRWeights,
    Region,
    binary_entropy,
    edge_density,
    score_regions,
)


class EntropyTests(unittest.TestCase):
    def test_max_at_half(self):
        h05 = binary_entropy(np.array([0.5], dtype=np.float32))[0]
        h03 = binary_entropy(np.array([0.3], dtype=np.float32))[0]
        h005 = binary_entropy(np.array([0.05], dtype=np.float32))[0]
        self.assertGreater(h05, h03)
        self.assertGreater(h03, h005)

    def test_bounded_at_extremes(self):
        for v in (0.0, 1.0):
            h = binary_entropy(np.array([v], dtype=np.float32))[0]
            self.assertGreaterEqual(h, 0.0)  # 裁剪后 p∈{0,1} 精确为 0
            self.assertLess(h, 1e-4)

    def test_clips_out_of_range(self):
        np.testing.assert_allclose(binary_entropy(np.array([1.5])),
                                   binary_entropy(np.array([1.0])))
        np.testing.assert_allclose(binary_entropy(np.array([-0.2])),
                                   binary_entropy(np.array([0.0])))

    def test_quantized_close_to_float(self):
        rng = np.random.default_rng(7)
        p = rng.random((64, 64)).astype(np.float32)
        q = np.rint(p * 255) / 255.0
        diff = float(np.abs(binary_entropy(p) - binary_entropy(q)).max())
        self.assertLess(diff, 0.05)


class EdgeDensityTests(unittest.TestCase):
    def test_constant_image_zero(self):
        gray = np.full((32, 32), 128, dtype=np.float32)
        self.assertTrue(np.allclose(edge_density(gray), 0.0, atol=1e-6))

    def test_linear_ramp_constant(self):
        gray = np.arange(64, dtype=np.float32)[None, :].repeat(32, axis=0)
        density = edge_density(gray)
        self.assertTrue(np.allclose(density, 1.0 / (255 * np.sqrt(2)), atol=1e-5))


def _service_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


class TileGridTests(unittest.TestCase):
    def _assert_matches_service(self, width, height):
        regions = tile_grid(width, height, 1024, 256)
        expected = []
        index = 0
        for y in _service_positions(height, 1024, 256):
            for x in _service_positions(width, 1024, 256):
                expected.append((x, y, min(1024, width - x), min(1024, height - y), index))
                index += 1
        self.assertEqual([(r.x, r.y, r.w, r.h, r.index) for r in regions], expected)
        for r in regions:
            self.assertTrue(0 <= r.x and r.x + r.w <= width)
            self.assertTrue(0 <= r.y and r.y + r.h <= height)

    def test_smaller_than_tile(self):
        self._assert_matches_service(512, 300)

    def test_exact_multiple(self):
        self._assert_matches_service(2048, 2048)

    def test_odd_and_edge_pinning(self):
        self._assert_matches_service(3000, 1500)
        self._assert_matches_service(2049, 2049)

    def test_grid_count(self):
        self.assertEqual(len(tile_grid(2048, 2048)), 9)
        self.assertEqual(len(tile_grid(3000, 1500)), 8)
        self.assertEqual(len(tile_grid(512, 512)), 1)


class BudgetAndSelectionTests(unittest.TestCase):
    def test_resolve(self):
        self.assertEqual(Budget(max_regions=3).resolve(10), 3)
        self.assertEqual(Budget(max_regions=30).resolve(10), 10)
        self.assertEqual(Budget(fraction=0.25).resolve(10), 2)
        self.assertEqual(Budget(fraction=0.25).resolve(1), 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            Budget(max_regions=1, fraction=0.5).resolve(10)
        with self.assertRaises(ValueError):
            Budget().resolve(10)
        with self.assertRaises(ValueError):
            Budget(max_regions=-1).resolve(10)
        with self.assertRaises(ValueError):
            Budget(fraction=1.5).resolve(10)

    def test_greedy_order_and_tie_by_index(self):
        regions = [Region(0, 0, 10, 10, i) for i in range(4)]
        scores = [
            regions[0], regions[1], regions[2], regions[3],
        ]
        scored = []
        for region, value in [(regions[0], 0.5), (regions[1], 0.9),
                              (regions[2], 0.5), (regions[3], 0.1)]:
            scored.append(type('S', (), {'region': region, 'score': value})())
        selected = select_regions(scored, Budget(max_regions=3))
        self.assertEqual([s.region.index for s in selected], [1, 0, 2])


class ScoringTests(unittest.TestCase):
    def _region(self):
        return Region(0, 0, 8, 8, 0)

    def test_empty_mask_flag_no_score_injection(self):
        prob = np.zeros((16, 16), dtype=np.float32)
        gray = np.full((16, 16), 100, dtype=np.float32)
        regions = [Region(0, 0, 16, 16, 0)]
        scores = score_regions(prob, gray, regions)
        self.assertTrue(scores[0].empty_mask)
        self.assertEqual(scores[0].entropy, 0.0)

    def test_disagreement_term_computed(self):
        p0 = np.full((16, 16), 0.5, dtype=np.float32)
        gray = np.full((16, 16), 100, dtype=np.float32)
        regions = [Region(0, 0, 16, 16, 0)]
        local = type('L', (), {'probability': np.full((16, 16), 0.9, dtype=np.float32)})()
        scores = score_regions(p0, gray, regions,
                               weights=AFRWeights(alpha=0.0, beta=1.0, gamma=0.0),
                               local_results={0: local})
        self.assertAlmostEqual(scores[0].disagreement, 0.4, places=5)
        self.assertAlmostEqual(scores[0].score, 0.4, places=5)
        self.assertEqual(scores[0].d_source, 'computed')

    def test_disagreement_unavailable(self):
        p0 = np.full((16, 16), 0.5, dtype=np.float32)
        gray = np.full((16, 16), 100, dtype=np.float32)
        regions = [Region(0, 0, 16, 16, 0)]
        scores = score_regions(p0, gray, regions,
                               weights=AFRWeights(alpha=1.0, beta=1.0, gamma=0.0))
        self.assertEqual(scores[0].disagreement, 0.0)
        self.assertEqual(scores[0].d_source, 'unavailable')


class FusionTests(unittest.TestCase):
    def _p0(self):
        return np.full((64, 64), 0.3, dtype=np.float32)

    def test_no_local_returns_p0(self):
        p0 = self._p0()
        out = fuse(p0, {}, [Region(0, 0, 64, 64, 0)])
        self.assertTrue(np.array_equal(out, p0))

    def test_lam_one_replaces_region(self):
        p0 = self._p0()
        local = type('L', (), {'status': 'success',
                               'probability': np.full((32, 32), 0.9, dtype=np.float32)})()
        regions = [Region(0, 0, 64, 64, 0), Region(32, 32, 32, 32, 1)]
        out = fuse(p0, {1: local}, regions, lam=1.0)
        np.testing.assert_allclose(out[32:, 32:], 0.9)
        np.testing.assert_allclose(out[:32, :32], 0.3)

    def test_lam_zero_keeps_p0(self):
        p0 = self._p0()
        local = type('L', (), {'status': 'success',
                               'probability': np.full((32, 32), 0.9, dtype=np.float32)})()
        regions = [Region(0, 0, 64, 64, 0), Region(32, 32, 32, 32, 1)]
        out = fuse(p0, {1: local}, regions, lam=0.0)
        self.assertTrue(np.array_equal(out, p0))

    def test_lam_bounds_raise(self):
        p0 = self._p0()
        with self.assertRaises(ValueError):
            fuse(p0, {}, [], lam=1.5)
        with self.assertRaises(ValueError):
            fuse(p0, {}, [], lam=-0.1)

    def test_overlap_average_deterministic(self):
        p0 = np.zeros((64, 64), dtype=np.float32)
        l1 = type('L', (), {'status': 'success',
                            'probability': np.full((32, 32), 1.0, dtype=np.float32)})()
        l2 = type('L', (), {'status': 'success',
                            'probability': np.full((32, 32), 0.5, dtype=np.float32)})()
        regions = [Region(0, 0, 32, 32, 0), Region(16, 16, 32, 32, 1)]
        out = fuse(p0, {0: l1, 1: l2}, regions, lam=1.0)
        # 重叠区 (16..32, 16..32) = (1.0 + 0.5)/2
        np.testing.assert_allclose(out[16:32, 16:32], 0.75)
        np.testing.assert_allclose(out[:16, :16], 1.0)

    def test_size_mismatch_raises(self):
        p0 = self._p0()
        local = type('L', (), {'status': 'success',
                               'probability': np.full((10, 10), 0.9, dtype=np.float32)})()
        with self.assertRaises(ValueError):
            fuse(p0, {0: local}, [Region(0, 0, 32, 32, 0)])

    def test_threshold_binary(self):
        prob = np.array([[0.49, 0.5], [0.51, 0.0]], dtype=np.float32)
        mask = threshold_binary(prob, t=0.5)
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        np.testing.assert_array_equal(mask, [[0, 255], [255, 0]])
        with self.assertRaises(ValueError):
            threshold_binary(prob, t=0.0)
        with self.assertRaises(ValueError):
            threshold_binary(prob, t=1.0)


class ConsistencyTests(unittest.TestCase):
    def test_disagreement_mask_xor(self):
        whole = np.array([[1, 1], [0, 0]], dtype=np.uint8) * 255
        local = np.array([[1, 0], [1, 0]], dtype=np.uint8) * 255
        region = Region(0, 0, 2, 2, 0)
        d = disagreement_mask(whole, local, region)
        np.testing.assert_array_equal(d, [[0, 255], [255, 0]])

    def test_iou_empty_rules(self):
        empty = np.zeros((4, 4), dtype=np.uint8)
        nonempty = np.ones((4, 4), dtype=np.uint8) * 255
        region = Region(0, 0, 4, 4, 0)
        self.assertEqual(region_disagreement_stats(empty, empty, region)['prediction_agreement_iou'], 1.0)
        self.assertEqual(region_disagreement_stats(empty, nonempty, region)['prediction_agreement_iou'], 0.0)
        self.assertEqual(region_disagreement_stats(nonempty, empty, region)['prediction_agreement_iou'], 0.0)

    def test_iou_normal(self):
        whole = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        local = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        stats = region_disagreement_stats(whole, local, Region(0, 0, 2, 2, 0))
        # 交 1 / 并 3
        self.assertAlmostEqual(stats['prediction_agreement_iou'], 1 / 3, places=5)
        self.assertEqual(stats['disagreement_pixels'], 2)
        self.assertAlmostEqual(stats['disagreement_ratio'], 0.5)

    def test_composite_and_review(self):
        d1 = np.full((4, 4), 255, dtype=np.uint8)
        d2 = np.full((2, 2), 255, dtype=np.uint8)
        canvas = composite_disagreement(8, 8, [(0, Region(0, 0, 4, 4, 0), d1),
                                               (1, Region(4, 4, 2, 2, 1), d2)])
        self.assertIsNotNone(canvas)
        self.assertEqual(int((canvas > 0).sum()), 16 + 4)
        self.assertIsNone(composite_disagreement(8, 8, []))
        review = review_regions([
            {'region_index': 0, 'disagreement_ratio': 0.5},
            {'region_index': 1, 'disagreement_ratio': 0.01},
        ], ratio_threshold=0.05)
        self.assertEqual(review['review_region_count'], 1)
        self.assertEqual(review['review_regions'][0]['region_index'], 0)
        self.assertIn('不是错误概率', review['notice'])
        self.assertIn('不是错误概率', DISAGREEMENT_NOTICE)

    def test_composite_overlap_preserves_prior_disagreement(self):
        first = Region(0, 0, 4, 4, 0)
        second = Region(2, 0, 4, 4, 1)
        d1 = np.full((4, 4), 255, dtype=np.uint8)
        d2 = np.zeros((4, 4), dtype=np.uint8)
        canvas = composite_disagreement(
            6, 4, [(0, first, d1), (1, second, d2)])
        self.assertTrue((canvas[:, :4] == 255).all())
        self.assertTrue((canvas[:, 4:] == 0).all())


if __name__ == '__main__':
    unittest.main()
