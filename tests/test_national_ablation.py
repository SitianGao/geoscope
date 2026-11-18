"""国赛消融指标单测：纯 numpy，无需模型服务或数据集。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.national_final.run_ablation import diagnostic_score, mean_metric, run_one
from geoscope_algorithms import AFRConfig, Budget


class DiagnosticScoreTests(unittest.TestCase):
    def test_error_discovery_and_precision(self):
        pred = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        gt = np.array([[0, 0], [255, 0]], dtype=np.uint8)
        disagreement = np.array([[255, 255], [0, 0]], dtype=np.uint8)

        result = diagnostic_score(pred, gt, disagreement)

        self.assertEqual(result['error_pixels'], 2)
        self.assertEqual(result['disagreement_pixels'], 2)
        self.assertEqual(result['discovered_error_pixels'], 1)
        self.assertEqual(result['difficult_region_error_discovery'], 0.5)
        self.assertEqual(result['disagreement_error_precision'], 0.5)

    def test_no_disagreement_discovers_no_existing_error(self):
        pred = np.array([[255]], dtype=np.uint8)
        gt = np.array([[0]], dtype=np.uint8)
        result = diagnostic_score(pred, gt, None)
        self.assertEqual(result['difficult_region_error_discovery'], 0.0)
        self.assertIsNone(result['disagreement_error_precision'])

    def test_no_prediction_error_is_not_scored_as_perfect(self):
        mask = np.zeros((2, 2), dtype=np.uint8)
        result = diagnostic_score(mask, mask, mask)
        self.assertIsNone(result['difficult_region_error_discovery'])

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic_score(np.zeros((2, 2)), np.zeros((1, 2)), None)


class MeanMetricTests(unittest.TestCase):
    def test_missing_and_non_finite_values_are_ignored(self):
        rows = [{'x': 1}, {'x': ''}, {'x': None}, {'x': float('nan')}, {'x': 3}]
        self.assertEqual(mean_metric(rows, 'x'), 2.0)
        self.assertIsNone(mean_metric(rows, 'missing'))


class RowIsolationTests(unittest.TestCase):
    def test_missing_input_becomes_failure_row(self):
        row = {
            'image_id': 'missing-image',
            'dataset': 'test',
            'image_path': 'missing.png',
            'ground_truth_path': 'missing-mask.png',
            'width': '8',
            'height': '8',
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_one(
                row, root, 'http://127.0.0.1:1/predict',
                AFRConfig(budget=Budget(max_regions=1)),
                'A3', root / 'artifacts', 1)

        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['protocol_version'], 'national-a3-v1')
        self.assertEqual(result['failure_code'], 'FileNotFoundError')
        self.assertEqual(result['image_id'], 'missing-image')


if __name__ == '__main__':
    unittest.main()
