"""Guard competition summaries against silently dropping failed trials."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.run_end_to_end_eval import select_cases


ROOT = Path(__file__).resolve().parents[1]


class EvaluationToolsTest(unittest.TestCase):
    def test_workflow_failure_remains_in_denominator(self):
        fields = ['task_id', 'workflow', 'completed', 'total_seconds', 'operation_steps',
                  'correct_target', 'correct_scope', 'export_complete', 'evidence_path']
        rows = [
            ['M1', 'manual', 'true', '20', '5', 'true', 'true', 'true', 'evidence/m1.json'],
            ['M2', 'manual', 'false', '30', '7', 'false', 'false', 'false', 'evidence/m2.json'],
            ['A1', 'basic_agent', 'true', '12', '3', 'true', 'true', 'true', 'evidence/a1.json'],
            ['G1', 'geoscope', 'true', '8', '2', 'true', 'true', 'true', 'evidence/g1.json'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); source = folder / 'trials.csv'; output = folder / 'out' / 'summary.json'
            with source.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
            subprocess.run([sys.executable, str(ROOT / 'evaluation' / 'analyze_workflow_comparison.py'),
                            str(source), '--output', str(output)], check=True, capture_output=True)
            summary = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(summary['verified_real_rows'], 4)
            self.assertEqual(summary['by_workflow']['manual']['completion_rate'], 0.5)
            self.assertEqual(summary['by_workflow']['manual']['correct_rate'], 0.5)
            self.assertEqual(summary['by_workflow']['manual']['export_complete_rate'], 0.5)

    def test_low_iou_success_is_not_mislabeled_as_execution_failure(self):
        rows = [{'status': 'completed', 'iou': 0.1, 'dice': 0.18,
                 'task_id': 'T1', 'image_id': 'I1', 'artifact_dir': 'one'},
                {'status': 'completed', 'iou': 0.8, 'dice': 0.88,
                 'task_id': 'T2', 'image_id': 'I2', 'artifact_dir': 'two'}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cases = select_cases(rows, folder / 'artifacts', folder / 'cases')
            self.assertEqual(cases[0]['level'], 'low_quality')
            self.assertEqual(cases[0]['image_id'], 'I1')


if __name__ == '__main__':
    unittest.main()
