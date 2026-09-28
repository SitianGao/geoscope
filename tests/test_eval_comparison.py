import csv
from pathlib import Path
import tempfile
import unittest

from evaluation.compare_runs import compare


BASELINE = Path(__file__).resolve().parents[1] / 'evaluation/results/agent_full_results.csv'


class EvaluationComparisonTests(unittest.TestCase):
    def test_complete_comparison_and_aircraft_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'all.csv'
            aircraft = Path(directory) / 'aircraft.csv'
            compare(BASELINE, BASELINE, output, aircraft)
            with output.open(newline='', encoding='utf-8') as stream:
                all_rows = list(csv.DictReader(stream))
            with aircraft.open(newline='', encoding='utf-8') as stream:
                aircraft_rows = list(csv.DictReader(stream))
            self.assertEqual(len(all_rows), 40)
            self.assertEqual([row['task_id'] for row in aircraft_rows],
                             ['E01', 'E02', 'E03', 'E04', 'E05', 'E06'])
            self.assertEqual(aircraft_rows[0]['before_outcome'], 'rejected_tool')


if __name__ == '__main__':
    unittest.main()
