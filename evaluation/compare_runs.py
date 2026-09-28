"""Compare two complete RemoteAgent evaluation CSVs without changing either run."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    'task_id', 'category', 'request', 'before_status', 'before_tool',
    'before_outcome', 'before_correct', 'before_failure_reason',
    'after_status', 'after_tool', 'after_outcome', 'after_correct',
    'after_failure_reason',
]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row['task_id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'duplicate task_id in {path}')
    return indexed


def compare(before: Path, after: Path, output: Path, aircraft_output: Path) -> None:
    old = read_rows(before)
    new = read_rows(after)
    if set(old) != set(new) or len(old) != 40:
        raise ValueError('both runs must contain the same original 40 tasks')

    rows = []
    for task_id, baseline in old.items():
        current = new[task_id]
        if (baseline['category'], baseline['request']) != (current['category'], current['request']):
            raise ValueError(f'task definition changed: {task_id}')
        rows.append({
            'task_id': task_id, 'category': baseline['category'], 'request': baseline['request'],
            'before_status': baseline['agent_status'], 'before_tool': baseline['tool_name'],
            'before_outcome': baseline['outcome'], 'before_correct': baseline['routing_correct'],
            'before_failure_reason': baseline['failure_reason'],
            'after_status': current['agent_status'], 'after_tool': current['tool_name'],
            'after_outcome': current['outcome'], 'after_correct': current['routing_correct'],
            'after_failure_reason': current['failure_reason'],
        })
    aircraft = [row for row in rows if row['category'] == 'aircraft']
    if len(aircraft) != 6:
        raise ValueError('expected six aircraft tasks')
    for path, subset in ((output, rows), (aircraft_output, aircraft)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aircraft-output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.before, args.after, args.output, args.aircraft_output)


if __name__ == '__main__':
    main()
