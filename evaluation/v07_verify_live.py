"""Recompute v0.7 integration facts from stored masks and result JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'test-results' / 'v07-live-integration'


def verify_run(session_id, run_id):
    folder = DATA / session_id / run_id
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    mask_path = folder / 'mask.png'
    record = {'session_id': session_id, 'run_id': run_id, 'status': result['status']}
    if not mask_path.exists():
        assert result['status'] in ('failed', 'needs_clarification')
        record['mask_exists'] = False
        return record
    with Image.open(mask_path) as mask:
        assert mask.size == (result['metrics']['width'], result['metrics']['height'])
        area = sum(mask.histogram()[1:])
        assert area == result['metrics']['pixel_area']
        record['pixel_area_recomputed'] = area
        record['mask_size'] = list(mask.size)
        record['mask_values'] = [i for i, count in enumerate(mask.histogram()) if count]
        roi = result.get('task', {}).get('roi')
        if roi:
            x1, y1, x2, y2 = roi['xyxy']
            outside = sum(mask.histogram()[1:]) - sum(mask.crop((x1, y1, x2, y2)).histogram()[1:])
            assert outside == 0
            record['roi_outside_foreground_pixels'] = outside
    record['quality_mode'] = result.get('service_metadata', {}).get('quality_mode')
    record['mode_parameters'] = result.get('service_metadata', {}).get('mode_parameters')
    record['agent_status'] = result.get('agent_decision', {}).get('status')
    full_stats = result.get('service_metadata', {}).get('full_mask_candidate_stats')
    if full_stats:
        with Image.open(folder / 'full_mask.png') as full_mask:
            full_area = sum(full_mask.histogram()[1:])
        assert full_area == full_stats['total_area_pixels']
        record['full_mask_area_recomputed'] = full_area
        record['full_mask_candidate_count'] = full_stats['candidate_count']
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', metavar='SESSION_ID/RUN_ID')
    parser.add_argument('--batch', action='append', default=[], metavar='BATCH_ID')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.runs and not args.batch:
        parser.error('provide at least one run or batch')
    records = []
    for identifier in args.runs:
        session_id, run_id = identifier.split('/', 1)
        records.append(verify_run(session_id, run_id))
    batches = []
    for batch_id in args.batch:
        batch = json.loads((DATA / 'batches' / f'{batch_id}.json').read_text(encoding='utf-8'))
        assert batch['batch_id'] == batch_id
        assert batch['completed_count'] == sum(item['status'] in ('completed', 'needs_review') for item in batch['items'])
        assert batch['failed_count'] == sum(item['status'] in ('failed', 'needs_clarification') for item in batch['items'])
        batches.append({'batch_id': batch_id, 'completed_count': batch['completed_count'],
                        'failed_count': batch['failed_count'],
                        'items': [verify_run(item['session_id'], item['run_id']) for item in batch['items']
                                  if item['run_id']]})
    report = {'evidence_mode': 'real_services', 'runs': records, 'batches': batches}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
