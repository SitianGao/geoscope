"""Run the frozen v0.7 tasks against configured real RemoteAgent and RemoteSAM services."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server

TASKS = Path(__file__).with_name('v07_agent_task_set.csv')


def roi_for(case, sample):
    width, height = server.SAMPLES[sample]['size']
    if case == 'none':
        return None
    variants = {
        'valid': [100, 400, 400, 700],
        'out_of_bounds': [0, 0, width + 1, 20],
        'wrong_image_size': [0, 0, 20, 20],
        'empty': [0, 0, 0, 20],
    }
    if case not in variants:
        raise ValueError(f'unknown ROI case: {case}')
    size = [width - 1, height] if case == 'wrong_image_size' else [width, height]
    return {'xyxy': variants[case], 'source': 'drawn', 'image_size': size}


def outcome(result):
    if result['status'] in ('completed', 'needs_review'):
        return 'segment'
    if result['status'] == 'answered':
        return 'answer'
    if result['status'] == 'needs_clarification':
        return 'clarify'
    return 'reject'


def evaluate(task, mode='live'):
    count = int(task['batch_size'])
    sessions = [server.new_session(task['image_id']) for _ in range(count)]
    request = {'query': task['request'], 'mode': mode}
    if task['quality_mode']:
        request['quality_mode'] = task['quality_mode']
    if task['roi_case'] != 'none':
        request['roi'] = roi_for(task['roi_case'], task['image_id'])
    if count == 1:
        results = [server.run_task(server.get_session(sessions[0]['id']), request)]
    else:
        batch = server.run_batch({**request, 'session_ids': [s['id'] for s in sessions]})
        results = [next(r for r in server.get_session(item['session_id'])['runs'] if r['id'] == item['run_id'])
                   for item in batch['items'] if item['run_id']]
    first = results[0] if results else {}
    decision = first.get('agent_decision', {})
    plan = first.get('task', {})
    actual = outcome(first) if first else 'reject'
    routed = (None if mode == 'demo' else
              (decision.get('status') == 'tool_call' and plan.get('target') == task['expected_target']
               if task['expected_outcome'] == 'segment' else actual == task['expected_outcome']))
    return {
        'evidence_mode': 'offline_demo' if mode == 'demo' else 'real_services',
        'task_id': task['task_id'], 'request': task['request'], 'expected_outcome': task['expected_outcome'],
        'actual_outcome': actual, 'routing_correct': routed,
        'task_outcome_correct': actual == task['expected_outcome'],
        'expected_target': task['expected_target'], 'resolved_target': plan.get('target'),
        'expected_quality': task['expected_quality'],
        'resolved_quality': plan.get('effective_quality_mode'),
        'agent_status': decision.get('status', 'not_called'),
        'agent_raw_response': decision.get('raw_response'),
        'agent_raw_sha256': decision.get('raw_response_sha256'),
        'agent_tool_call': decision.get('tool_call'),
        'harness_status': first.get('status'), 'harness_reason': first.get('message'),
        'run_id': first.get('id'), 'session_ids': [s['id'] for s in sessions],
        'batch_items': [{'status': r['status'], 'run_id': r['id']} for r in results],
    }


def summarize(rows):
    called = [r for r in rows if r['agent_status'] != 'not_called']
    attempts = [r for r in called if r['agent_status'] in ('tool_call', 'rejected_tool', 'unparsed')]
    legal = [r for r in attempts if r['agent_status'] == 'tool_call']
    return {
        'total': len(rows), 'real_agent_calls': len(called),
        'evidence_mode': rows[0].get('evidence_mode') if rows else None,
        'routing_correct': (sum(r['routing_correct'] is True for r in rows)
                            if rows and rows[0].get('evidence_mode') == 'real_services' else None),
        'routing_accuracy': (sum(r['routing_correct'] is True for r in rows) / len(rows)
                             if rows and rows[0].get('evidence_mode') == 'real_services' else None),
        'task_outcome_correct': sum(r.get('task_outcome_correct', False) for r in rows),
        'legal_call_rate': len(legal) / len(attempts) if attempts else None,
        'failure_types': {status: sum(r['harness_status'] == status for r in rows)
                          for status in sorted({r['harness_status'] for r in rows})},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'test-results' / 'v07-agent-experiments')
    parser.add_argument('--offline', action='store_true', help='run Harness demo validation; never count as real Agent evidence')
    args = parser.parse_args()
    if not args.offline and (not os.environ.get('GEO_AGENT_BASE_URL') or not os.environ.get('GEO_AGENT_MODEL')):
        raise SystemExit('RemoteAgent is not configured; no real evaluation was run.')
    if not args.offline and not os.environ.get('GEO_REMOTESAM_URL'):
        raise SystemExit('RemoteSAM is not configured; no real evaluation was run.')
    with TASKS.open(encoding='utf-8', newline='') as stream:
        tasks = list(csv.DictReader(stream))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    server.DATA = args.data_dir
    if args.limit:
        tasks = tasks[:args.limit]
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'evaluation' / 'results' / f'v07_agent_{"offline_" if args.offline else ""}{timestamp}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open('x', encoding='utf-8') as stream:
        for task in tasks:
            try:
                row = evaluate(task, 'demo' if args.offline else 'live')
            except Exception as exc:
                row = {'task_id': task['task_id'], 'evidence_mode': 'offline_demo' if args.offline else 'real_services',
                       'expected_outcome': task['expected_outcome'],
                       'actual_outcome': 'error', 'routing_correct': False, 'agent_status': 'not_called',
                       'harness_status': 'error', 'harness_reason': f'{type(exc).__name__}: {exc}'}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"{task['task_id']}: {row['actual_outcome']} ({row['agent_status']})", flush=True)
    summary = summarize(rows)
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
