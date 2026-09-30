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
from agent_bridge import AgentTurn, Clarification, decision_record, resolve_scope, segmentation_plan
from v07_contract import batch_requested, has_target_intent, resolve_quality, resolve_roi, validate_target_intent

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


def evaluate_agent_only(task):
    count = int(task['batch_size'])
    sessions = [server.new_session(task['image_id']) for _ in range(count)]
    decisions, actions = [], []
    actual, reason = 'reject', None
    try:
        query = task['request']
        try:
            validate_target_intent(query)
        except ValueError as exc:
            raise Clarification(str(exc)) from exc
        if batch_requested(query) and not has_target_intent(query):
            raise Clarification('批量任务缺少明确目标类别')
        if batch_requested(query) and count == 1:
            raise Clarification('批量任务缺少影像列表')
        for session in sessions:
            roi, _ = resolve_roi(query, {}, roi_for(task['roi_case'], task['image_id']),
                                 session['width'], session['height'])
            requested, effective, _ = resolve_quality(query, {}, task['quality_mode'] or None)
            side, scope_source = resolve_scope(query, {})
            dense = server.requires_dense_result(query)
            if dense and not has_target_intent(query):
                raise Clarification('请明确要提取的单一目标类别；当前不能提取所有物体的轮廓。')
            image_path = server.DATA / session['id'] / 'original.png'
            turn = AgentTurn(query, image_path, {}, side, scope_source,
                             task_route='dense' if dense else 'internal', roi=roi,
                             quality_mode=requested, batch_id='evaluation' if count > 1 else None)
            decisions.append(decision_record(turn.decision))
            if turn.decision.status == 'tool_call' and dense:
                plan = segmentation_plan(turn.decision, image_path, side, scope_source)
                actions.append({'tool': plan['tool_name'], 'target': plan['target'],
                                'roi': roi, 'quality_mode': requested,
                                'effective_quality_mode': effective})
                actual = 'segment'
            elif turn.decision.status == 'completed':
                actual = 'clarify' if dense else 'answer'
                if dense:
                    reason = '像素任务只返回文字，Harness 未执行分割'
            else:
                actual, reason = 'reject', f'Agent 输出状态 {turn.decision.status}；Harness 未执行工具'
            if actual != 'segment' and count > 1:
                break
    except Clarification as exc:
        actual, reason = 'clarify', str(exc)
    except ValueError as exc:
        actual, reason = 'reject', str(exc)
    return {
        'task_id': task['task_id'], 'evidence_mode': 'real_agent_decision_only',
        'request': task['request'], 'expected_outcome': task['expected_outcome'],
        'actual_outcome': actual, 'routing_correct': actual == task['expected_outcome'],
        'task_outcome_correct': actual == task['expected_outcome'],
        'expected_target': task['expected_target'],
        'resolved_target': actions[0]['target'] if actions else None,
        'expected_quality': task['expected_quality'],
        'resolved_quality': actions[0]['effective_quality_mode'] if actions else None,
        'agent_status': decisions[0]['status'] if decisions else 'not_called',
        'agent_decisions': decisions, 'harness_actions': actions,
        'harness_status': actual, 'harness_reason': reason,
        'session_ids': [s['id'] for s in sessions], 'model_calls': len(decisions), 'sam_calls': 0,
    }


def summarize(rows):
    called = [r for r in rows if r['agent_status'] != 'not_called']
    attempts = [r for r in called if r['agent_status'] in ('tool_call', 'rejected_tool', 'unparsed')]
    legal = [r for r in attempts if r['agent_status'] == 'tool_call']
    return {
        'total': len(rows), 'real_agent_calls': sum(r.get('model_calls', 1 if r['agent_status'] != 'not_called' else 0) for r in rows),
        'evidence_mode': rows[0].get('evidence_mode') if rows else None,
        'routing_correct': (sum(r['routing_correct'] is True for r in rows)
                            if rows and rows[0].get('evidence_mode') != 'offline_demo' else None),
        'routing_accuracy': (sum(r['routing_correct'] is True for r in rows) / len(rows)
                             if rows and rows[0].get('evidence_mode') != 'offline_demo' else None),
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
    parser.add_argument('--agent-only', action='store_true', help='real RemoteAgent decision and Harness validation; no RemoteSAM call')
    args = parser.parse_args()
    if args.offline and args.agent_only:
        parser.error('--offline and --agent-only are mutually exclusive')
    if not args.offline and (not os.environ.get('GEO_AGENT_BASE_URL') or not os.environ.get('GEO_AGENT_MODEL')):
        raise SystemExit('RemoteAgent is not configured; no real evaluation was run.')
    if not args.offline and not args.agent_only and not os.environ.get('GEO_REMOTESAM_URL'):
        raise SystemExit('RemoteSAM is not configured; no real evaluation was run.')
    with TASKS.open(encoding='utf-8', newline='') as stream:
        tasks = list(csv.DictReader(stream))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    server.DATA = args.data_dir
    if args.limit:
        tasks = tasks[:args.limit]
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = 'offline_' if args.offline else 'decision_' if args.agent_only else ''
    output = args.output or ROOT / 'evaluation' / 'results' / f'v07_agent_{label}{timestamp}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open('x', encoding='utf-8') as stream:
        for task in tasks:
            try:
                row = (evaluate_agent_only(task) if args.agent_only else
                       evaluate(task, 'demo' if args.offline else 'live'))
            except Exception as exc:
                evidence = 'offline_demo' if args.offline else 'real_agent_decision_only' if args.agent_only else 'real_services'
                row = {'task_id': task['task_id'], 'evidence_mode': evidence,
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
