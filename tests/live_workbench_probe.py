"""Run one real RemoteAgent -> Harness -> RemoteSAM -> Agent acceptance case."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--agent-url', required=True, help='OpenAI-compatible base ending in /v1')
    parser.add_argument('--agent-model', required=True, help='Exact id returned by /v1/models')
    parser.add_argument('--sam-url', required=True, help='RemoteSAM /predict URL')
    parser.add_argument('--sam-revision', required=True, help='Checkpoint hash or immutable version')
    parser.add_argument('--query', default='提取右侧建筑并计算面积占比')
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f'Image not found: {args.image}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        'GEO_AGENT_BASE_URL': args.agent_url,
        'GEO_AGENT_MODEL': args.agent_model,
        'GEO_REMOTESAM_URL': args.sam_url,
        'GEO_REMOTESAM_REVISION': args.sam_revision,
        'GEO_SERVICE_PROXY_MODE': 'direct',
    })

    import server

    server.DATA = args.output_dir
    server.SESSIONS.clear()
    uploaded = base64.b64encode(args.image.read_bytes()).decode('ascii')
    session_id = server.new_session(uploaded=uploaded, name=args.image.name)['id']
    session = server.SESSIONS[session_id]
    result = server.run_task(session, {'query': args.query, 'mode': 'live'})

    compact = {
        'status': result.get('status'),
        'message': result.get('message'),
        'duration_ms': result.get('duration_ms'),
        'task': result.get('task'),
        'metrics': result.get('metrics'),
        'agent_decision_status': (result.get('agent_decision') or {}).get('status'),
        'agent_feedback_status': (result.get('agent_feedback') or {}).get('status'),
        'service_metadata': result.get('service_metadata'),
        'full_mask_sha256': result.get('full_mask_sha256'),
        'run_id': result.get('id'),
        'session_id': result.get('session_id'),
        'agent_model': args.agent_model,
        'sam_revision': args.sam_revision,
    }
    (args.output_dir / 'full_chain_verification.json').write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    passed = (
        result.get('status') == 'completed'
        and bool(result.get('metrics'))
        and compact['agent_decision_status'] == 'tool_call'
        and compact['agent_feedback_status'] == 'completed'
    )
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
