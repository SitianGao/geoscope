"""Run a three-image real-service batch and save a compact, reproducible record."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server


def main():
    server.DATA = ROOT / 'test-results' / 'v07-live-integration'
    server.DATA.mkdir(parents=True, exist_ok=True)
    original = Image.open(server.WEB / 'assets' / 'airport.jpg').convert('RGB')
    variants = [original, original.crop((0, 0, 700, 700)), original.resize((640, 640))]
    sessions = []
    hashes = []
    for index, image in enumerate(variants, 1):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        raw = buffer.getvalue()
        hashes.append(hashlib.sha256(raw).hexdigest())
        sessions.append(server.new_session(uploaded=base64.b64encode(raw).decode(),
                                           name=f'airport-variant-{index}'))
    batch = server.run_batch({'session_ids': [s['id'] for s in sessions],
                              'query': '批量快速提取飞机', 'quality_mode': 'fast', 'mode': 'live'})
    rows = []
    for index, item in enumerate(batch['items']):
        run = next(r for r in server.get_session(item['session_id'])['runs']
                   if r['id'] == item['run_id']) if item['run_id'] else {}
        rows.append({'item_id': item['item_id'], 'session_id': item['session_id'],
                     'image_sha256': hashes[index], 'run_id': item['run_id'],
                     'status': item['status'], 'failure_reason': item['failure_reason'],
                     'pixel_area': (item['metrics'] or {}).get('pixel_area'),
                     'agent_status': run.get('agent_decision', {}).get('status'),
                     'quality_mode': run.get('service_metadata', {}).get('quality_mode'),
                     'mode_parameters': run.get('service_metadata', {}).get('mode_parameters'),
                     'timing_ms': run.get('service_metadata', {}).get('timing_ms')})
    record = {'evidence_mode': 'real_services', 'batch_id': batch['batch_id'],
              'completed_count': batch['completed_count'], 'failed_count': batch['failed_count'],
              'items': rows}
    output = server.DATA / f"{batch['batch_id']}-verification.json"
    server.save_json(output, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
