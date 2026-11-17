"""Collect REAL service evidence and verify candidate selection through old API.

Requires running national_server + RemoteCLIP. Never replaces segmentation with mock.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default='http://127.0.0.1:4180')
    parser.add_argument('--query', default='airport with airplanes')
    parser.add_argument('--output', default='docs/evidence/remoteclip_real_query.json')
    args = parser.parse_args()
    def request(path, data=None):
        req = urllib.request.Request(args.platform+path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response: return json.loads(response.read())
    status = request('/api/national/retrieval/status')
    first = request('/api/national/retrieval/search', {'query': args.query, 'top_k': 10})
    second = request('/api/national/retrieval/search', {'query': args.query, 'top_k': 10})
    assert first['success'] and first['data']['real_execution'] is True
    assert first['data']['results'] == second['data']['results']
    selected = request(first['data']['results'][0]['session_url'])
    assert selected['id'] == first['data']['results'][0]['image_id']
    record = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'status': status,
              'first_query': first, 'repeat_query': second, 'selected_session': selected,
              'repeat_exact_match': True, 'real_segmentation_executed': False,
              'notice': 'Actual RemoteCLIP inference and old platform image selection verified. Segmentation requires separately configured real RemoteAgent/RemoteSAM services.'}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'evidence': str(path), 'results': first['data']['results']}, ensure_ascii=False))


if __name__ == '__main__': main()
