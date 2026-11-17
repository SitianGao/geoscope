"""Reproduce API evidence with explicitly labeled demo masks; no model claims."""
import json
import sys
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from national_server import Handler
from geoscope_tools.backend import Backend
from geoscope_tools.planner import execute_plan, PLAN_SCHEMA
from geoscope_tools.registry import REGISTRY
from PIL import Image
from geoscope_tools.statistics import region_statistics


def main():
    root = Path(__file__).resolve().parents[1]
    responses, errors = root/'contracts/sample_responses', root/'contracts/sample_errors'
    responses.mkdir(parents=True, exist_ok=True); errors.mkdir(parents=True, exist_ok=True)
    def save(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    mask = Image.new('L', (10, 10)); mask.paste(255, (0, 0, 2, 10))
    save(responses/'synthetic_roi.json', {'evidence_source': 'synthetic unit input, not model output', 'data': region_statistics(mask, 'roi', [0, 0, 5, 5])})
    save(root/'contracts/national_schema.json', {'tools': REGISTRY, 'plan': PLAN_SCHEMA})
    with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'DATA', Path(tmp)), patch.object(server, 'SESSIONS', {}):
        session = server.new_session('urban')
        run = server.run_task(server.get_session(session['id']), {'query': '提取建筑', 'mode': 'demo'})
        args = {'image_id': session['id'], 'result_id': run['id'], 'target': 'building'}
        plan = {**args, 'task': 'region_comparison', 'scope': 'left_half', 'reference_version': run['id'],
                'steps': ['get_segmentation', 'apply_scope', 'region_statistics', 'compare_versions', 'report_generation']}
        result = execute_plan(plan, Backend(server))
        save(responses/'demo_plan.json', {'evidence_source': 'actual tools on explicitly demo masks, not model inference', 'request': plan, 'response': result})
        failed = execute_plan({**plan, 'reference_version': '000000000000'}, Backend(server))
        save(errors/'demo_failed_step.json', {'evidence_source': 'actual failure on demo input', 'response': failed})
        http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        import urllib.request, urllib.error
        try:
            with patch.dict('os.environ', {'GEO_REMOTECLIP_URL': ''}):
                try: urllib.request.urlopen(f'http://127.0.0.1:{http.server_port}/api/national/retrieval/status')
                except urllib.error.HTTPError as exc:
                    with exc: save(errors/'clip_not_configured.json', json.loads(exc.read()))
        finally: http.shutdown(); http.server_close(); thread.join()
    print('Wrote actual tool responses, schemas and explicit negative examples')


if __name__ == '__main__': main()
