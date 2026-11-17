import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import server
import national_server


class NationalHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = patch.object(server, 'DATA', Path(self.temp.name)); self.data.start()
        self.sessions = patch.object(server, 'SESSIONS', {}); self.sessions.start()
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), national_server.Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True); self.thread.start()
        self.base = 'http://127.0.0.1:'+str(self.http.server_port)

    def tearDown(self):
        self.http.shutdown(); self.http.server_close(); self.thread.join()
        self.sessions.stop(); self.data.stop(); self.temp.cleanup()

    def request(self, path, payload=None):
        req = urllib.request.Request(self.base+path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            return response.status, json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw

    def test_legacy_upload_run_export_and_new_plan(self):
        _, session = self.request('/api/session', {'sample': 'urban'})
        _, run = self.request('/api/run', {'session_id': session['id'], 'mode': 'demo', 'query': '提取全部建筑'})
        args = {'image_id': session['id'], 'result_id': run['id'], 'target': 'building'}
        code, result = self.request('/api/national/tools/region-statistics', args)
        self.assertEqual(code, 200); self.assertTrue(result['success'])
        self.assertEqual(result['data']['target_pixels_in_scope'], run['metrics']['pixel_area'])
        _, comparison = self.request('/api/national/tools/mask-compare', {**args, 'base_result_id': run['id']})
        code, png = self.request(comparison['data']['diff_mask_ref'])
        self.assertEqual(code, 200); self.assertTrue(png.startswith(b'\x89PNG'))
        plan = {**args, 'task': 'region_analysis', 'scope': 'left_half', 'steps': ['get_segmentation', 'apply_scope', 'region_statistics']}
        _, response = self.request('/api/national/plans/execute', plan)
        self.assertTrue(response['success']); self.assertEqual(len(response['data']['events']), 3)
        code, archive = self.request(run['export_url'])
        self.assertEqual(code, 200); self.assertTrue(archive.startswith(b'PK'))

    def test_errors_are_structured_and_clip_is_not_faked(self):
        with patch.dict('os.environ', {'GEO_REMOTECLIP_URL': ''}):
            code, result = self.request('/api/national/retrieval/status')
            self.assertEqual((code, result['success'], result['status']), (503, False, 'not_ready'))
            self.assertFalse(result['meta']['real_execution'])
        for payload in ([], {'query': ''}, {'query': 'x', 'top_k': True}, {'query': 'x', 'path': 'arbitrary'}):
            code, result = self.request('/api/national/retrieval/search', payload)
            self.assertEqual(code, 400); self.assertEqual(result['error']['code'], 'invalid_arguments')
        code, result = self.request('/api/national/tools/shell', {})
        self.assertEqual(result['error']['code'], 'unsupported_tool')
        _, registry = self.request('/api/national/tools/registry')
        self.assertFalse(registry['data']['tools']['adaptive_refinement']['available'])
