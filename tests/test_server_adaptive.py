"""server.py adaptive 模式集成测试：真实 HTTP 假服务 + 概率图，无 GPU。

沿用 test_workbench_integration.py 的 FakeModels 桩模式。
运行：PYTHONPATH=. python tests/test_server_adaptive.py -v
"""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import server


def _prob_png_b64(size, base=0.4, amp=0.3):
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    prob = (base + amp * ((x // 3 + y // 3) % 2)).astype(np.float32)
    u8 = np.rint(np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)
    return server.image_b64(Image.fromarray(u8, 'L'))


class FakeModels(BaseHTTPRequestHandler):
    calls = 0
    def log_message(self, *args): pass
    def reply(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path == '/v1/models': return self.reply({'data': [{'id': 'TEST-FAKE-ONLY'}]})
        if self.path == '/ready': return self.reply({'ready': True})
        self.send_error(404)
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/v1/chat/completions':
            content = data['messages'][-1]['content']
            text = ''.join(part.get('text', '') for part in content if isinstance(part, dict))
            prompt = 'all buildings' if '建筑' in text else 'all planes'
            return self.reply({'choices': [{'message': {'content':
                f'T_call(referring_expression_segmentation, "ignored.png", "{prompt}")'}}]})
        if self.path == '/predict':
            type(self).calls += 1
            image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
            prob_b64 = _prob_png_b64((image.size[1], image.size[0]))
            mask_u8 = np.rint(np.clip((np.asarray(Image.open(io.BytesIO(
                base64.b64decode(prob_b64))).convert('L'), dtype=np.float32) / 255.0) >= 0.5, 0, 1) * 255)
            mask = Image.fromarray(mask_u8.astype(np.uint8), 'L')
            return self.reply({'status': 'success', 'quality_mode': 'fast',
                               'requested_quality_mode': data.get('quality_mode'),
                               'mask': server.image_b64(mask),
                               'probability_map': prob_b64,
                               'timing_ms': {'inference': 1.0, 'total': 2.0},
                               'model': {'name': 'RemoteSAM-FAKE'}})
        self.send_error(404)


class ServerAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = ThreadingHTTPServer(('127.0.0.1', 0), FakeModels)
        cls.thread = threading.Thread(target=cls.models.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f'http://127.0.0.1:{cls.models.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.models.shutdown(); cls.models.server_close(); cls.thread.join()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_patch = patch.object(server, 'DATA', Path(self.temp.name)); self.data_patch.start()
        self.env_patch = patch.dict(os.environ, {
            'GEO_AGENT_BASE_URL': self.endpoint + '/v1', 'GEO_AGENT_MODEL': 'TEST-FAKE-ONLY',
            'GEO_REMOTESAM_URL': self.endpoint + '/predict', 'GEO_REMOTESAM_REVISION': 'fake-fixture-v1',
            'GEO_SERVICE_PROXY_MODE': 'direct', 'http_proxy': 'http://127.0.0.1:1',
            'https_proxy': 'http://127.0.0.1:1', 'no_proxy': '', 'NO_PROXY': '',
            'GEO_AFR_BUDGET_REGIONS': '1',
        }); self.env_patch.start()
        server.SESSIONS.clear(); FakeModels.calls = 0
        self.session = server.SESSIONS[server.new_session(uploaded=server.image_b64(
            Image.new('RGB', (11, 7), 'white')), name='FAKE TEST.png')['id']]
    def tearDown(self):
        server.SESSIONS.clear(); self.env_patch.stop(); self.data_patch.stop(); self.temp.cleanup()
    def run_task(self, query, **extra):
        return server.run_task(self.session, {'query': query, 'mode': 'live', **extra})

    def _run_folder(self, result):
        return server.DATA / self.session['id'] / result['id']

    def test_adaptive_run_completes_with_artifacts(self):
        result = self.run_task('自适应提取飞机')
        self.assertEqual(result['status'], 'completed', result.get('message'))
        self.assertEqual(result['afr']['status'], 'completed')
        self.assertEqual(result['afr']['selected_regions'], 1)
        self.assertEqual(result['afr']['grid_regions'], 1)
        self.assertEqual(result['afr']['request_count'], 2)
        self.assertEqual(result['task_options']['effective_quality_mode'], 'adaptive')
        self.assertEqual(result['task_options']['algorithm'], 'geoscope_afr_v1')
        self.assertIn('threshold_validated', result['afr'])
        self.assertFalse(result['afr']['threshold_validated'])
        folder = self._run_folder(result)
        self.assertTrue((folder / 'probability_map.png').is_file())
        self.assertTrue((folder / 'afr.json').is_file())
        self.assertTrue((folder / 'disagreement.png').is_file())
        self.assertTrue((folder / 'full_mask.png').is_file())
        self.assertTrue((folder / 'mask.png').is_file())
        hist = Image.open(folder / 'full_mask.png').convert('L').histogram()
        nonzero = {value for value, count in enumerate(hist) if count}
        self.assertTrue(nonzero.issubset({0, 255}))
        afr_json = json.loads((folder / 'afr.json').read_text(encoding='utf-8'))
        self.assertEqual(afr_json['algorithm'], 'geoscope_afr')
        self.assertIn('不是错误概率', afr_json['disagreement']['notice'])
        self.assertTrue(afr_json['honesty_notes'])

    def test_adaptive_cache_key_differs_from_fast(self):
        adaptive = self.run_task('自适应提取飞机')
        fast = self.run_task('快速提取飞机')
        self.assertEqual(adaptive['status'], 'completed')
        self.assertEqual(fast['status'], 'completed')
        self.assertNotEqual(adaptive['mask_cache_key'], fast['mask_cache_key'])
        self.assertNotIn('afr', fast)

    def test_building_target_gated_to_fast(self):
        result = self.run_task('自适应提取建筑')
        # 建筑目标：adapt_quality_for_target 门控回退 fast（不进入 AFR 编排）
        self.assertEqual(result['status'], 'completed', result.get('message'))
        self.assertNotIn('afr', result)
        adaptation = result['task_options'].get('quality_adaptation')
        self.assertIsNotNone(adaptation)
        self.assertEqual(adaptation['reason_code'], 'adaptive_not_validated_for_target')
        self.assertEqual(result['task_options']['effective_quality_mode'], 'fast')

    def test_adaptive_orchestration_failure_falls_back(self):
        def side_effect(url, payload):
            if payload.get('return_probability'):
                raise ValueError('afr boom')
            return {'status': 'success', 'quality_mode': 'fast',
                    'mask': server.image_b64(Image.new('L', (11, 7), 1)),
                    'timing_ms': {'inference': 1.0, 'total': 2.0},
                    'model': {'name': 'RemoteSAM-FAKE'}}
        with patch.object(server, 'post_json', side_effect=side_effect):
            result = self.run_task('自适应提取飞机')
        self.assertEqual(result['status'], 'completed', result.get('message'))
        self.assertEqual(result['afr']['status'], 'fallback')
        self.assertEqual(result['afr']['reason'], 'adaptive_orchestration_failed')
        self.assertEqual(result['task_options']['quality_adaptation']['reason_code'], 'adaptive_fallback')
        self.assertEqual(result['task_options']['effective_quality_mode'], 'fast')
        self.assertIn('mask_url', result)

    def test_fast_path_unchanged(self):
        result = self.run_task('快速提取飞机')
        self.assertEqual(result['status'], 'completed', result.get('message'))
        self.assertNotIn('afr', result)
        self.assertIn('metrics', result)


if __name__ == '__main__':
    unittest.main()
