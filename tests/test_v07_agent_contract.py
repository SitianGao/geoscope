import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import server
from agent_bridge import decision_record


class V07AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_patch = patch.object(server, 'DATA', Path(self.temp.name))
        self.data_patch.start()
        self.session = server.SESSIONS[server.new_session('airport')['id']]
        self.roi = {'xyxy': [100, 400, 400, 700], 'source': 'drawn', 'image_size': [800, 800]}

    def tearDown(self):
        server.SESSIONS.clear()
        self.data_patch.stop()
        self.temp.cleanup()

    def run_task(self, query, **kwargs):
        return server.run_task(self.session, {'query': query, 'mode': 'demo', **kwargs})

    def test_roi_clips_and_is_saved_with_version(self):
        first = self.run_task('提取框选区域内的飞机', roi=self.roi, quality_mode='fast')
        self.assertEqual(first['status'], 'completed')
        mask = Image.open(server.DATA / self.session['id'] / first['id'] / 'mask.png')
        self.assertEqual(mask.getbbox(), (100, 429, 383, 659))
        self.assertEqual(first['metrics']['roi'], self.roi)
        second = self.run_task('改成高精度模式', parent_run_id=first['id'])
        self.assertEqual(second['task']['roi'], self.roi)
        self.assertEqual(second['task']['quality_mode'], 'accurate')
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(first['task']['quality_mode'], 'fast')
        third = self.run_task('快一点即可', parent_run_id=second['id'])
        self.assertEqual(third['task']['target'], 'aircraft')
        self.assertEqual(third['task']['roi'], self.roi)
        self.assertEqual(third['task']['quality_mode'], 'fast')

    def test_invalid_roi_and_missing_roi_preserve_result(self):
        good = self.run_task('提取飞机')
        for roi in ({'xyxy': [0, 0, 801, 20], 'source': 'drawn', 'image_size': [800, 800]},
                    {'xyxy': [0, 0, 0, 20], 'source': 'drawn', 'image_size': [800, 800]},
                    {'xyxy': [0, 0, 20, 20], 'source': 'drawn', 'image_size': [600, 800]}):
            rejected = self.run_task('提取框选区域内的飞机', roi=roi)
            self.assertEqual(rejected['status'], 'failed')
            self.assertNotIn('mask_url', rejected)
        missing = self.run_task('提取框选区域内的飞机', roi={})
        self.assertEqual(missing['status'], 'failed')
        self.assertEqual(self.session['context']['target'], 'aircraft')
        self.assertTrue((server.DATA / self.session['id'] / good['id'] / 'mask.png').exists())

    def test_quality_rejection_and_batch_intent(self):
        self.assertEqual(self.run_task('提取飞机', quality_mode='turbo')['status'], 'failed')
        self.assertEqual(self.run_task('快速且高精度提取飞机')['status'], 'failed')
        self.assertEqual(self.run_task('把同一任务应用到这些影像')['status'], 'needs_clarification')
        with patch('agent_bridge.WorkbenchAgent._run_llm') as model, \
             patch.dict(server.os.environ, {'GEO_AGENT_BASE_URL': 'http://example.invalid/v1',
                                            'GEO_AGENT_MODEL': 'TEST'}):
            ambiguous = server.run_task(self.session, {'query': '提取所有物体的轮廓', 'mode': 'live'})
        self.assertEqual(ambiguous['status'], 'needs_clarification')
        model.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'batch_id'):
            self.run_task('批量提取飞机', batch_id='untrusted')
        with self.assertRaisesRegex(ValueError, 'session_ids'):
            server.run_batch({'session_ids': [{}, self.session['id']], 'query': '批量提取飞机'})
        other = server.SESSIONS[server.new_session('airport')['id']]
        with self.assertRaisesRegex(ValueError, '共同的目标'):
            server.run_batch({'session_ids': [self.session['id'], other['id']],
                              'query': '把同一任务应用到这些影像', 'mode': 'demo'})

    def test_batch_keeps_other_images_after_one_failure(self):
        other = server.SESSIONS[server.new_session('urban')['id']]
        third = server.SESSIONS[server.new_session('airport')['id']]
        batch = server.run_batch({'session_ids': [self.session['id'], other['id'], third['id']],
                                  'query': '批量提取飞机', 'mode': 'demo'})
        self.assertEqual([item['status'] for item in batch['items']],
                         ['completed', 'failed', 'completed'])
        self.assertEqual(batch['completed_count'], 2)
        self.assertEqual(batch['failed_count'], 1)
        self.assertTrue((server.DATA / 'batches' / f"{batch['batch_id']}.json").is_file())

    def test_live_mode_requires_service_confirmation(self):
        mask = Image.new('L', (800, 800), 255)
        with patch('agent_bridge.WorkbenchAgent._run_llm', return_value='T_call(referring_expression_segmentation, "ignored.png", "all airplanes")'), \
             patch.object(server, 'post_json', return_value={'status': 'success', 'mask': server.image_b64(mask)}), \
             patch.dict(server.os.environ, {'GEO_AGENT_BASE_URL': 'http://example.invalid/v1',
                                            'GEO_AGENT_MODEL': 'TEST', 'GEO_REMOTESAM_URL': 'http://example.invalid/predict'}):
            result = server.run_task(self.session, {'query': '高精度提取飞机', 'mode': 'live'})
        self.assertEqual(result['status'], 'failed')
        self.assertIn('未确认', result['message'])
        self.assertNotIn('mask_url', result)

    def test_live_batch_calls_each_image_and_keeps_failure_isolated(self):
        sessions = [self.session] + [server.SESSIONS[server.new_session('airport')['id']] for _ in range(2)]
        mask = server.image_b64(Image.new('L', (800, 800), 255))
        replies = [{'status': 'success', 'mask': mask, 'quality_mode': 'fast'},
                   ValueError('RemoteSAM service failed'),
                   {'status': 'success', 'mask': mask, 'quality_mode': 'fast'}]
        with patch('agent_bridge.WorkbenchAgent._run_llm', return_value='T_call(referring_expression_segmentation, "ignored.png", "all airplanes")'), \
             patch.object(server, 'post_json', side_effect=replies) as service, \
             patch.dict(server.os.environ, {'GEO_AGENT_BASE_URL': 'http://example.invalid/v1',
                                            'GEO_AGENT_MODEL': 'TEST', 'GEO_REMOTESAM_URL': 'http://example.invalid/predict'}):
            batch = server.run_batch({'session_ids': [s['id'] for s in sessions],
                                      'query': '批量提取飞机', 'mode': 'live', 'quality_mode': 'fast'})
        self.assertEqual(service.call_count, 3)
        self.assertEqual([item['status'] for item in batch['items']], ['completed', 'failed', 'completed'])
        self.assertEqual(batch['completed_count'], 2)
        for item in batch['items']:
            self.assertTrue(item['run_id'])

    def test_agent_record_redacts_service_address_and_token(self):
        raw = 'call http://192.168.1.9:8000/v1 Bearer secret-token'
        decision = SimpleNamespace(raw_response=raw, status='completed', text=raw,
                                   history=[{'response': raw}], tool_call=None)
        record = decision_record(decision)
        self.assertNotIn('192.168.1.9', str(record))
        self.assertNotIn('secret-token', str(record))
        self.assertTrue(record['raw_response_sha256'])


if __name__ == '__main__':
    unittest.main()
