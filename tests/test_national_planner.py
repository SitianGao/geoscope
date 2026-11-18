import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import server
from geoscope_tools.backend import Backend
from geoscope_tools.statistics import ToolError
from geoscope_tools.registry import validate_call
from geoscope_tools.planner import execute_plan


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(server, 'DATA', Path(self.temp.name)); self.patch.start()
        self.sessions = patch.object(server, 'SESSIONS', {}); self.sessions.start()
        self.s = server.new_session('urban')
        self.run = server.run_task(server.get_session(self.s['id']), {'query': '提取全部建筑', 'mode': 'demo'})
        self.backend = Backend(server)
        self.args = {'image_id': self.s['id'], 'result_id': self.run['id'], 'target': 'building'}
        self.plan = {**self.args, 'task': 'region_comparison', 'scope': 'left_half', 'reference_version': self.run['id'],
                     'steps': ['get_segmentation', 'apply_scope', 'region_statistics', 'compare_versions', 'report_generation']}

    def tearDown(self):
        self.sessions.stop(); self.patch.stop(); self.temp.cleanup()

    def test_actual_saved_artifacts_and_repeat(self):
        first = execute_plan(self.plan, self.backend)
        second = execute_plan(self.plan, self.backend)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['artifacts'], second['artifacts'])
        self.assertEqual(first['artifacts']['compare_versions']['prediction_agreement_iou'], 1)
        self.assertFalse(first['artifacts']['get_segmentation']['model_execution'])

    def test_whitelist_schema_and_no_side_effects(self):
        for name, args in [('exec', self.args), ('region_statistics', {**self.args, 'path': '/etc/passwd'}),
                           ('connected_region_analysis', {**self.args, 'min_area_pixels': True})]:
            with self.subTest(name=name), self.assertRaises(ToolError): validate_call(name, args)
        with self.assertRaises(ToolError):
            execute_plan({**self.plan, 'steps': ['compare_versions', 'get_segmentation']}, self.backend)

    def test_failure_keeps_prior_outputs_and_original(self):
        self.plan['reference_version'] = '000000000000'
        path = server.DATA/self.s['id']/self.run['id']/'full_mask.png'
        original = path.read_bytes()
        r = execute_plan(self.plan, self.backend)
        self.assertEqual(r['failed_step'], 4)
        self.assertIn('region_statistics', r['artifacts'])
        self.assertEqual(path.read_bytes(), original)

    def test_image_target_and_checksum_guards(self):
        for args in ({**self.args, 'image_id': '../escape'}, {**self.args, 'target': 'aircraft'},
                     {**self.args, 'result_id': '000000000000'}):
            with self.subTest(args=args), self.assertRaises(ToolError): self.backend.execute('region_statistics', args)
        path = server.DATA/self.s['id']/self.run['id']/'full_mask.png'
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ToolError, 'checksum'): self.backend.execute('region_statistics', self.args)

    def test_unavailable_capabilities_are_truthful(self):
        with self.assertRaisesRegex(ToolError, 'AFR'): self.backend.execute('adaptive_refinement', self.args)

    def test_final_mask_corruption_and_dimensions_rejected(self):
        from PIL import Image
        path = server.DATA/self.s['id']/self.run['id']/'mask.png'
        Image.new('L', (2, 2)).save(path)
        with self.assertRaisesRegex(ToolError, 'Final mask'):
            self.backend.execute('region_statistics', self.args)

    def test_missing_file_has_clear_error(self):
        path = server.DATA/self.s['id']/self.run['id']/'full_mask.png'
        path.unlink()
        with self.assertRaisesRegex(ToolError, 'Missing'):
            self.backend.execute('region_statistics', self.args)

    def test_saved_roi_is_preserved_when_new_scope_applied(self):
        run = server.run_task(server.get_session(self.s['id']), {'query': '提取建筑', 'mode': 'demo', 'roi': {'xyxy': [0, 0, 200, 200], 'source': 'drawn', 'image_size': [800, 600]}})
        self.assertIn(run['status'], ('completed', 'needs_review'))
        result = self.backend.execute('region_statistics', {**self.args, 'result_id': run['id']})
        self.assertEqual(result['target_pixels_in_scope'], run['metrics']['pixel_area'])
        self.assertEqual(result['mask_basis'], 'saved_final_mask')

    def test_different_image_comparison_rejected(self):
        s = server.new_session('urban')
        r = server.run_task(server.get_session(s['id']), {'query': '提取建筑', 'mode': 'demo'})
        with self.assertRaises(ToolError):
            self.backend.execute('mask_compare', {**self.args, 'base_result_id': r['id']})
