"""GeoScope-AFR 编排器单测：假 transport 注入，无需 GPU / 模型。

运行：PYTHONPATH=. python tests/test_afr_runner.py -v
"""
import unittest

import numpy as np
from PIL import Image

from geoscope_algorithms import (
    AFRConfig,
    AFRInitializationError,
    Budget,
    UpscaleConfig,
    config_fingerprint,
    run_adaptive,
)
from geoscope_algorithms.adaptive_refinement import image_to_b64


def _prob_png(arr):
    u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return image_to_b64(Image.fromarray(u8, 'L'))


def _pattern(size, base=0.4, amp=0.2):
    """确定性概率场：块状棋盘。"""
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    return (base + amp * ((x // 4 + y // 4) % 2)).astype(np.float32)


def _response(prob, quality_mode='fast'):
    return {'status': 'success', 'quality_mode': quality_mode, 'mask': _prob_png(prob >= 0.5),
            'probability_map': _prob_png(prob),
            'timing_ms': {'inference': 1.0, 'total': 2.0},
            'peak_vram_mb': 128.0,
            'model': {'name': 'FAKE-REMOTESAM'}, 'requested_quality_mode': quality_mode}


class RecordingTransport:
    def __init__(self, p0_fn, local_fn=None, whole_error=None, local_error=None,
                 missing_probability=False):
        self.calls = []
        self.p0_fn = p0_fn
        self.local_fn = local_fn or p0_fn
        self.whole_error = whole_error
        self.local_error = local_error
        self.missing_probability = missing_probability

    def __call__(self, request):
        import base64
        import io as _io
        from geoscope_algorithms.adaptive_refinement import decode_probability_png
        self.calls.append(request)
        encoded = request['image']
        raw = base64.b64decode(encoded, validate=True)
        img = Image.open(_io.BytesIO(raw))
        size = (img.size[1], img.size[0])
        if img.size != (1200, 900):
            if self.local_error is not None:
                raise self.local_error
            prob = self.local_fn(size)
        else:
            if self.whole_error is not None:
                raise self.whole_error
            prob = self.p0_fn(size)
        resp = _response(prob)
        if self.missing_probability:
            resp.pop('probability_map')
        return resp


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # 1200x900 → 网格 2 个 1024 区域（x: [0,176]，y: [0]）
        self.image = Image.new('RGB', (1200, 900))
        self.base_request = {'task': 'referring_seg', 'text': 'all planes'}

    def _run(self, transport, **overrides):
        config = AFRConfig(budget=Budget(max_regions=1),
                           upscale=UpscaleConfig(enabled=False))
        for key, value in overrides.items():
            setattr(config, key, value)
        return run_adaptive(self.image, self.base_request, transport, config)

    def test_budget_zero_no_local_calls(self):
        transport = RecordingTransport(_pattern)
        result = self._run(transport, budget=Budget(max_regions=0))
        self.assertEqual(result.metadata['request_count'], 1)
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(np.array_equal(result.probability, result.p0))
        self.assertIsNone(result.disagreement)
        self.assertEqual(result.metadata['selected_regions'], 0)

    def test_budget_one_selects_and_fuses(self):
        local_fn = lambda size: (np.full(size, 0.95, dtype=np.float32))
        transport = RecordingTransport(_pattern, local_fn=local_fn)
        result = self._run(transport)
        self.assertEqual(result.metadata['selected_regions'], 1)
        self.assertEqual(result.metadata['request_count'], 2)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result.metadata['local_success_count'], 1)
        self.assertEqual(result.metadata['local_service_inference_ms_total'], 1.0)
        self.assertEqual(result.metadata['local_service_ms_total'], 2.0)
        self.assertEqual(result.metadata['peak_vram_mb'], 128.0)
        # 选中区域被融合（lam=0.5 → 0.5*0.4 与 0.5*0.95 混合）
        selected = result.selection[0].region
        fused_slice = result.probability[selected.y:selected.y + selected.h,
                                         selected.x:selected.x + selected.w]
        self.assertGreater(float(fused_slice.mean()), 0.6)
        # 未选中区域：与选中区域不重叠的部分保持 P0（网格带 256 重叠，
        # 重叠部分本就会被融合修改）
        other = result.regions[1 - selected.index]
        other_slice = result.probability[other.y:other.y + other.h,
                                         other.x:other.x + other.w]
        p0_slice = result.p0[other.y:other.y + other.h, other.x:other.x + other.w]
        x0 = max(selected.x, other.x); x1 = min(selected.x + selected.w, other.x + other.w)
        y0 = max(selected.y, other.y); y1 = min(selected.y + selected.h, other.y + other.h)
        keep = np.ones(other_slice.shape, dtype=bool)
        if x0 < x1 and y0 < y1:
            keep[y0 - other.y:y1 - other.y, x0 - other.x:x1 - other.x] = False
        self.assertTrue(np.array_equal(other_slice[keep], p0_slice[keep]))
        self.assertIsNotNone(result.disagreement)
        self.assertTrue(result.review['review_region_count'] >= 0)
        self.assertEqual(result.metadata['threshold_validated'], False)

    def test_local_failure_records_fallback(self):
        transport = RecordingTransport(_pattern, local_error=RuntimeError('region boom'))
        result = self._run(transport)
        self.assertEqual(len(result.fallback_records), 1)
        self.assertEqual(result.fallback_records[0]['action'], 'whole_image_probability_kept')
        self.assertEqual(result.metadata['fallback_count'], 1)
        self.assertTrue(np.array_equal(result.probability, result.p0))
        self.assertIsNone(result.disagreement)

    def test_whole_failure_raises_initialization_error(self):
        transport = RecordingTransport(_pattern, whole_error=ValueError('p0 boom'))
        with self.assertRaises(AFRInitializationError):
            self._run(transport)

    def test_missing_probability_raises(self):
        transport = RecordingTransport(_pattern, missing_probability=True)
        with self.assertRaises(AFRInitializationError):
            self._run(transport)

    def test_lam_out_of_bounds_raises(self):
        transport = RecordingTransport(_pattern)
        with self.assertRaises(ValueError):
            self._run(transport, lam=1.5)
        with self.assertRaises(ValueError):
            self._run(transport, lam=-0.2)

    def test_threshold_bounds_raise(self):
        transport = RecordingTransport(_pattern)
        with self.assertRaises(ValueError):
            self._run(transport, threshold=0.0)
        with self.assertRaises(ValueError):
            self._run(transport, threshold=1.0)

    def test_deterministic_repeat(self):
        results = [self._run(RecordingTransport(_pattern)) for _ in range(2)]
        self.assertEqual(results[0].fused_mask.tobytes(), results[1].fused_mask.tobytes())
        self.assertTrue(np.array_equal(results[0].probability, results[1].probability))
        self.assertEqual(results[0].metadata['config_sha256'],
                         results[1].metadata['config_sha256'])

    def test_upscale_applied_and_recorded(self):
        small = Image.new('RGB', (200, 150))
        requests = []
        def transport(request):
            import base64
            import io as _io
            from geoscope_algorithms.adaptive_refinement import decode_probability_png
            requests.append(request)
            raw = base64.b64decode(request['image'], validate=True)
            img = Image.open(_io.BytesIO(raw))
            prob = _pattern((img.size[1], img.size[0]))
            return _response(prob)
        result = run_adaptive(small, self.base_request, transport,
                              AFRConfig(budget=Budget(max_regions=1)))
        local_sizes = [r['image'] for r in requests][1:]
        self.assertEqual(len(local_sizes), 1)
        import base64
        import io as _io
        from PIL import Image as _Image
        local_img = _Image.open(_io.BytesIO(base64.b64decode(requests[1]['image'], validate=True)))
        self.assertEqual(local_img.size, (400, 300))  # 200x150 → ×2
        li = list(result.local_results.values())[0]
        self.assertTrue(li.upscale_applied)

    def test_fingerprint_stable_and_sensitive(self):
        a = AFRConfig(budget=Budget(max_regions=1), lam=0.5)
        b = AFRConfig(budget=Budget(max_regions=1), lam=0.5)
        c = AFRConfig(budget=Budget(max_regions=1), lam=0.7)
        self.assertEqual(config_fingerprint(a), config_fingerprint(b))
        self.assertNotEqual(config_fingerprint(a), config_fingerprint(c))


if __name__ == '__main__':
    unittest.main()
