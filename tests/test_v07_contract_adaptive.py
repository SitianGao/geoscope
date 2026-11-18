"""v07_contract adaptive 模式解析与目标门控单测。

运行：PYTHONPATH=. python tests/test_v07_contract_adaptive.py -v
"""
import unittest

import v07_contract as contract


class AdaptiveQualityTests(unittest.TestCase):
    def test_text_keywords_resolve_to_adaptive(self):
        for query in ('自适应提取飞机', '请智能分析这张影像', 'adaptive mode'):
            requested, effective, source = contract.resolve_quality(query, {})
            self.assertEqual(requested, 'adaptive', query)
            self.assertEqual(effective, 'adaptive', query)
            self.assertEqual(source, 'task_text', query)

    def test_adaptive_precedence_over_other_text_modes(self):
        # '自适应精细' 同时命中 adaptive 与 accurate：adaptive 优先，不报冲突
        requested, effective, _ = contract.resolve_quality('自适应精细提取飞机', {})
        self.assertEqual(requested, 'adaptive')
        self.assertEqual(effective, 'adaptive')
        # '快速自适应' 同理
        requested, effective, _ = contract.resolve_quality('快速自适应提取飞机', {})
        self.assertEqual(requested, 'adaptive')
        self.assertEqual(effective, 'adaptive')

    def test_existing_conflict_still_raises(self):
        with self.assertRaises(ValueError):
            contract.resolve_quality('快速但要高精度', {})

    def test_explicit_selection_adaptive(self):
        requested, effective, source = contract.resolve_quality('提取飞机', {}, 'adaptive')
        self.assertEqual((requested, effective, source), ('adaptive', 'adaptive', 'explicit_selection'))

    def test_selection_text_conflict_raises(self):
        with self.assertRaises(ValueError):
            contract.resolve_quality('自适应提取飞机', {}, 'fast')

    def test_auto_with_adaptive_text(self):
        requested, effective, _ = contract.resolve_quality('自动模式，请自适应分析', {})
        self.assertEqual(requested, 'auto')
        self.assertEqual(effective, 'adaptive')

    def test_context_adaptive_valid(self):
        requested, effective, source = contract.resolve_quality('提取飞机', {'quality_mode': 'adaptive'})
        self.assertEqual((requested, effective, source),
                         ('adaptive', 'adaptive', 'selected_version_context'))


class AdaptiveTargetGatingTests(unittest.TestCase):
    def test_aircraft_keeps_adaptive(self):
        effective, adaptation = contract.adapt_quality_for_target('aircraft', 'adaptive', 'adaptive')
        self.assertEqual(effective, 'adaptive')
        self.assertIsNone(adaptation)

    def test_building_falls_back_to_fast(self):
        effective, adaptation = contract.adapt_quality_for_target('building', 'adaptive', 'adaptive')
        self.assertEqual(effective, 'fast')
        self.assertEqual(adaptation['requested_mode'], 'adaptive')
        self.assertEqual(adaptation['effective_mode'], 'fast')
        self.assertEqual(adaptation['reason_code'], 'adaptive_not_validated_for_target')
        self.assertIn('飞机', adaptation['message'])

    def test_existing_accurate_gate_unchanged(self):
        effective, adaptation = contract.adapt_quality_for_target('building', 'accurate', 'accurate')
        self.assertEqual(effective, 'fast')
        self.assertEqual(adaptation['reason_code'], 'detail_mode_not_validated_for_target')
        effective, adaptation = contract.adapt_quality_for_target('aircraft', 'accurate', 'accurate')
        self.assertEqual(effective, 'accurate')
        self.assertIsNone(adaptation)


if __name__ == '__main__':
    unittest.main()
