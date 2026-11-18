"""国赛 P0：离线复算 A0/A1/A2 与建筑基线（仅读已提交 CSV，不依赖服务/GPU）。

复算规则：
- A0   飞机：remotesam_full_results.csv 中 class_name=plane 的均值
- A1   飞机：prompt_sweep_results.csv 中 experiment_id=prompt_all_planes 的均值
- A2   飞机：tile_1024_high_threshold_results.csv 中
          experiment_id=tile_1024_overlap_256_p0_88 的均值
- 建筑：remotesam_full_results.csv 中 class_name=building 的均值
- 每行用 tp/fp/fn 重算 iou/dice 与 CSV 记录值核对（公式一致性检查）
- 与 MD 记录值（0.1585 / 0.2460 / 0.3056 / 0.3522）比对并记录差异
- 记录每个源文件的 sha256，保证可复算溯源

用法：python evaluation/national_final/recompute_baselines_a0a2.py [--write]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'evaluation' / 'results'
OUT_DIR = ROOT / 'evaluation' / 'national_final' / 'results'
OUT_FILE = OUT_DIR / 'baselines_a0_a2.json'

EXPECTED = {
    'A0_aircraft_iou': 0.1585, 'A0_aircraft_dice': 0.2281,
    'A1_aircraft_iou': 0.2460, 'A1_aircraft_dice': 0.3292,
    'A2_aircraft_iou': 0.3056, 'A2_aircraft_dice': 0.4133,
    'building_iou': 0.3522, 'building_dice': 0.5115,
}
TOLERANCE = 1e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [float(v) for v in values if v not in (None, '')]
    return sum(values) / len(values) if values else None


def col(rows, key):
    return [r[key] for r in rows if r.get(key) not in (None, '')]


def verify_tp_fp_fn(rows, label, errors):
    """用 tp/fp/fn 重算 iou/dice 并与 CSV 记录值核对。"""
    for r in rows:
        tp, fp, fn = float(r['tp']), float(r['fp']), float(r['fn'])
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if abs(iou - float(r['iou'])) > 1e-6 or abs(dice - float(r['dice'])) > 1e-6:
            errors.append(f'{label} {r.get("image_id")} tp/fp/fn 与 iou/dice 不一致')
    return not errors


def group_mean(rows, label, errors):
    verify_tp_fp_fn(rows, label, errors)
    return {'rows': len(rows),
            'iou': round(mean(col(rows, 'iou')), 6),
            'dice': round(mean(col(rows, 'dice')), 6),
            'precision': round(mean([float(r['tp']) / (float(r['tp']) + float(r['fp']))
                                     for r in rows if float(r['tp']) + float(r['fp']) > 0]), 6),
            'recall': round(mean([float(r['tp']) / (float(r['tp']) + float(r['fn']))
                                  for r in rows if float(r['tp']) + float(r['fn']) > 0]), 6),
            'mean_service_inference_ms': round(mean(col(rows, 'service_inference_ms')) or 0.0, 2),
            'mean_client_wall_ms': round(mean(col(rows, 'client_wall_ms')) or 0.0, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='写入 results/baselines_a0_a2.json')
    args = parser.parse_args()

    errors = []
    sources = {}
    report = {'generated_by': 'recompute_baselines_a0a2.py',
              'scope': '离线复算（仅已提交结果文件），不需要模型服务',
              'tolerance': TOLERANCE, 'baselines': {}, 'sources': {}, 'errors': errors}

    def load(rel_path):
        path = RESULTS / rel_path
        rows = read_csv(path)
        sources[rel_path] = {'path': str(path), 'sha256': sha256(path), 'rows': len(rows)}
        return rows

    full = load('remotesam_full_results.csv')
    prompt = load('aircraft_v2/prompt_sweep_results.csv')
    tile = load('aircraft_v2/tile_1024_high_threshold_results.csv')

    groups = {
        'A0_aircraft': [r for r in full if r['class_name'] == 'plane'],
        'A1_aircraft': [r for r in prompt if r['experiment_id'] == 'prompt_all_planes'],
        'A2_aircraft': [r for r in tile if r['experiment_id'] == 'tile_1024_overlap_256_p0_88'],
        'building': [r for r in full if r['class_name'] == 'building'],
    }
    for key, rows in groups.items():
        if not rows:
            errors.append(f'{key} 源数据为空')
            continue
        report['baselines'][key] = group_mean(rows, key, errors)
        report['baselines'][key]['_recorded_iou'] = EXPECTED.get(f'{key}_iou')
        recorded = EXPECTED.get(f'{key}_iou')
        if recorded is not None:
            delta = abs(report['baselines'][key]['iou'] - recorded)
            report['baselines'][key]['recorded_delta_abs'] = round(delta, 6)
            if delta > TOLERANCE:
                errors.append(f'{key} 复算 IoU 与 MD 记录值偏差 {delta:.6f} 超过容差')

    report['status'] = 'verified_offline' if not errors else 'mismatch_detected'
    report['sources'] = sources
    report['baselines']['building']['note'] = '建筑基线（整图 all buildings），用于 A0 对照背景'

    print(json.dumps({k: v for k, v in report.items() if k != 'sources'},
                     ensure_ascii=False, indent=2))
    if errors:
        print('ERRORS:', *errors, sep='\n  - ')
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'written: {OUT_FILE}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
