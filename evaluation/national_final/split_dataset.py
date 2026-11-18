"""国赛数据集角色划分（7.2 节）：历史诊断集 / 开发验证集 / 独立测试集。

规则：
- 现有 20 图固定诊断集 → 全部划入 historical_diagnostic（与省赛连续性）；
- 提供 --extended-manifest（新增影像清单）时，按"来源场景"分组（同一场景
  绝不跨 split，避免同机场裁剪/缩放造成的数据泄漏），以固定 seed 20261118
  把场景分入 dev_validation / independent_test；
- 无新增数据时，dev/test 如实标记 pending_data_required（不伪造规模）。

用法：python evaluation/national_final/split_dataset.py [--extended-manifest PATH]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'evaluation' / 'dataset_manifest.csv'
OUT_DIR = ROOT / 'evaluation' / 'national_final' / 'results'
SPLIT_JSON = OUT_DIR / 'national_split.json'
SPLIT_CSV = OUT_DIR / 'national_manifest.csv'
SEED = 20261118


def scene_of(row: dict) -> str:
    """来源场景：优先 source_file（WHU 原 TIFF），否则 iSAID 影像前缀。"""
    source = row.get('source_file', '') or ''
    if source:
        return f'{row.get("dataset", "")}:{source}'
    image_id = row.get('image_id', '')
    prefix = image_id.rsplit('_', 1)[0] if '_' in image_id else image_id
    return f'{row.get("dataset", "")}:{prefix}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extended-manifest', help='新增影像清单 CSV（列同 dataset_manifest.csv）')
    args = parser.parse_args()

    with open(MANIFEST, encoding='utf-8', newline='') as f:
        base_rows = list(csv.DictReader(f))
    for row in base_rows:
        row['national_role'] = 'historical_diagnostic'
        row['_note'] = '省赛固定诊断集；参数选择不可标注为独立测试验证'

    extended = []
    if args.extended_manifest:
        with open(args.extended_manifest, encoding='utf-8', newline='') as f:
            extended = list(csv.DictReader(f))
    roles = {}
    if extended:
        scenes = {}
        for row in extended:
            scenes.setdefault(scene_of(row), []).append(row)
        ordered = sorted(scenes, key=lambda s: int(hashlib.sha256(
            f'{SEED}:{s}'.encode()).hexdigest(), 16))
        split_at = len(ordered) * 3 // 5  # 60% dev / 40% test（按场景数，不按图数）
        for i, scene in enumerate(ordered):
            role = 'dev_validation' if i < split_at else 'independent_test'
            for row in scenes[scene]:
                row['national_role'] = role
                row['_note'] = f'场景级划分 seed={SEED}，同一场景不跨 split'
        roles = {'dev_validation': len([r for r in extended if r['national_role'] == 'dev_validation']),
                 'independent_test': len([r for r in extended if r['national_role'] == 'independent_test'])}
    else:
        roles = {'dev_validation': None, 'independent_test': None,
                 'status': 'pending_data_required',
                 'note': '无新增影像；A3-A5 参数选择与独立验证需新数据，如实标记待补'}

    report = {
        'generated_by': 'split_dataset.py',
        'seed': SEED,
        'historical_diagnostic': len(base_rows),
        'rule': '历史诊断集=省赛 20 图固定集；新增数据按来源场景分组（同场景不跨 split）',
        'extended_roles': roles,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    all_rows = base_rows + extended
    if all_rows:
        fields = list(dict.fromkeys(
            [k for row in all_rows for k in row]))  # 保持出现顺序
        with open(SPLIT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'written: {SPLIT_JSON}')
    if all_rows:
        print(f'written: {SPLIT_CSV} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
