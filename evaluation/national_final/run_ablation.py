"""国赛 A3/A4/A5 消融运行器（真实 RemoteSAM 服务，在实验室 GPU 0 执行）。

- A3 = A1 + 自适应区域选择（λ=1，局部替换）
- A4 = A3 + 跨尺度融合（λ 可配）
- A5 = A4 + 一致性诊断（分歧图 / 待复核区域落盘）

诚实性：失败行只写 failure_code/failure_detail，不伪造指标；未执行时汇总
状态为 pending_lab；summary 按实验合并写入 ablation_summary.json。

用法（在实验室机器 geoscope-national 目录内）：
python evaluation/national_final/run_ablation.py \
  --experiment A4 --manifest evaluation/dataset_manifest.csv \
  --data-root /path/to/data --base-url http://127.0.0.1:6657/predict \
  --budget-fraction 0.25 --lam 0.5 --limit 2 \
  --outdir artifacts/a4 --results-csv results/a4.csv --summary-json results/ablation_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service_transport import request_json  # noqa: E402
from geoscope_algorithms import AFRConfig, Budget, UpscaleConfig, run_adaptive  # noqa: E402

ALGORITHM = 'geoscope_afr'
VERSION = 'v1'

BASE_FIELDS = ['protocol_version', 'experiment_id', 'strategy', 'prompt', 'scale',
               'tile_size', 'overlap', 'probability_threshold', 'image_id', 'dataset',
               'width', 'height', 'image_sha256', 'ground_truth_sha256', 'status',
               'failure_code', 'failure_detail', 'request_count', 'output_width',
               'output_height', 'binary_valid', 'empty_mask',
               'prediction_foreground_pixels', 'ground_truth_foreground_pixels',
               'tp', 'fp', 'fn', 'iou', 'dice', 'precision', 'recall',
               'client_wall_ms',
               'service_inference_ms', 'service_total_ms', 'peak_vram_mb',
               'service_version', 'checkpoint_sha256']
EXTRA_FIELDS = ['budget_fraction', 'selected_regions', 'grid_regions',
                'fallback_regions', 'disagreement_mean_ratio', 'review_region_count',
                'error_pixels', 'disagreement_pixels', 'discovered_error_pixels',
                'difficult_region_error_discovery', 'disagreement_error_precision',
                'algorithm', 'algorithm_version', 'config_sha256', 'lam', 'threshold']
FIELDS = BASE_FIELDS + EXTRA_FIELDS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'iou': iou, 'dice': dice,
            'precision': precision, 'recall': recall}


def diagnostic_score(pred: np.ndarray, gt: np.ndarray,
                     disagreement: np.ndarray | None) -> dict:
    """量化 A5 分歧图对最终预测错误像素的定位能力。

    difficult_region_error_discovery = |error ∩ disagreement| / |error|。
    该值依赖真值，只是错误定位召回率，不是分割准确率或错误概率；当最终
    预测没有错误像素时记为 None，避免用 1.0 人为抬高均值。
    """
    pred_b = np.asarray(pred) > 0
    gt_b = np.asarray(gt) > 0
    if pred_b.shape != gt_b.shape:
        raise ValueError(f'预测/真值尺寸不一致：{pred_b.shape} != {gt_b.shape}')
    error = np.logical_xor(pred_b, gt_b)
    if disagreement is None:
        disagreement_b = np.zeros_like(error, dtype=bool)
    else:
        disagreement_b = np.asarray(disagreement) > 0
        if disagreement_b.shape != error.shape:
            raise ValueError(
                f'分歧图/真值尺寸不一致：{disagreement_b.shape} != {error.shape}')
    error_pixels = int(error.sum())
    disagreement_pixels = int(disagreement_b.sum())
    discovered = int(np.logical_and(error, disagreement_b).sum())
    return {
        'error_pixels': error_pixels,
        'disagreement_pixels': disagreement_pixels,
        'discovered_error_pixels': discovered,
        'difficult_region_error_discovery': (
            discovered / error_pixels if error_pixels else None),
        'disagreement_error_precision': (
            discovered / disagreement_pixels if disagreement_pixels else None),
    }


def mean_metric(rows: list[dict], key: str):
    """对成功行的可用有限数值求均值；全缺失时返回 None。"""
    values = []
    for row in rows:
        value = row.get(key)
        if value in ('', None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return round(float(np.mean(values)), 6) if values else None


def _run_one(row: dict, data_root: Path, base_url: str, config: AFRConfig,
             experiment: str, outdir: Path, timeout: int) -> dict:
    image_path = data_root / row['image_path']
    gt_path = data_root / row['ground_truth_path']
    result = {field: '' for field in FIELDS}
    result.update({
        'protocol_version': f'national-{experiment.lower()}-v1',
        'experiment_id': f'{experiment.lower()}_adaptive_focus_refinement',
        'strategy': 'adaptive_focus_refinement',
        'prompt': 'all planes',
        'scale': 1.0,
        'tile_size': config.tile_size,
        'overlap': config.overlap,
        'probability_threshold': config.threshold,
        'image_id': row['image_id'],
        'dataset': row.get('dataset', ''),
        'width': row.get('width', ''),
        'height': row.get('height', ''),
        'image_sha256': sha256_file(image_path),
        'ground_truth_sha256': sha256_file(gt_path),
        'budget_fraction': config.budget.fraction,
        'lam': config.lam,
        'threshold': config.threshold,
        'algorithm': ALGORITHM,
        'algorithm_version': VERSION,
        'config_sha256': '',
    })
    image = Image.open(image_path).convert('RGB')
    gt = np.asarray(Image.open(gt_path).convert('L'))
    start = time.perf_counter()
    try:
        def transport(payload):
            return request_json(base_url, payload, timeout=timeout)
        afr = run_adaptive(image, {'task': 'referring_seg', 'text': 'all planes'},
                           transport, config)
    except Exception as exc:
        result.update(status='failed',
                      failure_code=type(exc).__name__,
                      failure_detail=str(exc)[:300],
                      client_wall_ms=round((time.perf_counter() - start) * 1000, 2))
        return result
    result['client_wall_ms'] = round((time.perf_counter() - start) * 1000, 2)
    pred = afr.fused_mask
    metrics = score(pred, gt)
    result.update({
        'status': 'success',
        'request_count': afr.metadata['request_count'],
        'output_width': image.width, 'output_height': image.height,
        'binary_valid': bool(set(np.unique(pred).tolist()) <= {0, 255}),
        'empty_mask': bool((pred == 0).all()),
        'prediction_foreground_pixels': int((pred > 0).sum()),
        'ground_truth_foreground_pixels': int(row.get('ground_truth_foreground_pixels', '') or int((gt > 0).sum())),
        'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
        'iou': round(metrics['iou'], 6), 'dice': round(metrics['dice'], 6),
        'precision': (round(metrics['precision'], 6)
                      if metrics['precision'] is not None else ''),
        'recall': (round(metrics['recall'], 6)
                   if metrics['recall'] is not None else ''),
        'service_inference_ms': round(afr.metadata['local_service_inference_ms_total']
                                      + float(afr.metadata['p0_service'].get('timing_ms', {}).get('inference', 0.0)), 2),
        'service_total_ms': round(afr.metadata['local_service_ms_total']
                                  + float(afr.metadata['p0_service'].get('timing_ms', {}).get('total', 0.0)), 2),
        'peak_vram_mb': (round(float(afr.metadata['peak_vram_mb']), 2)
                         if afr.metadata.get('peak_vram_mb') is not None else ''),
        'service_version': (afr.metadata['p0_service'].get('service_version')
                            or afr.metadata['p0_service'].get('model', {}).get(
                                'service_version', '')),
        'checkpoint_sha256': afr.metadata['p0_service'].get('model', {}).get('checkpoint_sha256', '')
        if isinstance(afr.metadata['p0_service'].get('model'), dict) else '',
        'selected_regions': afr.metadata['selected_regions'],
        'grid_regions': afr.metadata['grid_regions'],
        'fallback_regions': afr.metadata['fallback_count'],
        'disagreement_mean_ratio': round(float(np.mean(
            [s['disagreement_ratio'] for s in afr.disagreement_stats])) if afr.disagreement_stats else 0.0, 6),
        'review_region_count': afr.review['review_region_count'],
        'config_sha256': afr.metadata['config_sha256'],
    })
    if experiment == 'A5':
        diagnostic = diagnostic_score(pred, gt, afr.disagreement)
        result.update({
            key: ((round(value, 6) if isinstance(value, float) else value)
                  if value is not None else '')
            for key, value in diagnostic.items()
        })
    image_out = outdir / row['image_id']
    image_out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred, 'L').save(image_out / 'mask.png')
    prob_u8 = np.rint(np.clip(afr.probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(prob_u8, 'L').save(image_out / 'probability_map.png')
    if experiment == 'A5' and afr.disagreement is not None:
        Image.fromarray(afr.disagreement, 'L').save(image_out / 'disagreement.png')
    (image_out / 'afr.json').write_text(json.dumps(
        {**afr.metadata, 'review': afr.review,
         'fallback_records': afr.fallback_records}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return result


def run_one(row: dict, data_root: Path, base_url: str, config: AFRConfig,
            experiment: str, outdir: Path, timeout: int) -> dict:
    """运行单图并隔离所有行级失败，确保坏样本不会终止整批实验。"""
    started = time.perf_counter()
    try:
        return _run_one(row, data_root, base_url, config, experiment, outdir,
                        timeout)
    except Exception as exc:
        result = {field: '' for field in FIELDS}
        result.update({
            'protocol_version': f'national-{experiment.lower()}-v1',
            'experiment_id': f'{experiment.lower()}_adaptive_focus_refinement',
            'strategy': 'adaptive_focus_refinement',
            'prompt': 'all planes',
            'scale': 1.0,
            'tile_size': config.tile_size,
            'overlap': config.overlap,
            'probability_threshold': config.threshold,
            'image_id': row.get('image_id', ''),
            'dataset': row.get('dataset', ''),
            'width': row.get('width', ''),
            'height': row.get('height', ''),
            'status': 'failed',
            'failure_code': type(exc).__name__,
            'failure_detail': str(exc)[:300],
            'client_wall_ms': round((time.perf_counter() - started) * 1000, 2),
            'budget_fraction': config.budget.fraction,
            'lam': config.lam,
            'threshold': config.threshold,
            'algorithm': ALGORITHM,
            'algorithm_version': VERSION,
        })
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=('A3', 'A4', 'A5'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--budget-fraction', type=float, default=0.25)
    parser.add_argument('--lam', type=float, default=1.0, help='A3 用 1.0（局部替换）；A4/A5 用开发验证集选定值')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-upscale', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='0=全部')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--results-csv', required=True)
    parser.add_argument('--summary-json', required=True)
    args = parser.parse_args()
    if args.experiment == 'A3' and not np.isclose(args.lam, 1.0):
        parser.error('A3 定义为局部直接替换，必须使用 --lam 1.0')

    with open(args.manifest, encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get('project_target') or r.get('class_name')) in ('aircraft', 'plane')]
    if not rows:
        print('manifest 中没有飞机影像行，终止（不生成伪造结果）')
        return 2
    if args.limit:
        rows = rows[:args.limit]
    data_root = Path(args.data_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    config = AFRConfig(
        budget=Budget(fraction=args.budget_fraction),
        lam=args.lam,
        threshold=args.threshold,
        upscale=UpscaleConfig(enabled=not args.no_upscale),
    )
    results = [run_one(row, data_root, args.base_url, config, args.experiment,
                       outdir, args.timeout) for row in rows]

    out_csv = Path(args.results_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    succeeded = [r for r in results if r['status'] == 'success']
    summary = {
        'experiment': args.experiment,
        'config': {
            'budget_fraction': args.budget_fraction, 'lam': args.lam,
            'threshold': args.threshold, 'upscale': config.upscale.enabled,
            'tile_size': config.tile_size, 'overlap': config.overlap,
            'algorithm': ALGORITHM, 'algorithm_version': VERSION,
            'config_sha256': succeeded[0]['config_sha256'] if succeeded else None,
        },
        'rows_total': len(results),
        'rows_succeeded': len(succeeded),
        'status': ('completed' if len(succeeded) == len(results)
                   else 'partial' if succeeded else 'failed_all'),
        'mean_iou': mean_metric(succeeded, 'iou'),
        'mean_dice': mean_metric(succeeded, 'dice'),
        'mean_precision': mean_metric(succeeded, 'precision'),
        'mean_recall': mean_metric(succeeded, 'recall'),
        'mean_request_count': mean_metric(succeeded, 'request_count'),
        'mean_client_wall_ms': mean_metric(succeeded, 'client_wall_ms'),
        'mean_service_inference_ms': mean_metric(succeeded, 'service_inference_ms'),
        'mean_peak_vram_mb': mean_metric(succeeded, 'peak_vram_mb'),
        'mean_difficult_region_error_discovery': mean_metric(
            succeeded, 'difficult_region_error_discovery'),
        'mean_disagreement_error_precision': mean_metric(
            succeeded, 'disagreement_error_precision'),
        'threshold_validated': False,
        'honesty_notes': [
            '本实验为探索性验证；参数选择必须在开发验证集完成，历史诊断集结果不可标注为独立测试验证',
            'disagreement 是整图与局部预测的分歧，不是错误概率；候选连通区域不是实例计数',
            'difficult_region_error_discovery 是分歧像素覆盖最终预测错误像素的召回率，依赖真值，不是分割准确率',
            'threshold/λ 未在开发验证集确认前 threshold_validated=false',
        ],
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    summary_path = Path(args.summary_json)
    merged = {}
    if summary_path.is_file():
        merged = json.loads(summary_path.read_text(encoding='utf-8'))
    if isinstance(merged, dict):
        merged[f'{args.experiment}_budget{args.budget_fraction}_lam{args.lam}'] = summary
    else:
        merged = {f'{args.experiment}_budget{args.budget_fraction}_lam{args.lam}': summary}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'results: {out_csv}')
    print(f'summary: {summary_path}')
    return 0 if succeeded else 1


if __name__ == '__main__':
    raise SystemExit(main())
