"""GeoScope-AFR 编排入口：包装层实现，不修改 RemoteSAM 服务。

执行链路（全部经由既有 /predict 接口，quality_mode 只发 fast）：
    整图 fast + return_probability → P0
    → 区域难度估计（模块 A）
    → 预算贪心选择（模块 B）
    → 选中区域局部精细推理（模块 B，裁剪/可选放大）
    → 跨尺度融合（模块 C）
    → 一致性诊断与待复核区域（模块 D）

- transport 由调用方注入（server 侧包 post_json；评测脚本包 service_transport；
  测试传假函数），本模块不持有任何网络实现；
- 全过程确定性（无随机数、排序带 index 次级键）；
- 任何区域级失败记入 fallback_records 并保留整图概率；
- 整图概率获取失败抛 AFRInitializationError，由调用方决定回退（按设计
  回退到既有 fast 主链路，绝不中断）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from .adaptive_refinement import (
    Budget,
    LocalInference,
    UpscaleConfig,
    build_local_request,
    decode_probability_png,
    image_to_b64,
    infer_region,
    select_regions,
    tile_grid,
)
from .consistency import (
    DISAGREEMENT_NOTICE,
    composite_disagreement,
    disagreement_mask,
    region_disagreement_stats,
    review_regions,
)
from .fusion import fuse, threshold_binary
from .region_scoring import AFRWeights, score_regions

ALGORITHM_NAME = 'geoscope_afr'
ALGORITHM_VERSION = 'v1'


class AFRInitializationError(RuntimeError):
    """整图概率图获取/校验失败：由调用方回退到既有稳定模式。"""


@dataclass
class AFRConfig:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    budget: Budget = field(default_factory=lambda: Budget(fraction=0.25))
    lam: float = 0.5
    lam_per_region: Optional[dict] = None
    threshold: float = 0.5
    tile_size: int = 1024
    overlap: int = 256
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    review_ratio_threshold: float = 0.05


@dataclass
class AFRResult:
    p0: np.ndarray
    probability: np.ndarray
    fused_mask: np.ndarray
    regions: list
    selection: list
    local_results: dict
    disagreement: Optional[np.ndarray]
    disagreement_stats: list
    review: dict
    fallback_records: list
    metadata: dict


def config_snapshot(config: AFRConfig) -> dict:
    """JSON 可序列化的配置快照（指纹与元数据共用）。"""
    return {
        'alpha': float(config.alpha),
        'beta': float(config.beta),
        'gamma': float(config.gamma),
        'budget': {'max_regions': config.budget.max_regions,
                   'fraction': config.budget.fraction},
        'lam': float(config.lam),
        'lam_per_region': config.lam_per_region,
        'threshold': float(config.threshold),
        'tile_size': int(config.tile_size),
        'overlap': int(config.overlap),
        'upscale': asdict(config.upscale),
        'review_ratio_threshold': float(config.review_ratio_threshold),
    }


def config_fingerprint(config: AFRConfig) -> str:
    """配置指纹：sha256(排序后的配置 JSON)，用于结果缓存隔离。"""
    payload = json.dumps(config_snapshot(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _p0_service_metadata(response: dict) -> dict:
    keys = ('model', 'timing_ms', 'quality', 'quality_mode', 'requested_quality_mode',
            'mode_parameters', 'original_size', 'output_size', 'peak_vram_mb')
    meta = {key: response[key] for key in keys if key in response}
    if 'candidate_stats' in response:
        meta['full_mask_candidate_stats'] = response['candidate_stats']
    meta['quality_confirmation'] = 'afr_orchestrated_fast_base'
    return meta


def _nonnegative_metric(value) -> Optional[float]:
    """把服务指标规范为有限非负浮点数；缺失/非法值返回 None。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0.0:
        return None
    return number


def run_adaptive(image, base_service_request: dict, transport: Callable[[dict], dict],
                 config: AFRConfig) -> AFRResult:
    """执行 GeoScope-AFR 主链路。失败语义见模块 docstring。"""
    if config.budget is None:
        raise ValueError('AFRConfig.budget 不能为空')
    # ---- 1. 整图 fast 推理取概率图 P0 ----
    whole_request = {**base_service_request, 'image': image_to_b64(image),
                     'quality_mode': 'fast', 'return_probability': True}
    try:
        whole_response = transport(whole_request)
    except Exception as exc:
        raise AFRInitializationError(f'整图概率推理请求失败：{type(exc).__name__}') from exc
    if (not isinstance(whole_response, dict)
            or whole_response.get('status') != 'success'
            or not whole_response.get('probability_map')):
        raise AFRInitializationError('服务未返回有效整图概率图')
    width, height = image.size
    try:
        p0 = decode_probability_png(whole_response['probability_map'], size=(width, height))
    except Exception as exc:
        raise AFRInitializationError(f'整图概率图解码失败：{type(exc).__name__}') from exc
    if p0.shape != (height, width):
        raise AFRInitializationError(f'整图概率图尺寸 {p0.shape[::-1]} 与原图不一致')
    gray = np.asarray(image.convert('L'), dtype=np.float32)

    # ---- 2. 区域难度估计 ----
    regions = tile_grid(width, height, config.tile_size, config.overlap)
    weights = AFRWeights(alpha=config.alpha, beta=config.beta, gamma=config.gamma)
    scores = score_regions(p0, gray, regions, weights, local_results=None)

    # ---- 3. 预算贪心选择 ----
    budget_used = config.budget.resolve(len(regions))
    selection = select_regions(scores, config.budget)

    # ---- 4. 局部精细推理（按区域 index 顺序，确定性） ----
    local_results: dict = {}
    fallback_records: list = []
    local_service_inference_ms_total = 0.0
    local_service_ms_total = 0.0
    peak_vram_values = []
    p0_peak_vram = _nonnegative_metric(whole_response.get('peak_vram_mb'))
    if p0_peak_vram is not None:
        peak_vram_values.append(p0_peak_vram)
    for score in sorted(selection, key=lambda s: s.region.index):
        region = score.region
        request, meta = build_local_request(image, region, base_service_request,
                                            config.upscale)
        local = infer_region(transport, region, request, (region.w, region.h),
                             config.threshold)
        local.upscale_applied = meta['upscale_applied']
        local_results[region.index] = local
        if local.status == 'failed':
            fallback_records.append({
                'region_index': region.index,
                'xywh': [region.x, region.y, region.w, region.h],
                'stage': 'local_inference',
                'error': local.error,
                'action': 'whole_image_probability_kept',
            })
        elif local.service_metadata:
            timing = local.service_metadata.get('timing_ms')
            if isinstance(timing, dict):
                inference_ms = _nonnegative_metric(timing.get('inference'))
                total_ms = _nonnegative_metric(timing.get('total'))
                if inference_ms is not None:
                    local_service_inference_ms_total += inference_ms
                if total_ms is not None:
                    local_service_ms_total += total_ms
            local_peak_vram = _nonnegative_metric(
                local.service_metadata.get('peak_vram_mb'))
            if local_peak_vram is not None:
                peak_vram_values.append(local_peak_vram)

    # ---- 5. 跨尺度融合 ----
    probability = fuse(p0, local_results, regions, config.lam, config.lam_per_region)
    fused_mask = threshold_binary(probability, config.threshold)

    # ---- 6. 一致性诊断 ----
    m_whole = threshold_binary(p0, config.threshold)
    items, stats_list = [], []
    for index, local in local_results.items():
        if local.status != 'success' or local.mask is None:
            continue
        region = regions[index]
        whole_slice = m_whole[region.y:region.y + region.h, region.x:region.x + region.w]
        d_mask = disagreement_mask(whole_slice, local.mask, region)
        items.append((index, region, d_mask))
        stats_list.append(region_disagreement_stats(whole_slice, local.mask, region))
    disagreement = composite_disagreement(width, height, items)
    review = review_regions(stats_list, config.review_ratio_threshold)

    metadata = {
        'algorithm_name': ALGORITHM_NAME,
        'algorithm_version': ALGORITHM_VERSION,
        'config': config_snapshot(config),
        'config_sha256': config_fingerprint(config),
        'image_size': [width, height],
        'grid_regions': len(regions),
        'selected_regions': len(selection),
        'budget_used': budget_used,
        'request_count': 1 + len(selection),
        'local_success_count': sum(1 for li in local_results.values() if li.status == 'success'),
        'fallback_count': len(fallback_records),
        'local_service_inference_ms_total': local_service_inference_ms_total,
        'local_service_ms_total': local_service_ms_total,
        'peak_vram_mb': max(peak_vram_values) if peak_vram_values else None,
        'threshold_validated': False,
        'disagreement_available': disagreement is not None,
        'disagreement_notice': DISAGREEMENT_NOTICE,
        'p0_service': _p0_service_metadata(whole_response),
        'honesty_notes': [
            '区域难度评分基于预测概率，不是校准置信度；',
            'disagreement 是整图与局部预测的分歧，不是错误概率；',
            '候选连通区域数量不等于准确实例数。',
        ],
    }
    return AFRResult(
        p0=p0, probability=probability, fused_mask=fused_mask,
        regions=regions, selection=selection, local_results=local_results,
        disagreement=disagreement, disagreement_stats=stats_list,
        review=review, fallback_records=fallback_records, metadata=metadata,
    )
