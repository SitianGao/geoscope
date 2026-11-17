"""GeoScope-AFR 模块 D：整图与局部预测的一致性诊断与可信复核。

    D(x,y) = |M_whole(x,y) − M_local(x,y)|   （蒙版统一编码为 0/1 后取差）

- 按选中区域统计分歧像素与占比，定位"待复核区域"；
- 两个预测之间的 IoU 命名 prediction_agreement_iou，空集规则固定：
  双空 = 1.0，仅一个为空 = 0.0；
- 分歧不等于错误概率，也不是目标数量 —— 该声明随结果强制附上。

只依赖 numpy。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .region_scoring import Region

DISAGREEMENT_NOTICE = (
    'disagreement 表示整图与局部预测的分歧像素，仅用于定位待复核区域；'
    '它不是错误概率，也不代表目标数量。')


def _as_bool(mask) -> np.ndarray:
    return np.asarray(mask) > 0


def disagreement_mask(m_whole, m_local, region: Region) -> np.ndarray:
    """区域内整图/局部二值蒙版的逐像素 XOR → uint8 0/255。"""
    d = np.logical_xor(_as_bool(m_whole), _as_bool(m_local))
    return (d * 255).astype(np.uint8)


def region_disagreement_stats(m_whole, m_local, region: Region) -> dict:
    """单区域分歧统计。prediction_agreement_iou 的空集规则：
    双空 = 1.0；仅一个为空 = 0.0。"""
    whole = _as_bool(m_whole)
    local = _as_bool(m_local)
    whole_pixels = int(whole.sum())
    local_pixels = int(local.sum())
    disagreement_pixels = int(np.logical_xor(whole, local).sum())
    area = region.w * region.h
    intersection = int(np.logical_and(whole, local).sum())
    union = int(np.logical_or(whole, local).sum())
    if union == 0:
        iou = 1.0
    elif intersection == 0 and union > 0:
        iou = 0.0
    else:
        iou = intersection / union
    return {
        'region_index': region.index,
        'xywh': [region.x, region.y, region.w, region.h],
        'disagreement_pixels': disagreement_pixels,
        'disagreement_ratio': disagreement_pixels / area if area else 0.0,
        'whole_pixels': whole_pixels,
        'local_pixels': local_pixels,
        'prediction_agreement_iou': iou,
    }


def composite_disagreement(width: int, height: int, items) -> Optional[np.ndarray]:
    """把各区域分歧图合成为全图 uint8 0/255（重叠区取逐像素最大值）。

    items: [(region_index, region, d_mask_uint8)]。无区域时返回 None。
    """
    if not items:
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    for _, region, d_mask in items:
        d_mask = np.asarray(d_mask, dtype=np.uint8)
        if d_mask.shape != (region.h, region.w):
            raise ValueError(
                f'区域 {region.index} 分歧图尺寸 {d_mask.shape} 与区域 '
                f'{(region.h, region.w)} 不一致')
        target = canvas[region.y:region.y + region.h,
                        region.x:region.x + region.w]
        np.maximum(target, d_mask, out=target)
    return canvas


def review_regions(stats_list: Sequence[dict], ratio_threshold: float = 0.05) -> dict:
    """生成待复核区域列表（按分歧占比降序）与强制诚实声明。"""
    flagged = [s for s in stats_list if s.get('disagreement_ratio', 0.0) >= ratio_threshold]
    flagged.sort(key=lambda s: (-s['disagreement_ratio'], s['region_index']))
    return {
        'ratio_threshold': ratio_threshold,
        'review_region_count': len(flagged),
        'review_regions': flagged,
        'notice': DISAGREEMENT_NOTICE,
    }
