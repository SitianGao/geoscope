"""GeoScope-AFR 模块 C：整图与局部预测的跨尺度融合。

    P_f(x,y) = [1-λ(x,y)]·P0(x,y) + λ(x,y)·P_tile(x,y)

- 只对选中且推理成功的区域融合，其余区域保留整图概率 P0；
- 网格有 256 重叠，重叠区逐像素取各区域融合值的平均（确定性）；
- λ 全局（默认 0.5）或按区域字典覆盖，必须落在 [0,1]；
- 局部推理失败的区域自动保留 P0（由 fallback_records 记录，不在这里抛错）。

诚实性声明：融合是否优于单路径需要对照实验判断；λ 与阈值需在开发验证集
确认，未确认前元数据标记 threshold_validated: false。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .adaptive_refinement import LocalInference
from .region_scoring import Region


def _validate_lam(lam: float) -> None:
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f'λ 必须在 [0,1]，收到 {lam}')


def fuse(p0, local_results: dict, regions: Sequence[Region],
         lam: float = 0.5, lam_per_region: Optional[dict] = None) -> np.ndarray:
    """跨尺度融合，返回与 p0 同形的 float32 概率图。

    local_results: {region_index: LocalInference}；只使用 status=='success' 的项。
    """
    _validate_lam(lam)
    p0 = np.asarray(p0, dtype=np.float32)
    fused = p0.copy()
    successful = {i: li for i, li in local_results.items()
                  if li.status == 'success' and li.probability is not None}
    if not successful:
        return fused
    accum = np.zeros(p0.shape, dtype=np.float64)
    count = np.zeros(p0.shape, dtype=np.int32)
    for i, li in successful.items():
        region = regions[i]
        lam_i = (lam_per_region or {}).get(i, lam)
        _validate_lam(lam_i)
        sl = (slice(region.y, region.y + region.h), slice(region.x, region.x + region.w))
        local = np.asarray(li.probability, dtype=np.float32)
        if local.shape != (region.h, region.w):
            raise ValueError(f'区域 {i} 局部概率图尺寸 {local.shape} 与区域 '
                             f'{(region.h, region.w)} 不一致')
        accum[sl] += (1.0 - lam_i) * p0[sl] + lam_i * local
        count[sl] += 1
    return np.where(count > 0, (accum / np.maximum(count, 1)).astype(np.float32), p0)


def threshold_binary(prob, t: float = 0.5) -> np.ndarray:
    """概率图二值化 → uint8 0/255。默认阈值 0.5（与 accurate 模式 0.88 不同，
    属待验证配置，必须如实标注）。"""
    if not 0.0 < t < 1.0:
        raise ValueError(f'阈值必须在 (0,1)，收到 {t}')
    return ((np.asarray(prob, dtype=np.float32) >= t) * 255).astype(np.uint8)
