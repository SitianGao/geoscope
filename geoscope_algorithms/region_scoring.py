"""GeoScope-AFR 模块 A：区域难度估计。

对整图快速推理得到的概率图 P0 划分候选区域，估计每个区域的分析难度：

    U_i = α·H̄_i + β·D_i + γ·C_i

- H̄_i：区域平均预测熵（二分类熵，见 binary_entropy）；
- D_i：整图与局部预测的概率分歧（局部推理返回后计算，见 score_regions 的
  local_results 参数；不可用时该项为 0 并在结果中标注来源）；
- C_i：归一化局部图像复杂度（梯度模的边缘密度，固定归一化常数，与图像无关）；
- α、β、γ：权重，默认 0.5 / 0.3 / 0.2，需在开发验证集上确定。

诚实性声明：
- 预测概率不等于校准后的真实置信度；难度评分只作为复核线索，不宣称是分割正确率。
- P0 经服务端 8-bit 量化传输（round(p*255)/255），熵在量化值上计算。
- 区域概率全零时只打 empty_mask 标记，不注入任何分数。

本模块只依赖 numpy，兼容 numpy 1.x / 2.x。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

EPS_DEFAULT = 1e-6


@dataclass(frozen=True)
class Region:
    """原图像素坐标系下的矩形区域（左上包含、右下不包含）。"""
    x: int
    y: int
    w: int
    h: int
    index: int


@dataclass(frozen=True)
class AFRWeights:
    """难度评分权重 U_i = α·H̄_i + β·D_i + γ·C_i。"""
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2


@dataclass(frozen=True)
class RegionScore:
    region: Region
    entropy: float
    complexity: float
    disagreement: float
    score: float
    empty_mask: bool
    d_source: str  # 'unavailable'（尚无局部结果）或 'computed'


def binary_entropy(p, eps: float = EPS_DEFAULT) -> np.ndarray:
    """二分类熵 H(p) = -p·log(p+ε) - (1-p)·log(1-p+ε)，输入裁剪到 [0,1]。

    解析上 H(0)=H(1)=-log(1+ε)≈-ε（微小负值，ε 正则化的已知边界效应），
    这里裁剪到 [0, ∞)，使 p∈{0,1} 时熵精确为 0。
    返回 float32 数组，形状与输入一致。
    """
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 0.0, 1.0)
    h = -p * np.log(p + eps) - (1.0 - p) * np.log(1.0 - p + eps)
    return np.clip(h, 0.0, None).astype(np.float32)


def edge_density(gray) -> np.ndarray:
    """归一化边缘密度：梯度模 / (255·√2)，逐像素 ∈ [0,1]。

    用 np.gradient 计算中心差分梯度，固定归一化常数（与单张图像无关，
    保证跨影像可比、不依赖逐图最大值）。纯灰度输入（float32，0-255）。
    """
    g = np.asarray(gray, dtype=np.float32)
    gy, gx = np.gradient(g)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return (magnitude / (255.0 * np.sqrt(2.0))).astype(np.float32)


def _slice(region: Region):
    return slice(region.y, region.y + region.h), slice(region.x, region.x + region.w)


def mean_entropy(prob: np.ndarray, region: Region) -> float:
    return float(binary_entropy(prob[_slice(region)]).mean())


def mean_complexity(gray: np.ndarray, region: Region) -> float:
    return float(edge_density(gray[_slice(region)]).mean())


def score_regions(
    prob: np.ndarray,
    gray: np.ndarray,
    regions: Sequence[Region],
    weights: Optional[AFRWeights] = None,
    local_results: Optional[dict] = None,
    ) -> list:
    """对每个候选区域计算难度评分 U_i = α·H̄_i + β·D_i + γ·C_i。

    local_results: {region_index: 带 probability 属性的对象}（LocalInference），
    提供时计算 D_i = 该区域内 |P0 - P_local| 的均值；缺省时 D_i = 0 且
    d_source 标为 'unavailable'。
    prob 为整图概率图（H×W），gray 为整图灰度图（H×W）。
    """
    weights = weights or AFRWeights()
    local_results = local_results or {}
    prob = np.asarray(prob, dtype=np.float32)
    scores = []
    for region in regions:
        sl = _slice(region)
        entropy = float(binary_entropy(prob[sl]).mean())
        complexity = float(edge_density(gray[sl]).mean())
        empty_mask = not bool((prob[sl] > 0).any())
        disagreement = 0.0
        d_source = 'unavailable'
        local = local_results.get(region.index)
        if local is not None and getattr(local, 'probability', None) is not None:
            local_prob = np.asarray(local.probability, dtype=np.float32)
            if local_prob.shape == (region.h, region.w):
                disagreement = float(np.abs(prob[sl] - local_prob).mean())
                d_source = 'computed'
        score = weights.alpha * entropy + weights.beta * disagreement + weights.gamma * complexity
        scores.append(RegionScore(
            region=region, entropy=entropy, complexity=complexity,
            disagreement=disagreement, score=score,
            empty_mask=empty_mask, d_source=d_source,
        ))
    return scores
