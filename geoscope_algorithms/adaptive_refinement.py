"""GeoScope-AFR 模块 B：预算约束下的自适应局部精细推理。

候选区域按难度评分贪心选择，在推理预算内只对选中区域调用既有 RemoteSAM
/predict 接口（整图 fast 模式 + return_probability=True）做局部精细推理。

- 区域网格与既有 accurate 模式完全一致（1024 切片 / 256 重叠、末块贴边，
  与服务端 _tile_positions 语义逐字节对齐）；
- 局部推理允许对小区域做放大（upscale），让模型 896px 内部重采样对小目标
  损失更小；放大信息如实记录；
- 每次调用成本记为 1（请求次数是诚实的成本单位），墙钟时间另行记录；
- 单个区域失败只记录 fallback 并保留整图概率，绝不抛出、绝不中断主链路。

只依赖 numpy + Pillow。
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .region_scoring import Region, RegionScore


@dataclass(frozen=True)
class Budget:
    """推理预算：max_regions 与 fraction（占网格总数比例）二选一。"""
    max_regions: Optional[int] = None
    fraction: Optional[float] = None

    def resolve(self, n_regions: int) -> int:
        if self.max_regions is not None and self.fraction is not None:
            raise ValueError('预算只能定义 max_regions 或 fraction 之一')
        if self.max_regions is not None:
            if self.max_regions < 0:
                raise ValueError('max_regions 不能为负')
            return max(0, min(int(self.max_regions), n_regions))
        if self.fraction is not None:
            if not 0.0 <= self.fraction <= 1.0:
                raise ValueError('fraction 必须在 [0,1]')
            return max(0, min(int(np.floor(self.fraction * n_regions)), n_regions))
        raise ValueError('预算未定义（max_regions / fraction 至少给一个）')


@dataclass(frozen=True)
class UpscaleConfig:
    """小区域放大推理配置（服务模型内部固定重采样到 896px）。"""
    enabled: bool = True
    min_size: int = 512
    scale: float = 2.0


@dataclass
class LocalInference:
    """单个区域的局部推理结果。失败时 status='failed'，probability/mask 为 None。"""
    region_index: int
    status: str  # 'success' | 'failed'
    probability: Optional[np.ndarray] = None  # float32 [0,1]，区域分辨率
    mask: Optional[np.ndarray] = None  # uint8 0/255，区域分辨率
    cost: float = 1.0
    error: Optional[str] = None
    service_metadata: Optional[dict] = None
    upscale_applied: bool = False


def _positions(length: int, tile_size: int, overlap: int):
    """与服务端 _tile_positions 逐字节对齐的切片起点列表。"""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_grid(width: int, height: int, tile_size: int = 1024, overlap: int = 256):
    """候选区域网格。每个区域 tile_size×tile_size（图像小于 tile 时为整图）。"""
    regions = []
    index = 0
    for y in _positions(height, tile_size, overlap):
        for x in _positions(width, tile_size, overlap):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            regions.append(Region(x=x, y=y, w=w, h=h, index=index))
            index += 1
    return regions


def select_regions(scores, budget: Budget):
    """贪心选择：按 (score 降序, index 升序) 取预算内区域，确定性无随机。"""
    n = budget.resolve(len(scores))
    ordered = sorted(scores, key=lambda s: (-s.score, s.region.index))
    return ordered[:n]


def image_to_b64(image: Image.Image) -> str:
    """PIL 图像 → PNG base64 字符串（与服务端请求格式一致）。"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def decode_probability_png(encoded: str, size: Optional[tuple] = None) -> np.ndarray:
    """解码服务端概率图（8-bit PNG，round(p*255)）→ float32 [0,1]。

    size=(w,h) 提供且尺寸不符时双线性重采样到目标尺寸（局部裁剪放大的还原路径）。
    """
    raw = base64.b64decode(encoded, validate=True)
    img = Image.open(io.BytesIO(raw)).convert('L')
    if size is not None and tuple(size) != img.size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def encode_probability_png(prob) -> str:
    """float32 [0,1] 概率图 → 8-bit PNG base64（与服务端量化语义一致）。"""
    arr = np.rint(np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, 'L')
    return image_to_b64(img)


def build_local_request(image, region, base_service_request, upscale: UpscaleConfig):
    """裁剪区域（小区域可选放大）并构造既有 /predict 请求。

    返回 (request_dict, meta)；meta 记录 upscale_applied 与请求尺寸。
    """
    crop = image.crop((region.x, region.y, region.x + region.w, region.y + region.h))
    upscale_applied = False
    if upscale.enabled and max(region.w, region.h) < upscale.min_size:
        new_size = (max(1, int(round(region.w * upscale.scale))),
                    max(1, int(round(region.h * upscale.scale))))
        crop = crop.resize(new_size, Image.Resampling.LANCZOS)
        upscale_applied = True
    request = {**base_service_request,
               'image': image_to_b64(crop),
               'quality_mode': 'fast',
               'return_probability': True}
    meta = {'upscale_applied': upscale_applied,
            'request_size': list(crop.size),
            'region_xywh': [region.x, region.y, region.w, region.h]}
    return request, meta


def infer_region(transport: Callable[[dict], dict], region: Region, request: dict,
                 region_size: tuple, threshold: float = 0.5) -> LocalInference:
    """对单个区域执行局部推理；任何失败只返回 failed 结果，绝不抛出。"""
    try:
        response = transport(request)
    except Exception as exc:  # 传输/网络/服务异常一律降级为区域失败
        return LocalInference(region_index=region.index, status='failed',
                              cost=1.0, error=f'{type(exc).__name__}: {exc}')
    if not isinstance(response, dict) or response.get('status') != 'success':
        return LocalInference(region_index=region.index, status='failed', cost=1.0,
                              error='service_response_invalid')
    encoded = response.get('probability_map')
    if not encoded:
        return LocalInference(region_index=region.index, status='failed', cost=1.0,
                              error='missing_probability_map')
    try:
        probability = decode_probability_png(encoded, size=region_size)
    except Exception as exc:
        return LocalInference(region_index=region.index, status='failed', cost=1.0,
                              error=f'probability_decode_failed: {type(exc).__name__}')
    mask = ((probability >= threshold) * 255).astype(np.uint8)
    metadata = {key: response[key] for key in ('timing_ms', 'peak_vram_mb')
                if key in response}
    return LocalInference(region_index=region.index, status='success',
                          probability=probability, mask=mask, cost=1.0,
                          service_metadata=metadata)
