"""GeoScope 自主算法层（队长独占模块，队员 A/B 只通过接口消费）。

包含 GeoScope-AFR（Adaptive Focus and Refinement）面向多尺度遥感目标的
区域自适应精细推理与一致性复核方法：

- region_scoring      区域难度估计（模块 A）
- adaptive_refinement 预算约束下的局部精细推理（模块 B）
- fusion              跨尺度概率融合（模块 C）
- consistency         整图与局部预测的一致性诊断（模块 D）
- adaptive_runner     编排入口

诚实性声明（与国赛方案一致）：
- 分歧（disagreement）表示两种推理路径的预测差异，不是错误概率；
- 候选连通区域不是准确实例计数；
- 概率图经服务端 8-bit 量化传输（round(p*255)/255），熵与融合均基于量化值计算。
"""
from .region_scoring import AFRWeights, Region, RegionScore
from .adaptive_refinement import (
    Budget,
    LocalInference,
    UpscaleConfig,
    decode_probability_png,
    encode_probability_png,
    image_to_b64,
    select_regions,
    tile_grid,
)
from .fusion import fuse, threshold_binary
from .consistency import (
    DISAGREEMENT_NOTICE,
    composite_disagreement,
    disagreement_mask,
    region_disagreement_stats,
    review_regions,
)
from .adaptive_runner import (
    AFRConfig,
    AFRInitializationError,
    AFRResult,
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    config_fingerprint,
    config_snapshot,
    run_adaptive,
)

__all__ = [
    'AFRWeights', 'Region', 'RegionScore',
    'Budget', 'LocalInference', 'UpscaleConfig',
    'decode_probability_png', 'encode_probability_png', 'image_to_b64',
    'select_regions', 'tile_grid',
    'fuse', 'threshold_binary',
    'DISAGREEMENT_NOTICE', 'composite_disagreement', 'disagreement_mask',
    'region_disagreement_stats', 'review_regions',
    'AFRConfig', 'AFRInitializationError', 'AFRResult',
    'ALGORITHM_NAME', 'ALGORITHM_VERSION',
    'config_fingerprint', 'config_snapshot', 'run_adaptive',
]
