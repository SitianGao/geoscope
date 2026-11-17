# GeoScope-AFR（Adaptive Focus and Refinement）

面向多尺度遥感目标的**区域自适应精细推理与一致性复核方法**。以预训练遥感分割模型
（RemoteSAM）为底座，通过区域难度估计、预算约束下的局部精细推理、跨尺度概率融合与
结果一致性检查，把固定推理配置升级为自适应分析流程。

> 本方法建立在 RemoteSAM 输出之上；RemoteSAM 本身的模型与训练成果不是团队贡献。
> 本目录为队长独占模块，队员 A/B 只通过 `run_adaptive` 入口与结果合同消费。

## 链路

```
整图 fast + 概率图 P0
  → 区域难度估计 U_i = α·H̄_i + β·D_i + γ·C_i   (region_scoring)
  → 预算贪心选择（B 次局部调用）                (adaptive_refinement)
  → 选中区域局部推理（裁剪 / 小区域放大）       (adaptive_refinement)
  → 跨尺度融合 P_f = (1-λ)·P0 + λ·P_local       (fusion)
  → 分歧诊断 D = |M_whole − M_local| → 待复核区域 (consistency)
```

全部经由既有 RemoteSAM `/predict` 接口完成（包装层编排），服务代码零改动；
对外只发 `quality_mode='fast'`（'adaptive' 是 Harness 层模式，服务端不认识）。

## 公式与语义

| 记号 | 含义 |
|---|---|
| H(p) | 二分类熵 `-p·log(p+ε) - (1-p)·log(1-p+ε)`，在 8-bit 量化概率（round(p·255)/255）上计算 |
| C | 梯度模边缘密度 `/ (255·√2)`，固定归一化（跨图可比） |
| D_i | 区域内 `\|P0 − P_local\|` 均值；局部结果不存在时为 0 并标 `d_source='unavailable'` |
| Budget | 推理预算：`max_regions` 或 `fraction`（占网格比例，向下取整）；成本单位 = 服务请求次数 |
| λ | 融合系数 ∈ [0,1]（默认 0.5，可 per-region 覆盖）；未验证前不宣称最优 |
| threshold | 二值化阈值 ∈ (0,1)（默认 0.5；既有 accurate 模式用 0.88，两者含义不同） |

## 诚实性约束（与国赛方案一致）

1. 预测概率不等于校准置信度；难度评分是复核线索，不是正确率。
2. `disagreement` 是整图与局部预测的分歧像素，**不是错误概率、不是目标数量**；
   该声明随结果强制附上（`DISAGREEMENT_NOTICE`）。
3. `prediction_agreement_iou` 只衡量两个预测之间的一致性，空集规则固定：
   双空 = 1.0，仅一个为空 = 0.0。
4. λ 与阈值需在开发验证集确认；未确认时元数据 `threshold_validated: false`，
   不得在报告里宣称"已优化"。
5. 局部区域推理失败 → 该区域保留整图概率（`fallback_records` 记录）；
   整图概率获取失败 → 抛 `AFRInitializationError`，由 Harness 回退既有 fast 链路。

## 使用

```python
from geoscope_algorithms import run_adaptive, AFRConfig, Budget

result = run_adaptive(
    image=pil_image,
    base_service_request={'task': 'referring_seg', 'text': 'all planes'},
    transport=lambda req: post_json(url, req),   # 注入，不持有网络实现
    config=AFRConfig(budget=Budget(fraction=0.25), lam=0.5, threshold=0.5),
)
# result.probability / result.fused_mask / result.disagreement / result.review
# result.fallback_records / result.metadata（含 config_sha256、request_count 等）
```

## 测试

```bash
python tests/test_geoscope_algorithms.py -v
python tests/test_afr_runner.py -v
```
