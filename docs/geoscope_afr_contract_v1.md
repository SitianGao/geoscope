# GeoScope 队长版最小接口合同（geoscope-afr-v1）

> 队长公布（《国赛迭代开发完整方案》13.5 步骤 2）。队员 A 的工具与队员 B 的
> 前端通过本合同的字段消费算法产物；字段版本号 `geoscope-afr-v1`。
> 本文件只描述**队长算法层**的产物语义；队员 A 的业务 API 合同另行维护。

## 1. 算法产物对象

### 1.1 `AlgorithmResult`（每次 AFR 运行的产物）

| 字段 | 类型 | 语义 |
|---|---|---|
| `image_id` | string | 当前影像 ID（会话/影像绑定，不接受任意本地路径） |
| `result_id` | string | 当前结果版本 ID（12 位 run id）；比较时必须标明 base / target |
| `mask` | base64 PNG | 融合后二值蒙版，**0/255**，尺寸 == 原图（width×height） |
| `width` / `height` | int | 原始影像像素尺寸；ROI 与区域坐标均为原图像素坐标 |
| `model_info` | object | 服务返回的模型元数据原样透传（名称/版本/checkpoint sha256 等） |
| `algorithm_info` | object | 见 1.2 |
| `status` | string | `completed` \| `fallback` \| `failed` \| `needs_review` |
| `probability_map_ref` | string \| null | 融合概率图（PNG）引用；**8-bit 量化**，见 1.3 |
| `disagreement_ref` | string \| null | 分歧图（PNG）引用；无局部成功区域时为 null |
| `review_regions` | array | 待复核区域列表（按分歧占比降序） |
| `fallback_records` | array | 区域级失败记录：`{region_index, xywh, stage, error, action}`，`action` 固定为 `whole_image_probability_kept` |
| `real_execution` | bool | 恒为 true（本链路无 mock）；demo 模式不产生本对象 |

### 1.2 `algorithm_info`

```json
{
  "name": "geoscope_afr",
  "version": "v1",
  "config": {
    "alpha": 0.5, "beta": 0.3, "gamma": 0.2,
    "budget": {"max_regions": null, "fraction": 0.25},
    "lam": 0.5, "lam_per_region": null,
    "threshold": 0.5,
    "tile_size": 1024, "overlap": 256,
    "upscale": {"enabled": true, "min_size": 512, "scale": 2.0},
    "review_ratio_threshold": 0.05
  },
  "config_sha256": "<配置指纹>",
  "threshold_validated": false
}
```

- `threshold_validated`：λ/阈值是否已在开发验证集确认；未确认前恒为
  `false`，任何报告不得据此宣称"已优化"。
- 配置指纹 = sha256(排序后配置 JSON)，用于结果缓存隔离与复算溯源。

### 1.3 概率图量化语义（必须遵守）

- 服务端传输的概率图为 8-bit 灰度 PNG：`round(p*255)`，p∈[0,1]，步长 1/255。
- 客户端还原：`p = value / 255`。熵、融合、分歧统计全部在**量化值**上计算。
- 概率图是模型输出，**不是校准置信度**；禁止把概率或相似度当作"目标存在概率"
  对外展示。

## 2. 语义红线（对 A/B 有效）

1. `disagreement`（分歧图/分歧占比）= 整图与局部预测的分歧像素，
   **不是错误概率、不是目标数量**；UI 必须附 `DISAGREEMENT_NOTICE` 声明。
2. `prediction_agreement_iou` = 两个预测之间的一致性；空集规则固定：
   双空 = 1.0，仅一个为空 = 0.0。**不是**分割准确率（无真值标签时禁止叫 iou）。
3. 候选连通区域数 ≠ 准确实例数（沿用平台既有 notice）。
4. `status=fallback` 表示 AFR 编排失败后回退整图 fast 路径完成，
   `result.afr.status='fallback'` + `quality_adaptation.reason_code='adaptive_fallback'`；
   主链路结果仍然有效，但不得声称本次运行了自适应推理。
5. 非飞机目标当前不开放 adaptive（`adaptive_not_validated_for_target`），
   前端选择 adaptive 而目标非飞机时按回退展示。

## 3. 质量模式

| 模式 | 含义 | 对外发送 |
|---|---|---|
| `fast` | 整图推理（服务端 fast） | `quality_mode=fast` |
| `accurate` | 服务端固定切片融合（仅飞机验证过） | `quality_mode=accurate` |
| `adaptive` | **Harness 层** GeoScope-AFR 编排（整图 fast + 局部 fast 多次调用） | 对外只发 `fast`；`adaptive` 永不发给服务端 |
| `auto` | 文字/尺寸解析（`≥1M px → accurate`，其余 fast；文字含"自适应/智能/adaptive" → adaptive） | — |

## 4. 区域坐标与网格

- 候选区域网格与服务端 accurate 切片语义一致：1024 切片 / 256 重叠，
  末块贴边（起点 = 边长 − 1024）；图像小于切片时为单区域整图。
- 坐标：原图像素坐标，左上包含、右下不包含；ROI 沿用平台既有约定。
- 局部推理成本单位 = 服务请求次数（`request_count = 1 + 选中区域数`）；
  墙钟/服务耗时单独记录（`client_wall_ms` / `service_total_ms`）。

## 5. 产物文件（每次运行的实验目录）

| 文件 | 内容 |
|---|---|
| `full_mask.png` | 融合二值蒙版（0/255，原图尺寸；与既有主链路语义一致） |
| `probability_map.png` | 融合概率图（8-bit 量化） |
| `disagreement.png` | 分歧图（0/255；无局部成功区域时不存在） |
| `afr.json` | 本合同的 algorithm_info + review + fallback_records + honesty_notes |
| `mask.png` / `overlay.png` / `statistics.json` / `result.json` / `run_log.json` / `report.md` | 既有主链路产物，语义不变 |

## 6. 回退语义（对 B 的展示要求）

- AFR 任一环节失败 → `afr.status='fallback'`，系统自动执行整图 fast 并保留
  结果；页面显示"自适应推理未完成，已回退整图模式"，**不是**程序崩溃。
- 区域级失败（部分区域）→ 该区域保留整图概率，`fallback_records` 列出；
  属于正常降级，结果仍为 `completed`。
