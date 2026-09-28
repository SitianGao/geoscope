# RemoteSAM 多图分割评测说明（队员 B，2026-09-27）

## 结论与边界

依据事前冻结的 [`sam-eval-v1.0`](../evaluation/REMOTESAM_EVALUATION_PROTOCOL.md) 在真实 RemoteSAM 服务上完成 5 图冒烟、20 图正式评测及 3 图独立重复。**20/20 次接口调用成功、输出尺寸及 0/255 二值合法，但分割精度明显不足，尤其是飞机。**正式批次的逐图宏平均 IoU 为 **0.2554**，Dice 为 **0.3698**。这是一组固定诊断子集上的 RemoteSAM 工具结果，**不是** WHU/iSAID 官方榜单成绩，也**不是** GeoScope Agent→Harness→RemoteSAM 端到端成功率。

| 类别 | 图数 | 成功 / 失败 | 平均 IoU | 平均 Dice | 平均 / 中位服务推理耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 建筑（WHU） | 10 | 10 / 0 | 0.3522 | 0.5115 | 262.11 / 236.48 ms |
| 飞机（iSAID+DOTA） | 10 | 10 / 0 | 0.1585 | 0.2281 | 364.66 / 346.49 ms |
| 合计 | 20 | 20 / 0 | 0.2554 | 0.3698 | 313.38 / 267.64 ms |

20 图均无空 mask；尺寸一致率、二值合法率均为 20/20，服务失败率和空 mask 率均为 0。**有效返回不能代替分割准确。**平均/中位耗时取服务响应 `timing_ms.inference`，仅反映模型推理，不含图像编码、传输和叠加图生成。逐图 CSV 同时保留客户端墙钟时间及服务总耗时，大图的端到端等待明显更长。

## 数据、模型与口径

- 固定清单：[`evaluation/dataset_manifest.csv`](../evaluation/dataset_manifest.csv)，SHA-256 `5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241`。WHU Building Dataset Satellite Dataset I 诊断子集 10 图；DOTA v1.0 验证集原图 + iSAID 验证集 `plane` 标注 10 图。所有图片与真值保留在本地数据目录，不进入 Git。
- 固定请求：`POST /predict`，`task=referring_seg`，`return_probability=false`；建筑 `all buildings`，飞机 `all airplanes`；RGB PNG 原尺寸、全图单次请求、不按单图改提示词、无裁切或平铺。飞机真值不含单独的 `helicopter` 类。
- 固定服务：`service_version=1.1.0`，权重 SHA-256 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`，`cuda:0`、FP16 开、EPOC 关；统一设置 40 MiB 输入上限。每次运行前校验 `/ready` 与 `/model-info`。正式批次 20 图每图只请求一次；任何接口/输出失败按 0 纳入主均值。
- 指标：真值与预测的 TP、FP、FN 逐像素计算 `IoU=TP/(TP+FP+FN)`、`Dice=2TP/(2TP+FP+FN)`，先逐图计算再宏平均。原始逐图记录、预测 mask 和真值可由独立校验脚本复算。

## 分批结果与可视化核查

5 图冒烟结果：5/5 成功，无空 mask；平均 IoU **0.3699**、Dice **0.5295**。这 5 图不是随机代表样本，不能用该均值代替完整 20 图结果。

正式 20 图的代表案例（下述“较好”只表示**本批相对较好**，并非达到可用精度阈值）：

| 案例 | IoU / Dice | 核查所见 |
| --- | ---: | --- |
| `ISAID_VAL_P1483`（较好、飞机） | 0.5665 / 0.7232 | 三架大型飞机主体基本定位，仍有边界差异；图中另有直升机，真值口径不含它。 |
| `WHU_SAT1_4_3`（较好、建筑） | 0.4640 / 0.6339 | 覆盖较多建筑，但预测前景 175,327 像素大于真值 110,326，仍存在明显误检。 |
| `WHU_SAT1_9_1`（中等、建筑） | 0.3397 / 0.5072 | 对部分屋顶有效，FP 39,707 与 FN 25,899 同时存在。 |
| `WHU_SAT1_51_2`（低质量、建筑） | 0.1282 / 0.2273 | 运动场/球场区域被大面积涂成建筑；FP 62,962，约为真值前景 16,219 的 3.9 倍。 |
| `ISAID_VAL_P0257`（冒烟低质量、飞机） | 0.2072 / 0.3433 | 多数真值飞机像素被覆盖，但停机坪/周边大片非飞机区域也被选中；FP 126,435，约为 TP 33,383 的 3.8 倍。 |
| `ISAID_VAL_P1738`（最差、飞机） | 0.0000 / 0.0000 | 预测前景 232,813 像素与右侧飞机真值 23,148 像素完全不相交，TP=0；属于**语义/定位失败**，不是接口失败。 |

飞机另有 `P1184`、`P1529` 等 IoU 低于 0.01 的大图，预测前景分别为 2,784,607 和 6,612,700 像素，远大于真值 5,125 和 10,427 像素；从逐图像素统计看，大面积误检是主要症状。大图及复杂背景可能放大问题，但这只是待验证假设，不能直接当作因果结论。

原图和官方真值按清单保留在 Git 仓库外的数据根目录；预测 PNG、服务叠加图、去除 Base64 后的响应元数据按批次保留在 Git 仓库外的 `smoke_20260927`、`full_20260927`、`repeat_20260927` 目录。以上案例均可按 `image_id` 找到原图、真值、预测和叠加图。由于数据再分发许可需单独确认，仓库只提交脱敏逐图表与汇总，不上传原始图片或叠加图。

## 3 图重复性

协议预定的 `WHU_SAT1_18_0`、`ISAID_VAL_P0257`、`ISAID_VAL_P1143` 在正式 20 图之外各独立再运行一次。3 次重复均成功，预测 PNG SHA-256 与正式批次逐字节一致，IoU/Dice 完全相同；重复结果**不并入** 20 图主均值。

| 影像 | 正式 / 重复 IoU | 正式 / 重复推理耗时 |
| --- | ---: | ---: |
| `WHU_SAT1_18_0` | 0.4305 / 0.4305 | 494.57 / 538.07 ms |
| `ISAID_VAL_P0257` | 0.2072 / 0.2072 | 262.64 / 255.96 ms |
| `ISAID_VAL_P1143` | 0.0187 / 0.0187 | 444.92 / 427.48 ms |

这说明此次同一环境下输出稳定；三图两次并不足以证明跨硬件、跨版本、长时间运行的稳定性。

## 复现与交付文件

运行环境需有 `numpy`、`Pillow`，并能访问已启动的 RemoteSAM 服务。`$dataRoot` 指向保存清单相对路径的本地目录；数据和权重需单独按许可取得。以下命令从仓库根目录运行，示例假设数据和运行产物在仓库的上一级目录；脚本会拒绝覆盖已有结果：

```powershell
$dataRoot = Join-Path (Split-Path (Get-Location)) 'datasets'
$artifactsDir = Join-Path (Split-Path (Get-Location)) '_runtime\remotesam\full_rerun'
python evaluation/verify_dataset_manifest.py --manifest evaluation/dataset_manifest.csv --data-root $dataRoot --expected-sha256 5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241
python evaluation/run_remotesam_eval.py --manifest evaluation/dataset_manifest.csv --data-root $dataRoot --base-url http://127.0.0.1:6657 --run full --results-csv evaluation/results/remotesam_full_rerun.csv --summary-json evaluation/results/remotesam_summary_rerun.json --artifacts-dir $artifactsDir --timeout 300
python evaluation/verify_remotesam_results.py --manifest evaluation/dataset_manifest.csv --data-root $dataRoot --results-csv evaluation/results/remotesam_full_rerun.csv --summary-json evaluation/results/remotesam_summary_rerun.json --artifacts-dir $artifactsDir --run full
```

已有结果文件：[`remotesam_smoke_results.csv`](../evaluation/results/remotesam_smoke_results.csv)、[`remotesam_full_results.csv`](../evaluation/results/remotesam_full_results.csv)、[`remotesam_summary.json`](../evaluation/results/remotesam_summary.json)、[`remotesam_repeat_results.csv`](../evaluation/results/remotesam_repeat_results.csv)。三组结果均已用 [`verify_remotesam_results.py`](../evaluation/verify_remotesam_results.py) 从本地保存的预测 mask 和真值独立复算通过。

## 交给团队的决策

1. **请勿把 20/20 接口成功宣传为分割准确或端到端成功。**答辩应同时展示分类别均值、最佳与最差样例，并如实解释错误。
2. 现有主流程 `agent_bridge.py` 对 `aircraft` 使用 `all aircraft`，本次为对齐 iSAID `plane` 真值使用 `all airplanes`。若队长希望把本结果称为**当前 GeoScope 整条链路**的精度，需主流程负责人先对齐语义并重新联调；否则仅标记 RemoteSAM 工具评测。
3. 后续优化应另开**不污染基线**的实验：优先验证飞机复杂背景下的提示词、平铺/多尺度和误检抑制；记录每种改动的完整 20 图逐图结果及额外耗时，不能仅选成功样例或替换失败图。任何实质性评测口径变化升级协议版本。模型未达精度目标前，前端应保留人工核查/修正提示。
