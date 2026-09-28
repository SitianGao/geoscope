# RemoteSAM 多图评测口径（sam-eval-v1.0）

制定人：队员 B。冻结日期：2026-09-27。适用范围：GeoScope 首批 **RemoteSAM 分割工具**评测；不是 Agent 任务成功率、整条链路成功率，也不是 WHU/iSAID 全量榜单成绩。规则必须在查看本批模型预测前确定；有实质性变更时升级版本并通知团队，不能悄悄替换失败图或修改提示词。

## 1. 固定数据与编号

- 唯一清单：[dataset_manifest.csv](dataset_manifest.csv)，SHA-256：`5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241`。
- 共 20 幅：WHU Building Dataset Satellite Dataset I 建筑 10 幅；iSAID 验证集标注 + DOTA v1.0 验证集原图，飞机 10 幅。清单中的 `image_id` 即团队统一编号，A、B、队长使用同一编号，不另起 ID。
- WHU 真值是建筑物二值标签；iSAID 真值由官方语义标注中 `plane` 的 RGB `(0,127,255)` 像素转成二值 mask，**不包含单列的 helicopter 类**。所有真值均有非零前景，故不存在“双空 mask”指标约定问题。
- 清单的图像/mask 路径均相对于外置数据根目录；路径、尺寸、前景像素数、源文件与输出文件哈希固定。运行前以 `verify_dataset_manifest.py` 校验。原始数据、权重和本机绝对路径不进入 Git。
- WHU 这 10 幅是固定诊断子集，并非官方测试集；飞机 10 幅是 iSAID 验证集子集。不得把子集均值称为整个数据集精度。

## 2. 固定请求语义

主指标使用当前系统真实工具路径 `POST /predict` 的 `task=referring_seg`，`return_probability=false`，每图一次请求：

| 真值类别 | GeoScope 目标 | 固定英文 `text` | 含义 |
| --- | --- | --- | --- |
| building | building | `all buildings` | 全图所有建筑 |
| plane | aircraft | `all airplanes` | 全图所有飞机，不包括直升机 |

使用清单中的 RGB PNG，客户端不缩放、不裁切、不平铺、不按结果更换压缩格式；预测 mask 必须与真值、原图同尺寸。只比较完整全图，空间约束任务另行评测。提示词不按单图调优，也不把真值信息传给模型。

**团队需注意的接口差异：**现有 `agent_bridge.py` 把 `aircraft` 生成为 `all aircraft`，比 iSAID 的 `plane` 真值更宽；固定子集中 `P1184`、`P1483` 还含单列的直升机标注。因此，本协议使用 `all airplanes` 评估飞机类。若答辩要把这组数字称为 GeoScope 当前整条系统链路的精度，应先由主流程负责人对齐提示词并重新联调；在对齐前只能称为 RemoteSAM 工具评测。

## 3. 固定服务与输入上限

- 首次正式请求前，`GET /ready` 必须返回 HTTP 200、`status=ready`；保存 `GET /model-info` 与服务代码提交号。
- 预期服务版本 `1.1.0`，模型权重 SHA-256 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`，`cuda:0`、FP16 开、EPOC 关。若任一关键配置不同，先暂停并发布新版实验配置，禁止混合统计。
- 选中的 `P0179` PNG 约 26.46 MiB，超过服务默认 20 MiB 请求上限。**在首张图之前**统一设 `GEO_REMOTESAM_MAX_IMAGE_MB=40` 并重启服务，20 幅采用同一配置。最大像素数保持 64,000,000；不为某一张图单独改变图像几何形状。
- 大图若出现显存不足、超时或接口错误，保留为失败案例并记录错误。若后来探索平铺/缩放，必须另列为 `sam-eval-v2` 实验，不能混进本版均值。

## 4. 逐图记录与指标

正式批次每图仅发一次请求。HTTP 错误、网络异常，以及 HTTP 200 但 `status=error`，均不是成功预测。客户端记录 HTTP 状态、响应状态、具体错误、墙钟耗时；成功响应另记服务 `timing_ms.inference`、`timing_ms.total`、输出尺寸、前景像素、空 mask、模型元数据。原始预测 mask 与必要诊断文件保存在本地受控目录，不提交 Base64 结果。

对尺寸正确、仅含 0/255 的预测二值 mask，与真值逐像素计算 `TP`、`FP`、`FN`：

- `IoU = TP / (TP + FP + FN)`；
- `Dice = 2TP / (2TP + FP + FN)`。

真值均有前景，因此合法的空预测取 `IoU=Dice=0`。无 mask、非二值或尺寸不一致记为失败；**主报告**为固定 20 图的逐图宏平均，失败图指标按 0 计并另列失败率。另报成功响应子集均值供诊断，但不得代替主报告。建筑与飞机分别报告，不只给总体均值。平均/中位推理耗时只对有合法服务耗时的响应统计；客户端墙钟耗时单列。所有均值从逐图 CSV 重新计算。

## 5. 执行批次与交付

1. 先运行 5 图冒烟：`WHU_SAT1_18_0`、`WHU_SAT1_25_1`、`WHU_SAT1_6_2`、`ISAID_VAL_P0257`、`ISAID_VAL_P1492`。交付 `evaluation/results/remotesam_smoke_results.csv` 和 2 个成功、1 个失败或低质量的可核验证据。
2. 在同一模型/服务配置下跑清单全部 20 图，交付 `evaluation/results/remotesam_full_results.csv`、`evaluation/results/remotesam_summary.json`，按建筑/飞机分别汇总 IoU、Dice、耗时、空 mask 与失败率。
3. 独立重复运行 `WHU_SAT1_18_0`、`ISAID_VAL_P0257`、`ISAID_VAL_P1143` 各至少一次；重复结果不计入主批次 20 图均值，用于稳定性说明。
4. 留下最好、中等、最差案例的原图、真值、预测、可视化及原因分析；队长复算指标后，将代码、脱敏结果与说明提交评审。公开材料中的原始影像须另核对数据集使用与再分发限制。

本协议是评测的**事前约定**，不是已取得的精度结论。当前数据已准备、校验；真实 5 图和 20 图模型推理结果尚待执行。
