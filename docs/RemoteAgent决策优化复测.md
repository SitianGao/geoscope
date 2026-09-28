# RemoteAgent 决策优化复测

## 改动

工作台现在向 RemoteAgent 发送仅包含已部署两种分割工具的短系统提示词，不再附带上游完整 MCP 工具库。提示词规定唯一 `T_call` 格式、整图分割、飞机指代 `all airplanes`，并把半幅裁切和面积统计交给 Harness。工具白名单与参数校验仍在执行前生效。

## 冻结基线（2026-09-24，真实模型）

原始 40 条任务见 `evaluation/task_set.csv`，逐条结果见 `evaluation/results/agent_full_results.csv`，原始输出见 `agent_full_results_raw.jsonl`，汇总见 `agent_summary.json`。整体路由正确 30/40（75.0%）；六条飞机任务正确 1/6。

| 任务 | 修改前状态 | 工具 | 失败原因 |
| --- | --- | --- | --- |
| E01 | rejected_tool | skysense_detection | 模型选择未部署的 object_detection |
| E02 | rejected_tool | skysense_detection | 模型选择未部署的 object_detection |
| E03 | unparsed | 无 | 英文任务生成未闭合的 XML 式调用 |
| E04 | reject_args | referring_expression_segmentation | 把“右侧的飞机”写进工具 prompt |
| E05 | unparsed | 无 | 生成不符合协议的 object_detection 调用 |
| E06 | segment | semantic_segmentation | 成功 |

## 真实模型复测（2026-09-28）

通过实验室主机上的原 RemoteAgent vLLM 0.15.1 服务运行同一 40 条任务，`temperature=0`。本机 SSH 配置权限异常，因此用原有密钥直接建立到实验室 `8000` 端口的隧道。工作台提示词只暴露两种部署工具；dense 决策上限 512 token，场景问答 1024 token。每轮均重新请求真实模型，未从基线回放。

| 指标 | 修改前 | 最终修改后 |
| --- | ---: | ---: |
| 整体路由正确 | 30/40（75.0%） | 39/40（97.5%） |
| 飞机任务正确 | 1/6 | 6/6 |
| 合法工具调用率 | 20/26（76.9%） | 29/29（100%） |
| 含糊/非法任务正确拦截 | 6/6 | 6/6 |
| `rejected_tool` | 6 | 0 |
| `unparsed` | 3 | 1 |

最终六条飞机任务均调用 `referring_expression_segmentation`，参数 prompt 均为 `all airplanes`；E04 的 `right` 由 Harness 确定。唯一失败是 A04 场景描述：模型返回 `<answer>` 开头和完整描述正文，但没有闭合 `</answer>`，因此解析器按协议标记 `unparsed`。没有把该正文强行视为成功。

最终证据：

- `evaluation/results/agent_full_results_final.csv`：40 条完整结果；
- `evaluation/results/agent_full_results_final_raw.jsonl`：40 条模型原始输出与结构化决策；
- `evaluation/results/agent_summary_final.json`：真实汇总；
- `evaluation/results/agent_before_after.csv`：40 条修改前后对照；
- `evaluation/results/agent_aircraft_before_after.csv`：单独的六条飞机对照；
- `evaluation/results/agent_full_results_final_baseline_vs_harness.csv`：Agent 与 Harness 动作对照。

第一次优化和第二次调整的完整原始记录也保留在同目录的 `optimized` 与 `optimized_v2` 文件，便于审查提示词迭代。第一次为 38/40，第二次和最终均为 39/40；最终场景问答 token 上限恢复到 1024，A04 仍偶发缺闭合标签。以上是决策和 Harness 校验结果，不包含真实蒙版像素精度测量。

## 复现命令

实验室 RemoteAgent 服务隧道连通后，从仓库根目录运行：

```powershell
$env:GEO_AGENT_BASE_URL = 'http://127.0.0.1:18000/v1'
$env:GEO_AGENT_MODEL = 'RemoteAgent'
$env:GEO_SERVICE_PROXY_MODE = 'direct'
python evaluation/agent_eval.py --out agent_full_results_final.csv --summary agent_summary_final.json
python evaluation/compare_runs.py evaluation/results/agent_full_results.csv evaluation/results/agent_full_results_final.csv --output evaluation/results/agent_before_after.csv --aircraft-output evaluation/results/agent_aircraft_before_after.csv
```

上述命令会保存 40 条完整新 CSV、原始 JSONL、汇总 JSON，以及 40 条和 6 条飞机任务的前后对照。评测只检查决策与 Harness 校验，不调用实际分割服务，因此不代表蒙版像素精度。
