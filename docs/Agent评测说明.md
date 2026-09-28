# RemoteAgent 评测说明（队员 A）

> 分支：`feature/competition-agent-eval`
> 对应任务：`docs/三人团队三天冲刺安排.md` 中队员 A 的第 1、2、3 天交付
> 结果冻结时间：2026-09-24

---

## 1. 目的

验证 GeoScope 主链路中 **RemoteAgent（认知核心）** 的决策质量，并量化
**Harness（校验/执行）** 在多大程度上拦截了非法动作、纠正了错误参数。
评测只关注“模型决策 + Harness 校验”这一段，不执行真正的分割（分割精度由队员 B 负责）。

评测必须做到三点：

1. **可复现**：复用工作台 `server.py` / `agent_bridge.py` 的真实路由与校验代码，不另写一套近似逻辑；
2. **可复算**：所有汇总数字都能从逐条 CSV 重新计算（`--from-raw` 提供纯复算入口，不调用模型）；
3. **诚实**：真实模型结果与确定性拦截分开标记，失败原因具体到类别，不统一写“模型问题”。

---

## 2. 评测对象与版本

| 项目 | 值 |
|---|---|
| 模型 | `AIMGroup/RemoteAgent`（Qwen2.5-VL，BF16） |
| 服务 | vLLM 0.15.1，模型 id `RemoteAgent`，实验室 GPU 0 |
| 访问方式 | 本地 SSH 隧道，评测进程内 `GEO_AGENT_BASE_URL=http://127.0.0.1:18000/v1`（`/v1` 后缀必须保留，`WorkbenchAgent` 直接拼 `/chat/completions`） |
| 允许工具 | `referring_expression_segmentation`、`semantic_segmentation`（`agent_bridge.ALLOWED_TOOLS`） |
| 采样 | `temperature=0`，`max_tokens=1024` |
| 分支 | `feature/competition-agent-eval`（基于最新 `main`） |

> 注：评测结果文件里的 `endpoint` 字段是 `127.0.0.1:18000/v1`（本机隧道回环地址），
> 不含实验室内网 IP（10.x）、令牌或密钥，符合“不提交校园网地址/令牌”的规则。

---

## 3. 数据与任务集

### 3.1 影像（7 幅，见 `evaluation/agent_dataset_manifest.csv`）

| image_id | 目标类别 | 尺寸 | 来源 |
|---|---|---|---|
| urban | building | 800×600 | `web/assets/urban.png`（SM3Det CC BY-NC 4.0） |
| airport | aircraft | 800×800 | RemoteSAM demo 图 |
| hkust | building | 226×234 | RemoteSAM 公开样例 HKUST |
| nuist | building | 333×333 | RemoteSAM 公开样例 NUIST |
| aerial_1831 | building | 800×600 | SM3Det `1831.tif`（CC BY-NC 4.0） |
| satellite_qingming | building | 1600×554 | DirectSAM 公开样例 |
| satellite_mixed | mixed | 1600×1130 | DirectSAM 公开样例 |

- 影像二进制**不提交 Git**（`.gitignore` 已排除 `evaluation/images/`），
  由 `evaluation/prepare_images.py` 从磁盘源确定性地重建并写 SHA-256。
- 许可证尚未逐幅独立核实，标注为“来源注明、未经独立授权核实”。

### 3.2 任务集（40 条，见 `evaluation/task_set.csv`）

| 类别 | 数量 | 期望动作 | 说明 |
|---|---|---|---|
| scene_qa（直接回答） | 8 | `answer` | 场景问答，不应调用工具 |
| full_extract（全图提取） | 8 | `tool_call` building/all | 中文 + 英文 |
| side_extract（左右/上下） | 8 | `tool_call` building/{left,right,top,bottom} | 空间条件 |
| modify（连续修改） | 4 | `tool_call` | 继承上下文，改侧/换目标 |
| aircraft（飞机提取） | 6 | `tool_call` aircraft | 中文 + 英文 |
| ambiguous（含糊任务） | 3 | `clarify` | 必须 fail-closed |
| illegal（非法/多工具） | 3 | `clarify` | 车辆 / 多类别 / 轮廓 |

> 任务集 `task_set.csv` 的 `expected_side` 是地面真值标注。D03「现在换成提取飞机」的期望
> 侧向已按 Harness 语义从 `all` 校正为 `right`：该指令是“换目标、不改侧”，Harness 会从
> 上下文继承上一版本的 `side=right`（`resolve_scope` 在未显式给出侧向时回退到
> `context.get('side', 'all')`）。这是标注修正，不是模型结果改动，原始模型输出未做任何改动。

---

## 4. 评测管线（复用真实代码）

每条任务的判定严格复刻 `server.py` 的 live 流程：

1. **确定性预检（Harness 侧，不调用模型）**
   - `requires_dense_result(query, context)` 决定路由 `dense` / `internal`；
   - `resolve_scope(query, context, None)` 解析像素范围（左/右/上/下/全图），
     对 `靠近/东侧/平方米/除外` 等复杂关系直接抛 `Clarification`；
   - `那个|那片|那块|这块` 模糊指代直接 `Clarification`。
   - 命中即标记为 `deterministic_clarify`（模型**未被调用**，`agent_status=n/a`）。
2. **模型决策（RemoteAgent，单步，不执行工具）**
   - `AgentTurn(...)` → `agent.plan(...)` → `AgentResult`，状态为
     `completed` / `tool_call` / `rejected_tool` / `unparsed`。
3. **Harness 校验（`segmentation_plan`）**
   - `completed` 且 dense → `dense_answered_text`（失败）；internal → `scene`（成功）；
   - `tool_call` 且 internal → `internal_called_tool`（失败）；
   - `tool_call` 且 dense → `segmentation_plan(...)`：通过得 `segment`（含目标/侧），
     参数不合法（多类别、其他类别、空间关系写进 prompt）抛 `Clarification`/`ValueError` → `reject_args`；
   - 其余为 `rejected_tool` / `unparsed`。

判定正确性：`answer→scene`、`tool_call→segment`、`clarify→非 segment`（fail-closed 即算对）。

---

## 5. 运行方式

```bash
# 1) 准备影像（首次/换机时）
python evaluation/prepare_images.py

# 2) 起隧道到实验室 vLLM（GPU 0，端口 8000 → 本机 18000）
ssh -N -L 18000:127.0.0.1:8000 lab

# 3) 小样本（10 条）
GEO_AGENT_BASE_URL=http://127.0.0.1:18000/v1 \
GEO_AGENT_MODEL=RemoteAgent \
GEO_SERVICE_PROXY_MODE=direct \
python evaluation/agent_eval.py --limit 10 --out agent_smoke_results.csv

# 4) 全量（40 条）
python evaluation/agent_eval.py --out agent_full_results.csv --summary agent_summary.json

# 5) 纯复算（不调用模型，从冻结的原始记录重算汇总与对照表）
python evaluation/agent_eval.py --from-raw agent_full_results_raw.jsonl \
    --out agent_full_results.csv --summary agent_summary.json
```

`GEO_SERVICE_PROXY_MODE=direct` 用于绕过实验室失效的 `http_proxy=127.0.0.1:7890`，
否则 urllib 会走死代理导致连接失败。

---

## 6. 结果文件清单

| 文件 | 内容 |
|---|---|
| `evaluation/results/agent_smoke_results.csv` | 10 条小样本结构化结果 |
| `evaluation/results/agent_smoke_results_raw.jsonl` | 10 条原始输出（含模型原文） |
| `evaluation/results/agent_full_results.csv` | 40 条结构化结果（可复算） |
| `evaluation/results/agent_full_results_raw.jsonl` | 40 条原始输出（含模型原文与脱敏 decision_record） |
| `evaluation/results/agent_summary.json` | 汇总指标（脚本生成） |
| `evaluation/results/agent_full_results_baseline_vs_harness.csv` | 基础 Agent 动作 vs Harness 结果逐条对照 |

---

## 7. 汇总指标（40 条）

| 指标 | 值 | 口径 |
|---|---|---|
| 任务路由正确率 | **75.0%（30/40）** | answer→scene，tool_call→segment，clarify→fail-closed |
| 合法工具调用率 | **76.9%（20/26）** | 26 次工具调用中 20 次为允许工具 |
| 目标类别正确率 | **100%（16/16）** | 通过 Harness 的 16 次分割目标全部正确 |
| 空间条件解析正确率 | **100%（16/16）** | 通过 Harness 的 16 次分割侧向全部正确 |
| 需澄清任务识别率 | **100%（6/6）** | 6 条含糊/非法任务全部 fail-closed |
| 平均决策耗时 | 2524.8 ms | 仅计实际调用模型的 38 条 |
| 中位决策耗时 | 2447.0 ms | 同上；按偶数样本的中间两项平均值计算 |

---

## 8. 逐任务结果

> 完整 40 条原始记录见 `evaluation/results/agent_full_results.csv` 与
> `..._raw.jsonl`；下表为逐条摘要，汇总数字可由该表逐条重算。

| 任务 | 类别 | 请求 | 期望 | Agent 状态 | 工具 | Harness 结果 | 目标/侧 | 通过 |
|---|---|---|---|---|---|---|---|---|
| A01 | 直接回答 | 这张遥感影像主要是什么场景？ | answer | completed | — | scene | — | ✅ |
| A02 | 直接回答 | 描述这幅影像的内容。 | answer | completed | — | scene | — | ✅ |
| A03 | 直接回答 | 图中是否有大量建筑物？ | answer | completed | — | scene | — | ✅ |
| A04 | 直接回答 | 描述图中的地形和土地利用特征。 | answer | completed | — | scene | — | ✅ |
| A05 | 直接回答 | 这幅影像拍摄的是城市还是乡村？ | answer | completed | — | scene | — | ✅ |
| A06 | 直接回答 | 图中有没有明显的植被或水域？ | answer | completed | — | scene | — | ✅ |
| A07 | 直接回答 | 概括这幅影像的主要地物类型。 | answer | completed | — | scene | — | ✅ |
| A08 | 直接回答 | 这是一幅什么类型的遥感影像？ | answer | completed | — | scene | — | ✅ |
| B01 | 全图提取 | 提取全图的所有建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| B02 | 全图提取 | 提取整幅影像中的建筑像素。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| B03 | 全图提取 | 把图中的建筑分割出来。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| B04 | 全图提取 | Extract all buildings in this image. | tool_call | unparsed | — | unparsed | — | ❌ |
| B05 | 全图提取 | 分割出全图的建筑物。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| B06 | 全图提取 | 提取这幅影像中的所有建筑并计算面积占比。 | tool_call | rejected_tool | crossearth_semantic_segmentation | rejected_tool | — | ❌ |
| B07 | 全图提取 | 标注出图中的所有建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| B08 | 全图提取 | 分割整幅影像的建筑区域。 | tool_call | tool_call | semantic_segmentation | segment | building/all | ✅ |
| C01 | 左右/上下 | 提取右侧建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/right | ✅ |
| C02 | 左右/上下 | 提取左侧建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/left | ✅ |
| C03 | 左右/上下 | 提取上半部分的建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/top | ✅ |
| C04 | 左右/上下 | 提取下半部分的建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/bottom | ✅ |
| C05 | 左右/上下 | 提取右侧建筑并计算面积占比。 | tool_call | tool_call | semantic_segmentation | segment | building/right | ✅ |
| C06 | 左右/上下 | 只提取左边的建筑。 | tool_call | tool_call | referring_expression_segmentation | reject_args | — | ❌ |
| C07 | 左右/上下 | 提取图中右侧的建筑像素。 | tool_call | completed | — | dense_answered_text | — | ❌ |
| C08 | 左右/上下 | 分割出右半幅的建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/right | ✅ |
| D01 | 连续修改 | 改成左侧的建筑。 | tool_call | tool_call | semantic_segmentation | segment | building/left | ✅ |
| D02 | 连续修改 | 把提取结果改成左侧。 | tool_call | tool_call | semantic_segmentation | segment | building/left | ✅ |
| D03 | 连续修改 | 现在换成提取飞机。 | tool_call | tool_call | referring_expression_segmentation | segment | aircraft/right | ✅ |
| D04 | 连续修改 | 改成提取全图的建筑。 | tool_call | rejected_tool | crossearth_semantic_segmentation | rejected_tool | — | ❌ |
| E01 | 飞机提取 | 提取图中的飞机。 | tool_call | rejected_tool | skysense_detection | rejected_tool | — | ❌ |
| E02 | 飞机提取 | 分割出机场里的所有飞机。 | tool_call | rejected_tool | skysense_detection | rejected_tool | — | ❌ |
| E03 | 飞机提取 | Extract all aircraft in this image. | tool_call | unparsed | — | unparsed | — | ❌ |
| E04 | 飞机提取 | 提取右侧的飞机。 | tool_call | tool_call | referring_expression_segmentation | reject_args | — | ❌ |
| E05 | 飞机提取 | 标注出图中的飞机目标。 | tool_call | unparsed | — | unparsed | — | ❌ |
| E06 | 飞机提取 | 分割整幅影像中的飞机。 | tool_call | tool_call | semantic_segmentation | segment | aircraft/all | ✅ |
| F01 | 含糊任务 | 提取那个区域。 | clarify | n/a | — | deterministic_clarify | — | ✅ |
| F02 | 含糊任务 | 把那个东西分割出来。 | clarify | n/a | — | deterministic_clarify | — | ✅ |
| F03 | 含糊任务 | 提取画面中央的目标。 | clarify | tool_call | referring_expression_segmentation | reject_args | — | ✅ |
| G01 | 非法/多工具 | 提取图中的车辆。 | clarify | rejected_tool | skysense_detection | rejected_tool | — | ✅ |
| G02 | 非法/多工具 | 同时提取建筑和水体两个类别。 | clarify | tool_call | semantic_segmentation | reject_args | — | ✅ |
| G03 | 非法/多工具 | 提取图中所有物体的轮廓。 | clarify | rejected_tool | contour_extraction | rejected_tool | — | ✅ |

---

## 9. 失败类型分析（10 条失败，非统一“模型问题”）

### 9.1 非法工具名（4 条）→ `rejected_tool`

| 任务 | 模型调用的工具 | 根因 |
|---|---|---|
| B06 | `crossearth_semantic_segmentation` | CrossEarth 命名漂移：模型 GRPO 先验更偏好训练时用的 CrossEarth 工具名，而非 GeoScope 契约的 `semantic_segmentation` |
| D04 | `crossearth_semantic_segmentation` | 同上 |
| E01 | `object_detection`（别名→`skysense_detection`） | “提取飞机”被模型路由为**目标检测**而非语义分割 |
| E02 | `object_detection`（→`skysense_detection`） | 同上（即便指令含“分割”） |

其中 B06、D04、E01、E02 是本应成功却因工具名/路由错误而失败；G01、G03（见 9.4）虽也
`rejected_tool`，但属于**正确的 fail-closed**，不算失败。

### 9.2 解析失败（3 条）→ `unparsed`

| 任务 | 现象 | 根因 |
|---|---|---|
| B04（英文） | 模型在 `<think>` 里推理过长，`T_call` 未在 1024 token 内落地；或输出 `<T_call(...)></T_call>`（无空格变体）未被归一化器覆盖 | 英文触发更长 CoT；解析器对 `<T_call(...)></T_call>` 无空格形态存在归一化缺口 |
| E03（英文） | 同上，think 过长未输出 T_call | 英文长推理 |
| E05 | `<T_call>object_detection(...)</T_call>` 参数缺逗号 + 工具非法 | 输出格式不规范 + 检测路由 |

### 9.3 合法工具但参数被 Harness 拒绝（2 条）→ `reject_args`

| 任务 | 现象 | 根因 |
|---|---|---|
| C06「只提取左边的建筑」 | `referring_expression_segmentation(prompt="左侧的建筑")` | 模型把空间关系**写进 referring prompt**，而 Harness 的 `target_from_label` 只接受六类目标名，侧向由 Harness 确定性裁切 |
| E04「提取右侧的飞机」 | `referring_expression_segmentation(prompt="右侧的飞机")` | 同上 |

这两条模型都调用了**允许**的工具，失败发生在 Harness 校验层——正是 Harness 与基础 Agent 的差异点。

### 9.4 密集任务被文字回答（1 条）→ `dense_answered_text`

| 任务 | 现象 | 根因 |
|---|---|---|
| C07「提取图中右侧的建筑像素」 | 返回“请提供包含建筑元素的图像” | 模型对 HKUST 小图（226×234）产生幻觉，误判“无建筑” |

### 9.5 正确的 fail-closed（6 条，不计入失败）

F01/F02 由确定性指代规则拦截；F03（“中央的目标”无类别）、G02（双类别）、G01（车辆）、
G03（轮廓）分别被 Harness 参数校验或非法工具检查拦截。这 6 条全部符合预期，
支撑“需澄清任务识别率 100%”。

---

## 10. 基础 Agent 与 Harness 对照

| 基础 Agent（RemoteAgent 原始动作） | 数量 | Harness 结果 | 数量 |
|---|---|---|---|
| 直接回答 | 9 | 场景回答（成功） | 8 |
| 合法工具调用 | 20 | 通过并生成分割计划 | 16 |
| 非法工具调用 | 6 | 被拦截（fail-closed） | 6 |
| 解析失败 | 3 | 被拦截 | 3 |
| 确定性预检拦截（未调用模型） | 2 | 直接澄清 | 2 |
| — | — | 合法工具但参数被拒 | 4 |
| — | — | 密集任务文字回答 | 1 |

**结论**：基础 Agent 的 26 次工具调用中只有 20 次合法，且其中还有 4 次参数不满足契约；
Harness 共拦截/校正 **16 条**（6 非法工具 + 3 解析失败 + 4 参数拒绝 + 1 文字回答 + 2 确定性澄清），
最终只有 16 条进入执行。这就是“RemoteAgent 负责决策、Harness 负责校验”边界的量化证据。

---

## 11. 已知问题与未解决项

1. **工具命名漂移**：模型对建筑类任务偶发输出 `crossearth_semantic_segmentation`。
   实测非确定（同一条 B01 在探针中输出 crossearth、正式记录中输出 `semantic_segmentation`），
   尽管 `temperature=0`。候选改进：在 GeoScope 契约里显式加一句“不得调用
   `crossearth_semantic_segmentation`”。
2. **飞机类别弱**：6 条飞机任务仅 1 条成功，模型强烈把“飞机”路由为目标检测（SkySense）。
   候选改进：在契约中为 aircraft 增加“用 `semantic_segmentation` 而非 `object_detection`”的显式约束。
3. **英文长推理**：英文指令触发更长的 `<think>`，在 1024 token 内未落地 `T_call`。
   候选改进：提高 `max_tokens`，或提示“先输出动作、后解释”。
4. **解析归一化缺口**：`<T_call(...)></T_call>`（无空格）未被 `_normalize_single_tool_call_markup`
   覆盖，属 Harness 侧可补的一处解析。
5. **小图幻觉**：HKUST 小图上模型误判“无建筑”。

以上均为模型/Harness 边界上的**可解释、可分类**问题，未在评测脚本中人为规避；
如需修复，属“影响统计可信性”范畴，应与队长确认后在 Harness 契约或解析器统一处理，
不改动本评测的判定口径。

---

## 12. 合规与脱敏

- 未提交模型权重、`.env`、令牌、密钥、实验室内网 IP（结果中仅含本机回环 `127.0.0.1`）。
- 影像二进制未提交（`evaluation/images/` 已加入 `.gitignore`），由 `prepare_images.py` 重建。
- 原始记录中的 `<think>` 推理在 `decision_record` 中已替换为占位符；`raw_response` 保留原文
  以便复算与复核，但均为模型生成文本，不含第三方受限数据。
- 真实模型结果与确定性拦截在 `agent_status`（`n/a`）与 `outcome`（`deterministic_clarify`）字段
  明确分离，未混入同一统计。
