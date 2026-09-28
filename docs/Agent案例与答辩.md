# Agent 案例与答辩问答（队员 A）

> 配套 `docs/Agent评测说明.md`。以下案例的原始输出均可在
> `evaluation/results/agent_full_results_raw.jsonl` 中逐字复核。

---

## 一、3 个成功案例

### 成功 1：全图建筑提取（B01）

- **请求**：提取全图的所有建筑。
- **模型输出**：
  ```
  T_call(semantic_segmentation, "<…>/urban.png", ["building"])
  ```
- **Harness 结果**：`segment`，目标 `building`，侧向 `all`。
- **说明**：模型正确把“全图建筑提取”路由为 dense 语义分割，选用了契约内的
  `semantic_segmentation` 与单类别 `building`，一条合法调用直接进入执行。

### 成功 2：右侧建筑提取（C01）

- **请求**：提取右侧建筑。
- **模型输出**：
  ```
  T_call(semantic_segmentation, "<…>/urban.png", ["building"])
  ```
- **Harness 结果**：`segment`，目标 `building`，侧向 `right`。
- **说明**：模型只负责“整图建筑分割”，**右侧这个空间条件由 Harness 确定性裁切**——
  模型不需要理解“右半幅”，它只要输出整图 `building`。这正是把空间关系从模型能力
  剥离、交给确定性程序的价值：侧向解析正确率因此达到 100%。

### 成功 3：连续修改换目标（D03）

- **请求**：现在换成提取飞机。（上下文：上一版本为“右侧建筑”）
- **模型输出**：
  ```
  T_call(referring_expression_segmentation, "<…>/airport.png", "飞机")
  ```
- **Harness 结果**：`segment`，目标 `aircraft`，侧向 `right`（从上下文继承）。
- **说明**：“换目标、不改侧”的连续修改，模型正确切换目标类别到飞机，侧向由 Harness
  从上一版本继承为 `right`，形成“建筑→飞机”的连贯实验序列。

---

## 二、2 个失败案例

### 失败 1：飞机被路由为目标检测（E01）

- **请求**：提取图中的飞机。
- **模型输出**：
  ```
  T_call(object_detection, "<…>/airport.png", ["airplane"])
  ```
- **结果**：`rejected_tool`（`object_detection` 别名解析为 `skysense_detection`，不在允许列表）。
- **分析**：模型把“提取飞机”判断成**目标检测**而非**语义分割**，即便上下文要求像素级结果。
  这暴露了 aircraft 类别的路由偏好：模型训练数据里飞机常与目标检测（SkySense）共现。
  应对：在 Harness 契约中为 aircraft 显式约束“用 `semantic_segmentation`，不用 `object_detection`”。

### 失败 2：小图幻觉 + 文字回答（C07）

- **请求**：提取图中右侧的建筑像素。
- **模型输出**：
  ```
  <answer>请提供包含建筑元素的图像以进行右侧建筑像素的提取。</answer>
  ```
- **结果**：`dense_answered_text`（需要像素结果却只返回文字）。
- **分析**：HKUST 样例图（226×234）被模型误判为“标志/无建筑”，于是拒绝执行并索要新图。
  这是**假阴性幻觉**：影像里实际有建筑。应对：提示模型“先默认图中存在地物，按请求输出动作”。

> 另两类高频失败（英文长推理未落地 `T_call`、工具命名漂移为 `crossearth_semantic_segmentation`）
> 已在《Agent 评测说明》第 9 节量化，此处不再重复。

---

## 三、1 个被 Harness 正确拦截的案例（G02）

- **请求**：同时提取建筑和水体两个类别。
- **模型输出**：
  ```
  T_call semantic_segmentation("<…>/urban.png", ["building", "water"])
  ```
- **结果**：`reject_args`（Harness 抛出 `Clarification`：每次实验只支持一个目标类别）。
- **说明**：模型这次调用的是**合法工具**，但一次传了**两个类别**。Harness 的
  `segmentation_plan` 拒绝多类别，要求“分别提取并保存版本”。这就是“多工具/多类别调用要拒绝”
  的典型证据：合法工具 + 非法参数仍会被拦在真正的分割执行之前，保证实验的版本可追溯与
  统计口径一致。

> 同类的拦截还有：G01「提取图中的车辆」→ `skysense_detection`（非法工具）、
> G03「提取图中所有物体的轮廓」→ `contour_extraction`（非法工具）、
> F03「提取画面中央的目标」→ `referring_expression_segmentation(prompt="中央的目标")`
> （无类别，被 `target_from_label` 拒绝）。

---

## 四、答辩问答（4 问）

### Q1：为什么需要 RemoteAgent？

GeoScope 面向的是**自然语言驱动的遥感实验**，用户指令是“提取右侧建筑”“改成左侧”这类
非结构化表达。传统的固定脚本需要用户先理解工具与参数，无法直接把一句话变成一条实验。
RemoteAgent（Qwen2.5-VL）作为多模态认知核心，负责理解影像 + 解析指令，把自然语言路由到
“内部问答”或“调用分割工具”两个动作空间——评测中 40 条任务里 30 条被正确路由（75%），
其中 16 条合法分割调用全部目标/侧向正确。它解决的是“从语言到动作”的认知层，而不是替代
下游的精确分割模型。

### Q2：Agent 与 Harness 的边界是什么？

一句话：**Agent 做决策，Harness 做校验与执行**。

- RemoteAgent 只输出“回答”或“调哪个工具、带什么参数”，**从不真正执行**分割，也不掌握
  面积、占比等确定量；
- Harness（`server.py` + `agent_bridge.py`）负责三件事：①确定性解析像素范围（左/右/上/下/全图）、
  ②校验工具与参数合法性（允许列表、单类别、影像路径一致）、③执行分割并做几何/面积统计。
- 边界落在 `segmentation_plan`：模型输出要么在这里变成可执行计划，要么被抛
  `Clarification`/`ValueError` 拦下。评测里 Harness 共拦截/校正 16 条，把 26 次工具调用
  收敛为 16 次可靠执行。

### Q3：为什么多工具调用要拒绝？

多工具/多类别调用意味着**一次实验混入多个不可追溯的结果**：面积、占比、版本都无法定位到
单一目标与单一范围，无法复算，也无法支撑“左右条件修改 → 保存版本”的对比流程。
因此契约规定**每次实验只允许一个目标类别、一次调用**；模型若一次给两个类别（G02）或调用
非允许工具（G01/G03），Harness 一律 fail-closed。这不是能力不足，而是**实验可信性的硬约束**：
宁可拒绝让用户澄清，也不生成口径不清的统计。

### Q4：基础 Agent 与完整 GeoScope 有什么差异？

基础 Agent = RemoteAgent 的原始动作：40 条里有 20 次合法工具调用、6 次非法工具、3 次解析失败，
还有 4 次合法工具但参数不合契约。**它只“想做什么”，不保证“能做对”**。

完整 GeoScope = 基础 Agent + Harness 校验 + RemoteSAM 执行 + 几何/面积统计 + 版本管理：

- 6 次非法工具、3 次解析失败、4 次参数错误、1 次文字回答共 **16 条被拦截或校正**，
  最终只有 16 条进入真正执行；
- 空间范围、面积、占比由确定性程序计算，不依赖模型“说一个数字”；
- 结果可追溯（目标、侧向、版本、原始决策 SHA-256 都落盘）。

量化结论：合法工具调用率 76.9%，经 Harness 后 100% 的通过项目标/侧向正确，
这正是“认知核心 + 校验执行器”分离架构相对“裸调 Agent”的可信度收益。

---

## 五、配合队长复算

所有数字均可复算：

```bash
python evaluation/agent_eval.py --from-raw agent_full_results_raw.jsonl \
    --out agent_full_results.csv --summary agent_summary.json
```

该命令**不调用模型**，从冻结的 `agent_full_results_raw.jsonl` 重新生成
`agent_full_results.csv`、`agent_summary.json` 与 baseline-vs-harness 对照表，
保证答辩中引用的每个数字都能定位到原始任务记录。
