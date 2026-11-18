# GeoScope National API v1（队员 A 实现合同，待集成审查）

## 2026-11-18 联调补充（优先于下方早期说明）

已合入队长 `c1ef225` 和 B `8e022c2`，原始算法及 B 前端原样保留。现在 `national_server.py` 复用队长版 server。推荐使用 `scripts/start_national.py` 一起启动真实 CLIP 与工作台。

- `tools/registry.data.tools` 为 B 所需的 `{name,available,reason,input_schema,...}` 数组；旧名称映射移至 `data.tools_by_name`。动态报告 AFR 适配器与 CLIP 配置状态；CLIP 真实模型/索引就绪情况仍以 retrieval/status 为准。
- `tools/mask-compare` 额外兼容 B 的 `{image_id,base_result_id,target_result_id}`。仅从该影像已保存的目标版本推导 target；仍验证两个版本目标一致，不允许越过绑定。旧 result_id/target 形式继续支持；两种结果 ID 矛盾时拒绝。
- `retrieval/status.data` 补充 model、index_size、max_top_k；候选补充真实 session 的 name、thumbnail_url。B 页面直接消费统一响应 data。
- `plans/execute` 将实际执行事件保存到对应 run 的 `tool_execution={image_id,result_id,status,failed_step,steps}`，steps 含实际 inputs/outputs；保存 session/result JSON。B 重开版本即可展示步骤。会话忙时拒绝，不覆盖已有结果或蒙版。
- `tools/adaptive-refinement` 已接队长原 run_task/Harness，通过明确的 parent_run_id、新结果 ID、live/adaptive/force_perception 参数运行。调用前检查已有版本、范围及真实 Agent/SAM 配置；不改算法策略。AFR 回退时保留服务器结果历史并返回 not_ready，绝不冒充 AFR 完成。真实部署运行仍需服务地址。
- 新入口允许读取绑定 run 的 `disagreement.png`、`probability_map.png`，修复原 handler 静态文件白名单产生的 404。无绑定版本/无文件仍返回 404；不开放任意文件。
- 新入口的原 `/api/export/{sid}/{rid}` 保留旧文件并按实际存在情况补入 full_mask、afr.json、probability_map.png 和 disagreement.png，均从该绑定结果目录取出。

实际真实 CLIP→B 页面选图→明确 demo 分割→VerifiedPlan 步骤展示已通过无网络拦截的浏览器验收，证据见 docs/evidence/team_live_browser.txt。未把 demo 分割写作真实 SAM 推理。

实际基线为仓库根目录 v0.7，不存在 `geoscope-v07-integration/` 子目录。通过 `python national_server.py` 启用全部新增入口；`python server.py` 原行为不变。原有 `server.py`、`agent_bridge.py`、`v07_contract.py`、`web/`、RemoteSAM 和队长算法目录均未修改。本合同记录已实现协议，不代表队长已签收。

## 统一响应

`success: boolean`、`request_id: string`、`image_id/result_id: string|null`、`status: completed|failed|not_ready`、`data: object`、`error: null|{code,message}`、`meta: {source,tool_version,real_execution}`。HTTP 成功 200；非法请求/工具失败 400；依赖或服务未就绪 503。失败时 success=false。计划中途失败的 `data.artifacts` 仍含前序实际产物。

`meta.real_execution` 表示本次工具是否完成实际计算；`data.model_execution` 独立说明输入是否来自 live 模型。基于 demo 蒙版的确定性统计是真实计算，但不能作为模型预测证据。查询错误时该字段为 false。注册表/状态查询的 true 仅表示实际查询，不表示执行过模型推理。

## 身份、坐标与蒙版

`image_id` 映射 v0.7 `session_id`；`result_id` 映射该 session 内的 `run_id`；均为 12 位小写十六进制 ID。不得传文件路径。`target` 必填，必须与保存结果一致。

默认计算基于**保存的最终 mask.png**，保留该版本已有的反选及半幅/ROI 限制。新增 `scope` 进一步与此蒙版相交；不能恢复已被旧版本裁掉的目标。返回 `saved_scope/saved_roi/mask_basis` 表明这一点。`scope_pixels` 仅为本次请求几何范围的面积；分母不按前景面积计算。需要整图目标时，应选整图分割版本。

原图尺寸为坐标空间；蒙版仅 0/255，尺寸不一致一律拒绝，无隐式 resize。读取 `full_mask.png` 验证已有 `full_mask_sha256`，按保存的反选和范围重算并逐像素校验最终蒙版。上游未记录历史原图哈希，本适配器报告当前原图哈希，不能据此追认旧原图从未被外部更换；应保持 experiments 原图不可变。

`scope`：`all`（默认）、`left_half`、`right_half`、`top_half`、`bottom_half`、`roi`。ROI 为 `[x1,y1,x2,y2]` 整数、左上包含右下不含；严格拒绝空/逆序/越界/与半幅同时使用。奇数余量归右侧/下侧；几何范围为空时拒绝。所有方向均为像素方向。

## 路由和输入

| 方法/路由（均加 `/api/national/`） | 必填 | 可选 | 输出要点 |
|---|---|---|---|
| GET `tools/registry` | 无 | 无 | 所有工具能力、输入 JSON Schema、计划 schema |
| POST `tools/region-statistics` | image_id,result_id,target | scope,roi | target_pixels_in_scope,image_pixels,scope_pixels,ratio_of_full_image,ratio_within_scope,mask_hash |
| POST `tools/connected-regions` | 同上 | scope,roi,min_area_pixels>=1 | 8 邻域区域及面积分布、过滤前后数量、过滤理由 |
| POST `tools/mask-compare` | 同上,base_result_id | scope,roi | base/target/交/并/新增/移除像素、prediction_agreement_iou、diff_mask_ref |
| POST `tools/spatial-relations` | 同上,base_result_id | scope,roi,near_distance_pixels | 相交、外接框间欧氏距离、外接框中心的相对像素方向 |
| POST `tools/report` | image_id,result_id,target | scope,roi | 从本次实际统计和溯源生成 JSON 报告 |
| POST `tools/adaptive-refinement` | 同上 | scope,roi | missing_dependency（尚无队长 AFR 适配器） |
| POST `tools/geo-measurement` | 同上 | scope,roi | missing_dependency（暂无地理参考/单位验证） |
| POST `plans/execute` | task,image_id,result_id,target,steps | scope,roi,reference_version | events,artifacts,failed_step,error |
| GET `retrieval/status` | 无 | 无 | 真实服务和索引状态 |
| POST `retrieval/search` | query | top_k:整数1–100，默认5 | 排序候选 image_id,rank,similarity,image_sha256,session_url 与模型/索引版本 |
| GET `artifacts/{sha256}.png` | 无 | 无 | 差异 PNG：红=移除，绿=新增，黑=其余 |

输入拒绝未知字段，schema 可通过注册表读取。注册表同时保留原两种分割工具名称及 `/api/run` 入口，原 RemoteAgent 规划/执行路径不变；这两项不能通过新增工具注册表绕过原 Harness 直接执行。

版本比较须同影像、同目标、同尺寸；base 为参考，result_id 为 target。两者都空的一致性 IoU=1，单方空=0；一致性不等于准确率。连通域数是候选区域数，先范围裁剪再计算，可能被边界截断或分裂。空间关系的 `near` 是外接框距离阈值关系，不能推断真实目标边界间距离或地理方位。

## VerifiedPlan

手动结构化计划无需模型生成器即可执行。`get_segmentation` **加载明确选中的已有版本**，不会隐式生成或替换分割结果；先通过原 `/api/run` 生成结果，再传 result_id。无会话级“最新版本”隐式回退、无共享结果缓存。全计划先检查字段和步骤依赖，再顺序执行；apply_scope 与统计比较均使用实际前序蒙版。

```json
{"task":"region_comparison","image_id":"<session_id>","result_id":"<run_id>","target":"building","scope":"left_half","reference_version":"<base_run_id>","steps":["get_segmentation","apply_scope","region_statistics","compare_versions","report_generation"]}
```

task 支持 `region_comparison/region_analysis`；步骤见注册表。get_segmentation 一次，apply_scope 在统计前一次；统计/连通域/比较要求 apply_scope；比较要求 reference_version；报告要求前序 region_statistics。失败步骤号从 1 开始。POST 请求、响应及执行事件写入 `experiments/national_events/{request_id}.json`；原结果不覆盖。报告字段均来自计算而非语言模型。

## RemoteCLIP 候选选择

客户端取候选 `session_url`（即原 GET `/api/session/{image_id}`），用其 `id` 作为原 POST `/api/run` 的 session_id。检索端只返回已绑定且当前原图哈希相符的 session。`similarity` 是单位向量点积；不表示目标存在概率或像素定位。top_k 超图库时截断。查询执行不创建新的分割结果。

错误码：`unsupported_tool`、`invalid_arguments`、`missing_dependency`、`service_unavailable`、`artifact_integrity_error`、`missing_index`、`stale_index`、`invalid_embedding`、`gpu_out_of_memory`。错误不生成假结果。

可复现样例位于 `sample_responses/`、`sample_errors/`，均注明合成/demo/真实来源；仅实际真实推理证据才能用于宣称 CLIP 已运行。
