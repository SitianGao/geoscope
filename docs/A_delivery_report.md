# 队员 A 最终交付报告

日期：2026-11-18。分支：`feature/national-tools-clip`，已推送到 GitHub，供队长审查和联调；不自动合并 main。快速接管见 [A_captain_handoff.md](A_captain_handoff.md)。

## 交付范围与团队集成

已合入队长算法 c1ef225 和 B 工作台 8e022c2，保留原提交归属。A 实现统计/差异/受控计划/CLIP 服务，并适配双方真实接口，没有重写队长 AFR 策略；按用户反馈对指代合同和 B 的检索页面作了有测试覆盖的修改。基线 2cb8da8 保留标签 baseline/national-before-a-20261117。

- region_statistics：目标面积、整图与范围两种分母；严格 ROI、奇数半幅和空集处理。
- connected_region_analysis：8 邻域候选区域、过滤与分布；不将连通域数称作实例数。
- mask_compare：同图同目标、无隐式 resize，交并差与预测一致性、差异图和版本绑定。
- registry/VerifiedPlan：schema/白名单/依赖预检，实际前序产物传递、事件与步骤持久化、失败保留产物和明确失败步骤。
- spatial_relation_analysis、确定性 JSON report_generation；地理测量明确不可用。
- adaptive_refinement：调用队长原 Harness/run_task，parent/result ID 明确，保留原算法和回退行为。
- RemoteCLIP：可信权重 SHA256、配套预处理/tokenizer、真实图文编码、原图哈希/模型版本绑定索引、归一化余弦 top-k、候选回到平台 session；不提供固定排序回退。
- A/B 兼容：target_result_id 参数、数组注册表、候选缩略图与就绪字段、run.tool_execution.steps。修复概率/分歧图 404，导出加入真实存在的 AFR 附件。

启动入口为 national_server.py；旧 server.py 启动仍保留。服务管理 scripts/manage_server.py 支持幂等启动、健康状态和只结束本部署进程的停止；端口冲突不接管他人进程。接口合同、机器 schema、请求响应样例、模型说明、测试和来源记录齐备。

## 服务器真实验证

用户要求后续运行在服务器，最终模型/测试/浏览器验收均在服务器执行。使用 A 的独立目录和独立 uv 环境，只读复用既有 PyTorch/SAM 环境与权重；仅 GPU0，GPU1 未使用。实际私有目录配置保存在服务器 outputs/server_runtime.json，不提交凭据或私有部署配置。

RemoteCLIP 为 ViT-B-32，权重 SHA256=`60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4`，OpenCLIP 2.24.0、Torch 2.4.1+cu121、cuda:0。图库为已绑定的城市/机场/P1529 三幅真实影像。修复 Python 3.10 哈希兼容，以及旧 OpenCLIP transform repr 随机地址影响 embedding 版本的问题；受控重启后版本稳定，重复查询一致。

用户原 RemoteAgent 四分片权重已复制至服务器并校验：729 个张量、16,584,333,312 字节数据。已实际读取 ModelScope 申请制 AIMGroup/RemoteAgent 的文件页，页面提交 3a0698c4，16.60GB；未绕过下载权限。旧 vLLM 环境已不存在，因此新增可选 Transformers serving，使用同一官方权重、匹配 config 的 Transformers 4.57.1、保存的 slow processor、BF16/SDPA、16K 上下文。原 Harness/解析路径不改；不宣称与 vLLM 性能一致。SHA256 和来源见 agent_inventory.json。

真实链路已完成：CLIP 检索候选→RemoteAgent→RemoteSAM fast→AFR 新版本→VerifiedPlan→统计/比较→分歧图→导出。P1529 为 4000×4000 真实影像，image_id=`2b2acb90b45d`，base=`1e9614581f1c`，AFR=`37cfda0a9fc3`。实际执行 6 个局部区域，产生 6 个待复核区域；Agent 反馈 completed，无反馈告警。服务器真实 Chrome 显示步骤、对照与分歧图，浏览器逐像素差异与后端一致，导出附件完整。

用户反馈的“最下面的船”已修复，原图新版本 e8e1eb4a6be7 只高亮底部船舶（120,398 px / 2.36%），旧错误版本明确告警。检索弹窗完成桌面/手机改版，保留原始精度和完整追溯。详见 A_user_feedback_fixes.md。

原始请求响应、索引/权重 manifest、真实截图和测试日志位于 docs/evidence/server/。早期依赖/版本失败及 8K 超限记录保留用于审计，均已修复；早期 demo 工程证据有明确标注，不作为真实模型证据。

## 测试与复现

最终服务器 **168 项测试全部通过，无跳过（25.160 秒）**，包含真实权重测试及 5 项生命周期保护测试；原始输出 docs/evidence/server/tests_server_all.txt。部分单元测试使用显式 test doubles，与真实全链路/浏览器证据分开。受控停止/启动和四服务健康检查均通过。

完整命令和固定版本操作顺序见 A_captain_handoff.md。重新推理会生成新 ID，应保存新的证据文件名，不覆盖固定验收记录。编译/格式检查、Git 对象完整性检查及权重/凭据排除检查已完成。

## 边界与回退

没有新增自然语言计划生成器，手动结构化计划已可运行；地理面积测量保持不可用。预测一致性不是准确率，候选连通域不是实例数；不把本次工程验收当作独立精度评测或泛化提升。新 scope 与已保存最终蒙版相交，不能恢复旧 ROI 裁掉的目标；当前原图哈希也不能反证历史原图从未被外部替换。

需要回退 A 入口时运行原 server.py；旧会话/缓存/ID/蒙版格式不变。服务管理 stop 仅作用于本部署 PID、匹配命令和工作目录，不修改模型/队友目录。当前四项服务保持就绪。

## 版本与审核

主要服务器集成提交 af7c294，部署文档提交 6c5c2ab；本次收尾再补齐生命周期接管、最新测试和交接说明。早期 Git smart HTTP 不可达时使用官方精确快照和 commit/tree SHA 恢复，随后补齐中间提交树，git fsck 检查通过；没有重写队友提交。当前 GitHub 推送正常。

后续仅需队长审查 PR 并决定合并主线。权重、虚拟环境、索引、原始图库、.env 和服务器私有配置均不在 Git。
