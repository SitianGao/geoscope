# 队员 A 最终联调交接（2026-11-18）

分支：`feature/national-tools-clip`。已包含队长算法 `c1ef225` 和 B 工作台 `8e022c2`，由 A 完成接口兼容及服务器真实联调。请整体检出该分支审查，不必再次把这两个提交 cherry-pick 一遍。主分支不由 A 自动合并。

此前 [PR #9](https://github.com/SitianGao/geoscope/pull/9) 已由队长合并。当前用户反馈修复在 [A 修复分支](https://github.com/SitianGao/geoscope/tree/feature/national-tools-clip)，另开后续 PR；不自动合并 main。

最新用户反馈已处理：检索弹窗重整桌面/手机布局，原始相似度与长编号保留在追溯详情；“提取最下面的船”通过单目标指代完整传入 SAM，不再退化成全图类别。船舶复测 image_id=`7c8d57152d4b`、新 result_id=`e8e1eb4a6be7`，真实面积 120,398 px（2.36%），仅底部船舶高亮。旧错误版本会显示需重跑告警，不覆盖原产物。截图见 docs/evidence/server/ship_reference_workbench.png、retrieval_redesign_desktop.png。

## 已部署与验收

服务器独立 A 目录中，SAM/Agent/CLIP/工作台四项服务均保持运行，只使用 GPU0。具体登录/目录配置由现有授权 SSH 及服务器私有 `outputs/server_runtime.json` 保存，不提交凭据。工作台服务器内地址：`http://127.0.0.1:4180/?workbench=national`；CLIP 4190、Agent 8000、SAM 6657，均绑定 loopback。远程浏览需使用已有 SSH 端口转发，界面在客户端显示，模型仍在服务器执行。

P1529 的真实固定验收版本：

| 字段 | 值 |
|---|---|
| image_id | `2b2acb90b45d` |
| baseline result_id | `1e9614581f1c` |
| AFR result_id | `37cfda0a9fc3` |
| 实际局部推理 / 待复核区域 | 6 / 6 |
| Agent 结果反馈 | completed，无 feedback_warning |
| 浏览器/后端版本差异 | 逐像素一致 |

该数字说明工程执行，不代表准确目标实例数或分割精度提升。真实调用与截图在 `docs/evidence/server/`；早期带 demo 标记的工程证据保留，不能混用。

## 服务接管

在 A 的实际服务器部署目录运行：

```bash
.venv/bin/python scripts/manage_server.py status
.venv/bin/python scripts/manage_server.py start
# 需要主动停止本部署时才执行：
.venv/bin/python scripts/manage_server.py stop
```

start 幂等；模型加载有短暂 not_ready，稍后 status 检查四项 owned_process/ready 都为 true。stop 仅结束本部署 PID 且命令/工作目录匹配的进程，不按端口或进程名批量结束。原环境、权重、会话均保留。日志/PID 位于 outputs，私有配置示例为 `contracts/server_runtime.example.json`，不要把实际配置加入 Git。

服务器环境：`.venv` 只读引用原 PyTorch/SAM 环境并加入 OpenCLIP；`.venv-agent` 独立使用 Transformers 4.57.1，复用 PyTorch 2.4.1+cu121。Agent 沿用同一官方权重，通过可选 Transformers serving 恢复旧接口，原 vLLM 代码路径不变。来源和实际 SHA256 见 agent_inventory.json、index_manifest.json。

## 联调顺序

1. 打开国赛入口，点击“影像库”中的 RemoteCLIP 检索，检查就绪；查询 `airport with airplanes`，选择任意实际候选。
2. 验收固定 P1529 时通过历史记录打开上述 image_id，选择两个固定版本，查看“版本对照”，点击后端复算。这样不用重复消耗模型推理时间。
3. 选择 AFR 版本，看实际执行步骤和质量诊断。分歧图/概率图现在可读取；下载实验包包含 afr.json、概率图和分歧图。
4. 需要重新推理时选模型模式，先 fast 再 adaptive；新结果 ID 应与固定验收 ID 区分，不能把旧图当新推理。

可复现接口命令（服务器运行）：

```bash
.venv/bin/python scripts/remoteclip_evidence.py
# 下条将创建新的真实分割版本，属于重新验收，不是状态检查：
.venv/bin/python scripts/server_full_chain_evidence.py --image-id 2b2acb90b45d --output-name new_full_chain.json
# 使用已保存结果验证 B 页面，不重新推理：
TMPDIR="$PWD/outputs/browser-tmp" .venv/bin/python tests/national_server_live_ui.py
```

新机器需先创建图库/session，固定 ID 不可跨全新存储复用。新计划入口 `POST /api/national/plans/execute` 支持手动结构化计划；步骤会保存到对应 run.tool_execution 并由 B 展示。完整字段合同见 contracts/api_contract_national.md。

## 验收与剩余边界

模型真实全链路及服务器真实 Chrome 验收通过；测试日志见 docs/evidence/server/tests_server_all.txt。没有增加地理面积测算或自然语言新计划生成器；地理测量显式不可用，手动 VerifiedPlan 已完成。未增加新的独立精度评测，不宣称 AFR 泛化增益；队长原评测数据和算法策略保持原样。

接口/源码/验收证据全部交付；后续是队长的 PR 审查和主线合并，不属于尚未实现的 A 功能。
