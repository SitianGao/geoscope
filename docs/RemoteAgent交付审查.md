# RemoteAgent 分支交付审查

> 历史记录：本文记录 2026-09-20 对 `efce7b8` 的问题。相关缺失文件已经由队员补交并合并；2026-09-21 的当前结论以 `队长执行与验收.md` 为准。保留本文用于说明问题如何被发现和关闭，不应再把它当作当前阻塞。

审查日期：2026-09-20
对象：`origin/feature/remoteagent`，提交 `efce7b8`。这是该提交的快照结论，不代表队员未推送的新版本。

## 已有进展

已加入 `plan`、`continue_with_tool_result`、AgentResult/ToolCall 字段、运行工具白名单和当前影像路径绑定。设计上将 Harness 与 Agent 的执行所有权分开，符合项目方向。队长暂未合并该分支。

## 需要补齐的提交

1. **导入失败：缺少 `image_data_url`。**
   `remoteagent/core/inference.py` 引用了 `remoteagent.utils.image.image_data_url`，但同一提交的 `utils/image.py` 只有 `encode_image`。把该分支独立导出后实际执行 `from remoteagent import RemoteAgent`，得到 ImportError。该问题不需要 GPU 即可复现，也不是模型权重缺失造成的。
2. **自执行路径引用未实现接口。**
   新 `_agent_loop` 调用 `ServiceExecutor.execute_structured`，但同一提交的 executor.py 没有该方法。即使修复导入，执行该路径时仍会失败。若首版只交付 plan，也应明确独立 run 尚不可用，不能报告它已验证通过。
3. **测试文件与说明不一致。**
   DEPLOYMENT.md 写到新增 5 项 Agent 流程测试和 18 项总测试，但该分支 tests 中仅有原来的 `test_harness.py` 与 `browser.cjs`。请提交相应测试文件，并从干净检出重新运行。

## 合并前需要的证据

- 补齐辅助函数、结构化执行接口或准确缩小已支持范围。
- 干净克隆中导入成功。
- Agent 专属测试进入 Git，图像输入、白名单、路径绑定和反馈结果的测试实际通过。
- 明确目标和半幅范围字段如何传给 Harness。
- 更新部署说明，区分“本人机器上已完成”“仓库已提交”和“真实模型已验证”。

远程部署、GPU 占用和权重权限信息来自 A 的部署文档，本轮没有登录实验室机器复验。队长侧不把它们当作独立确认的事实。
