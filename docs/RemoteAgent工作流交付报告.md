# RemoteAgent 工作流交付报告

更新时间：2026-09-22。本文已按最终整合分支更新。

## 已完成

- 工作台模型模式已从旧的自定义 JSON 规划器切换为官方 `RemoteAgent.plan(...)`。
- Agent 只负责理解请求和选择工具；Harness 独占工具执行、二值蒙版校验、尺寸校验、空间裁切、统计、版本和产物保存。
- 当前工具白名单开放 `referring_expression_segmentation` 和单类别 `semantic_segmentation`，其他工具调用失败关闭。
- Agent 给出的图片路径会被绑定为当前实验影像，不能重定向到任意本地文件。
- RemoteSAM 返回的完整蒙版由 Harness 保留；回传 Agent 时删除 Base64，只提供请求摘要、服务元数据、确定性测量和产物哈希。
- Agent 二次响应通过 `continue_with_tool_result(...)` 形成 `plan → tool → result → answer` 闭环。
- vLLM 客户端改为 Python 标准库实现，不依赖 `openai` 包，也不会要求关闭 TLS 校验。

## 运行配置

```env
GEO_AGENT_BASE_URL=http://127.0.0.1:8000/v1
GEO_AGENT_MODEL=RemoteAgent
GEO_AGENT_API_KEY=
GEO_REMOTESAM_URL=http://<REMOTESAM_HOST>:6657/predict
GEO_REMOTESAM_REVISION=<RemoteSAM权重SHA256或版本号>
GEO_SERVICE_PROXY_MODE=direct
```

`GEO_AGENT_BASE_URL` 可带或不带 `/v1`。模型服务和 RemoteSAM 默认绕过下载代理，避免实验室机的 `127.0.0.1:7890` 代理污染局域网服务调用。

## 验证结果

- 最终整合代码本机全量测试：50/50 通过，其中包含 A 的无额外依赖 vLLM 客户端、正式 Agent 决策/反馈合同、场景理解/像素任务分流、非目标区域取反、官方模型多种单调用包装的兼容测试、多个工具调用拒绝和工作台闭环测试。
- 机场样例连续修改已现场通过：在全图飞机结果上输入“改成左侧的飞机”，Agent 返回单一语义分割决策，Harness 复用同一完整蒙版并裁切左半幅，得到 22,804 px、占整图 3.5631%；未重复调用 RemoteSAM。
- 官方 `AIMGroup/RemoteAgent` 已在实验室 GPU0 通过 vLLM 0.15.1 运行；`/v1/models` 返回模型 id `RemoteAgent`，上下文长度 4096。真实图像描述返回 `completed`，真实建筑提取请求返回已校验的 `semantic_segmentation` 工具调用。
- 2026-09-22 完整现场闭环通过：中文“提取右侧建筑并计算面积占比”得到单一 `semantic_segmentation` 决策，RemoteSAM 返回 133,780 px 完整建筑蒙版，Harness 裁切右半幅后得到 74,476 px、占整图 15.5158%，Agent 结果回填状态为 `completed`。完整蒙版 SHA-256 为 `f5e4f80b4857a2408dcbeef3820c720ea8447e01052cee2066d896f9315b4dc9`。
- 队员A后续提交 `721a5d1` 已按当前架构审查整合：保留其权重分片清单和严格结果回填约束；其旧版编排代码未覆盖主分支的新任务路由。队员B重启后的 RemoteSAM 服务已完成 `/ready` 和真实影像 `/predict` 复核；内网地址不进入评审材料。
- 实验室机 `/home/gst/projects/geoscope-remoteagent`：9/9 通过，命令通过 `CUDA_VISIBLE_DEVICES=0` 固定 GPU0 环境。
- 早前真实 RemoteSAM 真图验证：800×600 输入，返回同尺寸 0/255 二值 PNG；前景 52,637 px，占比约 10.966%；服务耗时约 574.53 ms，峰值显存约 1836.53 MiB；检查点 SHA-256 为 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`。
- 2026-09-20 21:44 的一次联调曾连接超时，代码正确返回失败且未生成伪蒙版或伪统计。2026-09-21 队长侧复验时 B 服务已经恢复：`/ready` 返回版本 1.1.0、CUDA 0、FP16 与同一权重哈希；使用 `web/assets/urban.png` 和 `buildings` 再次真图调用成功，800×600 输入与蒙版完全对齐、掩膜值仅为 0/255、前景 52,637 px。B 模块已经具备最终联调条件；启动官方 Agent 后用下列探针验证整条真实链路：

```bash
CUDA_VISIBLE_DEVICES=0 python -B tests/live_workbench_probe.py \
  --image web/assets/urban.png \
  --output-dir experiments/workbench-live \
  --agent-url http://127.0.0.1:8000/v1 \
  --agent-model "<从 /v1/models 复制的真实名称>" \
  --sam-url http://<REMOTESAM_HOST>:6657/predict \
  --sam-revision "<RemoteSAM权重SHA256或固定版本>"
```

## 现场验收结果

两个模型分别验证后，队长电脑通过自己的 SSH 转发运行了 `live_workbench_probe.py`。第一次模型输出两个工具，Harness 正确失败关闭；强化“只调用一次整图分割”的运行时合同后，全图英文任务和右半幅中文任务均完成完整闭环。详细运行记录见 `完整现场联调验收_2026-09-22.md`。

`GEO_AGENT_MODEL` 已确认使用 `RemoteAgent`，`GEO_REMOTESAM_REVISION` 使用正式权重 SHA-256 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`。权重不得进入 Git。A 的部署报告明确记录安装期间曾暴露 ModelScope 令牌；仓库扫描未发现令牌内容，但原令牌仍应立即作废并重新生成。
