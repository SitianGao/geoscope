# RemoteAgent 工作流交付报告

更新时间：2026-09-21。本文已按最终整合分支更新。

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
GEO_REMOTESAM_URL=http://10.199.156.100:6657/predict
GEO_REMOTESAM_REVISION=<RemoteSAM权重SHA256或版本号>
GEO_SERVICE_PROXY_MODE=direct
```

`GEO_AGENT_BASE_URL` 可带或不带 `/v1`。模型服务和 RemoteSAM 默认绕过下载代理，避免实验室机的 `127.0.0.1:7890` 代理污染局域网服务调用。

## 验证结果

- 最终整合代码本机全量测试：43/43 通过，其中包含 A 的无额外依赖 vLLM 客户端、正式 Agent 决策/反馈合同和工作台闭环测试。
- 实验室机 `/home/gst/projects/geoscope-remoteagent`：9/9 通过，命令通过 `CUDA_VISIBLE_DEVICES=0` 固定 GPU0 环境。
- 早前真实 RemoteSAM 真图验证：800×600 输入，返回同尺寸 0/255 二值 PNG；前景 52,637 px，占比约 10.966%；服务耗时约 574.53 ms，峰值显存约 1836.53 MiB；检查点 SHA-256 为 `f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7`。
- 2026-09-20 21:44 的一次联调曾连接超时，代码正确返回失败且未生成伪蒙版或伪统计。2026-09-21 队长侧复验时 B 服务已经恢复：`/ready` 返回版本 1.1.0、CUDA 0、FP16 与同一权重哈希；使用 `web/assets/urban.png` 和 `buildings` 再次真图调用成功，800×600 输入与蒙版完全对齐、掩膜值仅为 0/255、前景 52,637 px。B 模块已经具备最终联调条件；启动官方 Agent 后用下列探针验证整条真实链路：

```bash
CUDA_VISIBLE_DEVICES=0 python -B tests/live_workbench_probe.py \
  --image web/assets/urban.png \
  --output-dir experiments/workbench-live \
  --agent-url http://127.0.0.1:8000/v1 \
  --agent-model "<从 /v1/models 复制的真实名称>" \
  --sam-url http://10.199.156.100:6657/predict \
  --sam-revision "<RemoteSAM权重SHA256或固定版本>"
```

## 仍需队长提供

官方 Agent 权重尚未在实验室机落盘，因此还不能启动真实 vLLM Agent 推理。需要队长确认：

1. ModelScope 权重的具体文件/版本或已审批的下载方式；
2. 权重目录和 `GEO_AGENT_MODEL` 实际名称；
3. `GEO_REMOTESAM_REVISION` 应记录的正式权重 SHA-256。

权重必须保存在实验室电脑，不进入 Git，也不下载到个人电脑。启动时固定 `CUDA_VISIBLE_DEVICES=0`。
