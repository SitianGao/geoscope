# 队员 A：RemoteAgent

先阅读仓库根目录 README 和 `docs/三人团队七天开发交接与AI任务说明.md` 第四节，将完整提示词交给你的 AI 编程助手。

本仓库把原 `代码/RemoteAgent-main/` 中的认知核心放在 `services/remoteagent/upstream/`。提示词里的旧目录名应映射到这里。

重点文件：

- `upstream/README.md`：上游说明，包含未纳入本项目的工具，不能直接认为它们已存在。
- `upstream/remoteagent/core/inference.py`：真实影像输入和推理循环。
- `upstream/remoteagent/eval_common.py`：多模态请求参考。
- `upstream/remoteagent/parsing/`：官方工具调用解析。
- `upstream/remoteagent/services/executor.py`：工具执行与结果返回。

本目录不提交权重。实验室主机上的 Python 3.10、PyTorch CUDA 环境、官方 RemoteAgent 权重、vLLM 服务和 RemoteSAM 真图调用均已分别验证；完整工作台现场闭环仍按部署文档推进。

在 `feature/remoteagent` 分支工作。负责本目录及相关测试、说明；根目录工作台与 Harness 的修改交由队长整合。保留上游来源，不把官方算法作为团队自研成果。

## 当前实现（2026-09-20）

认知核心工作流已完成第一轮工程化：

- `RemoteAgent.plan(...)` 接收真实影像、用户请求与实验上下文，只返回直接答案或经过校验的工具调用，不与工作台 Harness 抢占工具执行权。
- `RemoteAgent.continue_with_tool_result(...)` 可把 Harness 的真实工具结果回填给模型生成最终说明；大体积 mask 不会再次塞回模型上下文。
- 模型输出的任意本地影像路径会被绑定为当前实验影像，且只允许调用运行时显式开放的工具。
- `ServiceExecutor.execute_structured(...)` 保留 RemoteSAM 的完整结构化结果与 mask，便于工作台继续统计、版本管理和导出。
- 独立 `run(...)` 验证路径继续可用，并已改为真正发送影像内容，而不是只发送本地路径字符串。

详细接口、部署命令和联调约定见 [DEPLOYMENT.md](DEPLOYMENT.md)，真实 RemoteSAM 验收数据见 [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md)。配置模板见 [.env.example](.env.example)。

本仓库不提交模型权重、虚拟环境、密钥或缓存。官方 `AIMGroup/RemoteAgent` 权重启用了申请制下载，部署机目录固定为 `/home/gst/projects/geoscope-remoteagent/RemoteAgent`，运行时固定使用 GPU0。

## 验证

从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

当前整仓 44 项测试通过，其中 RemoteAgent 工作流专项测试 6 项，覆盖真实图片输入、工具白名单、影像路径绑定、完整 mask 保留、工具结果回填，以及官方模型实际出现的 `<answer>T_call(...)</answer>` 包装。
