# 队员 A：RemoteAgent

先阅读仓库根目录 README 和 `docs/三人团队七天开发交接与AI任务说明.md` 第四节，将完整提示词交给你的 AI 编程助手。

本仓库把原 `代码/RemoteAgent-main/` 中的认知核心放在 `services/remoteagent/upstream/`。提示词里的旧目录名应映射到这里。

重点文件：

- `upstream/README.md`：上游说明，包含未纳入本项目的工具，不能直接认为它们已存在。
- `upstream/remoteagent/core/inference.py`：真实影像输入和推理循环。
- `upstream/remoteagent/eval_common.py`：多模态请求参考。
- `upstream/remoteagent/parsing/`：官方工具调用解析。
- `upstream/remoteagent/services/executor.py`：工具执行与结果返回。

本目录没有权重，也没有已验证的推理环境。第一天确认部署条件和与队长的接口；第二天交真实看图推理和工具调用记录；第三天协助端到端联调。

在 `feature/remoteagent` 分支工作。负责本目录及相关测试、说明；根目录工作台与 Harness 的修改交由队长整合。保留上游来源，不把官方算法作为团队自研成果。
