## 改动和对应任务

补齐队员 A 后端与队友正式交付接口的差异，并在服务器完成真实 CLIP→Agent/SAM→AFR→统计/比较/导出的闭环。A 分支已合入队长 c1ef225 与 B 的 8e022c2，保留原提交归属；main 未修改。

A 的新增改动包括：B 的 target_result_id 参数兼容、数组工具注册表、CLIP 缩略图与状态字段、保存 tool_execution.steps、队长 AFR 的 Harness 适配、绑定版本的概率/分歧图读取、导出 AFR 附件。A 未重写 B 的前端或队长算法。接口见 contracts/api_contract_national.md 的 2026-11-18 补充。

增加 Python 3.10 兼容哈希实现，去除旧 OpenCLIP transform repr 的随机地址以稳定 embedding 版本；提供一体启动入口与无 mock 的服务器验收脚本。因服务器旧 vLLM 环境已不存在，增加已有 RemoteAgent 的可选 Transformers serving 入口，在独立环境中使用用户已提供的同一官方模型权重，不修改原 Harness。

## 实际验证

- 服务器 GPU0 真实运行三模型，GPU1 未使用。CLIP 为 ViT-B-32 / OpenCLIP 2.24.0 / Torch 2.4.1+cu121；Agent 为官方四分片检查点 / Transformers 4.57.1，权重和运行时 manifest 均在证据目录。
- 最终服务器 unittest 回归 155 项全部通过，无跳过（23.301s）；原始输出 docs/evidence/server/tests_server_all.txt。部分单元测试显式使用 test doubles；它们与真实集成证据分开保存。
- 4000×4000 P1529：真实 Agent/SAM baseline 与 AFR 新版本，实际 6 个局部区域推理、6 个待复核区域；Agent 最终反馈 completed，无反馈告警。不是重新计算的分割精度评测，没有宣称 IoU 提升。
- 服务器真实 Chrome：CLIP 查询和候选选择、真实 P1529 的步骤展示/版本对照/分歧图/导出均通过；浏览器逐像素差异与后端一致。无网络拦截和伪造分数。截图、完整请求响应、权重 SHA256、模型/索引 manifest 在 docs/evidence/server/。

## 已知边界与回退

地理测量仍明确不可用；没有引入自然语言新计划生成器。手动结构化计划消费显式选定的真实分割版本，失败保留前序产物。候选连通域不是实例数，预测一致性不是准确率，当前原图哈希不能反证历史原图未被外部替换。

服务器服务保持运行于 A 的独立目录和独立环境，只读引用队长原环境/权重；没有修改队友原部署目录。需要回退 A 新入口时可运行原 server.py（保留队长/B 的合并代码）；不删除既有会话。A 的提交尚未推送、未合并到远端主线；队长可审查本分支。
