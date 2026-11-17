## 改动和对应任务

为队员 A 增加研究区统计、连通域分布、蒙版差异及 VerifiedPlan 后端，并接入真实 RemoteCLIP 图文检索。用户可通过显式 image_id/result_id 对已有蒙版执行受控分析，查看逐步产物与失败步骤；检索候选直接打开旧 session。

新增 `geoscope_tools/`、`services/remoteclip/`、`national_server.py`、合同、测试和证据脚本。原 `server.py`、`agent_bridge.py`、`v07_contract.py`、`web/` 及 RemoteSAM 未修改；启动 `national_server.py` 才开放 `/api/national/`。协议见 `contracts/api_contract_national.md`。

## 实际验证

- 原基线 69 项通过；新版本隔离环境加载真实权重后 95 项全部通过、无跳过。命令与环境见 A_delivery_report.md；原始输出见 docs/evidence/tests_real_weights.txt。
- RemoteCLIP RN50 真实 CPU 推理，权重 SHA256 核验；查询机场文本两次排序/分数一致，并通过旧 session API 打开第一候选。原始 JSON、索引 manifest 和启动日志在 docs/evidence/。
- `python scripts/national_evidence.py` 复现 ROI 双分母、demo 计划与真实错误响应。demo/合成样例明确标注；原 Agent/SAM 的部分工程回归使用显式 fake 服务，不作为真实模型证据。
- 编译及 `git diff --check` 通过。未提交权重、图库、索引、虚拟环境、凭据。

## 未解决的问题

当前机器未配置真实 Agent/SAM，检索后重新分割需在部署环境验收。AFR 待队长提供适配器，地理测量保持不可用；B 需按合同接 UI。旧结果的历史原图哈希不可追认，新统计范围不会恢复旧 ROI 裁掉的目标。多工具自然语言生成器未新增，手动结构化计划已可运行。

回退：停止 national_server.py 并运行原 server.py；旧接口和存量产物不变。仅在本地开发分支提交，尚未推送或合并。
