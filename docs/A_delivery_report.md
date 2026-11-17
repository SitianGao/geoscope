# 队员 A 交付报告

日期：2026-11-17。开发分支：`feature/national-tools-clip`。基线：`2cb8da8b1293882e24bdf75f2ce70628a96a934a`，保留标签 `baseline/national-before-a-20261117`。

## 已实现

- P0：双分母研究区统计、严格 ROI 校验、8 邻域候选连通域/过滤/分布、同图同目标版本交并差及可下载差异图。
- 注册表、严格字段/schema 校验、影像/版本/目标绑定、全蒙版 SHA256 和最终蒙版重算校验；禁止任意文件路径、未知工具和未知参数。
- VerifiedPlan：整计划依赖预检、手动结构化执行、实际前序蒙版传递、逐步事件、失败步骤号与保留产物、重复执行不串版本。全部新 POST 请求和执行响应保存到 `experiments/national_events/`。
- P1：像素坐标空间关系、确定性 JSON 报告；AFR 仅保留可读的未就绪能力，等待队长模块。地理测量未实现并明确不可用。
- RemoteCLIP：可信权重校验、官方 OpenCLIP 预处理/tokenizer、真实图文编码、哈希与版本绑定的持久索引、归一化余弦 top-k、候选回到旧 session API、失效索引/服务断开/OOM 错误处理。
- 独立 `national_server.py` 新入口复用原工作台，原主程序和前端文件均不修改。

## 工程与复现

Git smart HTTP 端点多次连接失败，使用官方 codeload 源码和官方 API 提交信息恢复**精确的浅仓库**，逐一核验 Git tree SHA 与 commit SHA；不是新建一个不相关的初始提交。保留原 origin，Git connectivity 检查通过。完整历史未下载；网络恢复后可 `git fetch --unshallow origin`。

基础依赖仍只有原 requirements.txt 的 Pillow。模型依赖隔离在 `.venv-remoteclip/`；安装版本快照见 `services/remoteclip/requirements-tested.txt`。CPU 可执行真实推理，无需 GPU；CUDA 部署需安装匹配 PyTorch。

启动工具：`python national_server.py`，默认本机 4180。CLIP 的权重下载、图库创建、服务启动及环境变量配置见 [RemoteCLIP README](../services/remoteclip/README.md)。接口与字段详见 [合同](../contracts/api_contract_national.md)，机器可读 schema 为 `contracts/national_schema.json`，与运行注册表一致。

纯计算/主链路回归：

```powershell
python -m unittest discover -s tests -v
python scripts/national_evidence.py
python -m compileall -q geoscope_tools services/remoteclip national_server.py
git diff --check
```

原版基线 69 项全部通过。开发后回归 95 项：94 项通过、1 项真实权重测试明确跳过（未设置权重时）。完整记录：`docs/evidence/tests_all.txt`。新增覆盖 ROI 算术、非法 ROI、奇数半幅、空/满蒙版、连通域裁剪分裂、交并差、跨影像/目标拒绝、损坏/缺失文件、非法工具/schema、执行依赖/失败隔离、重复执行、旧 HTTP/导出兼容及 CLIP 空状态。显式 fake 模型测试只证明工程行为，不代表真实 RemoteAgent/RemoteSAM 推理通过。

最终在隔离环境设置真实 RN50 权重后，完整 **95 项测试全部通过，无跳过**（22.899 秒）；见 `docs/evidence/tests_real_weights.txt`。此前的跳过记录保留用于审计，不代表最终验收状态。编译检查和 Git whitespace 检查通过。

## 真实 RemoteCLIP 证据

已实际下载 408,397,807 字节官方 RN50 权重；通过镜像传输，SHA256 与官方 LFS 指针完全相同：`11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9`。运行于 CPU，OpenCLIP 4.2.2 / PyTorch 2.15.0+cpu。真实服务启动日志、模型预处理/tokenizer、索引 manifest、两次查询与候选选择响应均在 `docs/evidence/`。

查询 `airport with airplanes`：机场样例 `8e1939ccc3e1` 排名 1，余弦相似度 `0.24212190508842468`；城市样例 `b273fa9ac2d6` 排名 2，余弦相似度 `0.13405519723892212`。两次实际查询排序和分数完全相同。首项通过旧 `/api/session/8e1939ccc3e1` 打开且 ID 一致。这只证明检索与选择可运行，不代表普遍检索精度、目标存在概率或像素分割完成。

本次 embedding_version：`6165b7a2d369dce42a0dba628aed21c01e8ca1c5ea64343225384a9a12b25932`；index_version：`c2f38e9e37fc8546e653566317754937bbf0dba850bf748810703fef9db758a0`。

在当前本地图库上复现（脚本会自动启停临时服务，不留下后台进程）：

```powershell
.venv-remoteclip/Scripts/python.exe scripts/run_remoteclip_acceptance.py
```

真实权重全测试：设置 `REMOTECLIP_TEST_WEIGHTS=weights/RemoteCLIP-RN50.pt`、`REMOTECLIP_TEST_SHA256` 为上述哈希、`REMOTECLIP_TEST_ARCH=RN50`，再用 `.venv-remoteclip/Scripts/python.exe -m unittest discover -s tests -v`。新机器先按照 RemoteCLIP README 创建图库；重新创建的 session ID 和索引版本会变化，应以新实际输出为准。

## 验收对照与边界

| 编号 | 证据/状态 |
|---|---|
| A01 | 原有 69 项回归，新增 HTTP 验证 session/run/export；本机未连接真实 Agent/SAM |
| A02–A07 | statistics/regions/compare/planner/http 测试与 contracts 样例 |
| A08–A09 | demo_plan.json、demo_failed_step.json；结构化步骤及第 4 步失败、前序产物保留 |
| A10 | CPU 真实 RN50 推理和归一化检索通过，remoteclip_startup.txt / remoteclip_real_query.json |
| A11 | 真实检索候选已经原 session API 打开；后续真实分割需外部 Agent/SAM 服务 |
| A12 | 实际 HTTP 503 not_ready 负例；旧主链路回归通过 |
| A13 | 工具重复执行断言及真实 CLIP 两次查询精确一致 |
| A14 | provenance、模型/索引版本、原始 JSON、环境版本与源码 |

尚未完成/依赖外部：队长 AFR 核心适配器；有地理参考的真实面积测量；B 前端接入；队长合并签收；真实 Agent/SAM 在目标部署环境下的全链路重跑。VerifiedPlan 消费显式选定的已有分割版本；自然语言生成新计划不在本次后端实现内。前端选择 CLIP 候选可直接使用返回的 session_url 及 image_id 调用旧 API。

无真值标签时只输出预测一致性，不宣称精度提升。候选连通域不是实例数；像素单位不转平方米。新工具对已保存最终蒙版继续限制范围，不能恢复旧 ROI 裁掉的目标。历史 session 未保存原图历史哈希；报告中的 image_sha256 是当前原图的哈希，不能反证历史原图未被外部更换。

已核查失败例：越界/空/非法 ROI，概率蒙版，影像与版本不匹配，目标不匹配，全蒙版校验失败、最终蒙版损坏、文件缺失，非法工具/参数，依赖顺序错误，比较步骤缺版本，CLIP 未配置/索引缺失/图片变化。GPU OOM 有明确错误处理，本机 CPU 环境未实际制造 GPU OOM。

## 回退与交付

停止新启动入口后运行原 `python server.py` 即回到原路由集合；不用删除任何旧数据。新产物位于 `experiments/national_artifacts/`、`experiments/national_events/` 与 `outputs/`。不修改旧会话 ID、缓存键、蒙版格式和模型权重。无需把新功能合并进 server.py 才能审查；队长可选择独立启动或自行审核挂载。

本地提交 hash 会在实现提交完成后记录；不会自动推送或合并。权重、虚拟环境、索引与原始图库均排除提交；交付源码、合同、测试及小体积真实性证据。
