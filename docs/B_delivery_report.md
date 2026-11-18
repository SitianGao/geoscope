# GeoScope 队员 B 国赛交付报告

交付日期为 2026 年 11 月 18 日。B 的前端交付分支为 `feature/national-workbench`，功能提交为 `9bbc70e` 和 `8e022c2`。服务器复核以三方集成提交 `1d0a138` 为准，接口为 GeoScope National API v1 的 2026 年 11 月 18 日联调补充。 [工作台分支](https://github.com/SitianGao/geoscope/tree/feature/national-workbench) · [集成 PR #9](https://github.com/SitianGao/geoscope/pull/9) · [接口合同](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/contracts/api_contract_national.md)

## 已交付功能

- 国赛入口 `/?workbench=national` 支持影像上传和样例切换、任务输入、ROI、版本选择、同图蒙版对照、差异图例、分歧复核和导出。首次使用可从首页点击「国赛工作台」。
- 结果区显示目标像素面积、整图占比、研究区内占比及各自分母。连通域分析展示候选数、面积分布、中心点和外接框。统计、图层、版本元数据和报告始终跟随所选结果 ID。
- 执行面板按保存的事件展示实际步骤、工具输入、状态、已记录的耗时和产物。RemoteCLIP 页面显示候选、原始相似度和模型及索引版本，支持打开候选影像。
- 实验页展示 A0–A2 基线与 A3–A5 的六组实测配置，A3–A5 合计 60 条逐图记录。每组可追溯到样本数、配置、服务推理耗时及对应 CSV。所有配置采用固定飞机历史诊断集，参数及独立泛化验证仍待新增数据。 [前端代码](../web/national.js) · [A3–A5 页面数据](../web/data/national-experiments.json) · [服务器消融报告](https://github.com/SitianGao/geoscope/blob/c1ef2259bb2148be93ace13d55cf1227540cb554/evaluation/national_final/results/lab_runs/20261117_ablation/README.md)

## 启动与复核

在 PR #9 的集成工作树中安装 `requirements.txt`，运行 `python national_server.py`，打开 `http://127.0.0.1:4180/?workbench=national`。本地可用内置样例复核前端，真实模型链路使用团队已有部署与索引。服务器演示前由部署负责人检查 SAM、Agent、CLIP 和工作台的就绪状态。初赛入口使用原 `python server.py` 与 `/`。 [服务交接](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/A_captain_handoff.md) · [演示脚本](demo_script.md) · [状态矩阵](ui_state_matrix.md)

P1529 的真实固定验收会话为 `2b2acb90b45d`，整图版本为 `1e9614581f1c`，AFR 版本为 `37cfda0a9fc3`。会话 ID 和两个版本 ID 由服务器当前存储保留。

| 接口 | 实际验收结果 | 证据 |
| --- | --- | --- |
| `GET /api/national/retrieval/status` 与 `POST /api/national/retrieval/search` | 真实 RemoteCLIP 检索成功，选择候选后打开对应影像会话 | [检索响应](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_query.json) |
| `POST /api/national/plans/execute` | 计划在 P1529 AFR 版本完成，实际步骤保存到结果，页面可回看 | [完整调用](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/full_chain_p1529.json) |
| `POST /api/national/tools/mask-compare` | 基准与对照 ID 对齐，新增 16,753 像素、减少 2,135,139 像素，浏览器结果与后端复算一致。一致性 IoU 约 0.6348 | [真实对照截图](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_version_compare.png) |
| 绑定版本的分歧图、概率图与 `GET /api/export/{session_id}/{run_id}` | P1529 AFR 有 6 个局部推理区域和 6 个待复核区域。ZIP 包含原图、蒙版、叠图、统计、日志及报告，另有 `afr.json`、`probability_map.png`、`disagreement.png` | [真实分歧截图](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_disagreement.png) · [浏览器验收清单](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/real_ui.json) |

## 验收记录

| 验收项 | 结果与证据 |
| --- | --- |
| B01–B03 页面、上传、样例、ROI | 新旧页面已通过回归，上传与 ROI 操作可用 |
| B04–B05 两版对照、差异层 | 服务器真实 P1529 版本对照通过，浏览器和后端逐像素一致 |
| B06–B08 统计及版本、异常状态 | 本地验证了分别显示整图与 ROI 分母、版本切换、无效 ROI、空蒙版和失败提示 |
| B09 执行过程 | 已保存的 VerifiedPlan 步骤可在页面回看，错误路径显示失败或停止，已有版本可继续打开 |
| B10–B11 CLIP 检索与未就绪 | 服务器浏览器完成真实检索和选图，本地复核了未就绪提示和继续使用样例库的路径 |
| B12 实验页 | 9 行实验配置，包含 A0–A2 三个基线和 A3–A5 六组真实配置，数据来源及逐图链接可核对。 [实验页截图](screenshots/national-experiments.png) |
| B13 报告及导出 | 本地 ROI ZIP 与所选版本 ID 一致，服务器验收已检查 AFR 完整附件 |
| B14 连续操作录像 | 已保存集成版的本地工作台彩排录像，正式提交视频待团队按脚本录制 |

队员 B 前端分支的 69 项 Python 测试及三套浏览器测试通过。PR #9 的服务器日志记录 159 项测试全部通过。此次 Windows 集成版复验中，`unittest` 发现 159 项，154 项通过，4 项 Linux 进程归属用例和 1 项需配置真实 RemoteCLIP 权重的用例按条件跳过。集成版另通过 `tests/national_browser.cjs`、`tests/browser.cjs` 和 `tests/v07_browser.cjs`。 [服务器测试输出](https://github.com/SitianGao/geoscope/blob/1d0a138bbe9355e98761d9f17fe49db7646748d6/docs/evidence/server/tests_server_all.txt) · [本地对照截图](screenshots/national-demo-difference.png)

本地录像和导出样例存放在工作台仓库的 `test-results/national-delivery-1d0a138/`。主要文件为 20 秒的本地样例彩排 `workbench-rehearsal.webm`、`roi-experiment.zip`、`roi-report.md` 和 `07-afr-archive-review.png`。本机连接验收服务器时仍缺少已核验的 SSH 主机密钥，正式演示由已有访问权限的成员打开已部署的会话。

队长签收与正式答辩视频提交待完成。VerifiedPlan 采用手动结构化计划，地理面积测量在现版标为不可用。展示候选数量时使用连通域口径，展示一致性 IoU 时说明它衡量预测之间的一致程度。
